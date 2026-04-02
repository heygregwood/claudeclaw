import { mkdir, readFile, writeFile } from "fs/promises";
import { join } from "path";
import { existsSync } from "fs";
import { getSession, createSession, incrementTurn, markCompactWarned } from "./sessions";
import { getSettings, type ModelConfig, type SecurityConfig } from "./config";
import { buildClockPromptPrefix } from "./timezone";
import { selectModel } from "./model-router";

const LOGS_DIR = join(process.cwd(), ".claude/claudeclaw/logs");
const MEMORY_DIR = join(process.cwd(), ".claude", "claudeclaw", "memory");
const CORRECTIONS_FILE = join(process.cwd(), ".claude", "claudeclaw", "corrections.jsonl");
// Resolve prompts relative to the claudeclaw installation, not the project dir
const PROMPTS_DIR = join(import.meta.dir, "..", "prompts");
const HEARTBEAT_PROMPT_FILE = join(PROMPTS_DIR, "heartbeat", "HEARTBEAT.md");
// Project-level prompt overrides live here (gitignored, user-owned)
const PROJECT_PROMPTS_DIR = join(process.cwd(), ".claude", "claudeclaw", "prompts");
const PROJECT_CLAUDE_MD = join(process.cwd(), "CLAUDE.md");
const LEGACY_PROJECT_CLAUDE_MD = join(process.cwd(), ".claude", "CLAUDE.md");
const CLAUDECLAW_BLOCK_START = "<!-- claudeclaw:managed:start -->";
const CLAUDECLAW_BLOCK_END = "<!-- claudeclaw:managed:end -->";

/**
 * Compact configuration.
 * COMPACT_WARN_THRESHOLD: notify user that context is getting large.
 * COMPACT_TIMEOUT_ENABLED: whether to auto-compact on timeout (exit 124).
 */
const COMPACT_WARN_THRESHOLD = 25;
const COMPACT_TIMEOUT_ENABLED = true;

export type CompactEvent =
  | { type: "warn"; turnCount: number }
  | { type: "auto-compact-start" }
  | { type: "auto-compact-done"; success: boolean }
  | { type: "auto-compact-retry"; success: boolean; stdout: string; stderr: string; exitCode: number };

type CompactEventListener = (event: CompactEvent) => void;
const compactListeners: CompactEventListener[] = [];

/** Register a listener for compact-related events (warnings, auto-compact notifications). */
export function onCompactEvent(listener: CompactEventListener): void {
  compactListeners.push(listener);
}

function emitCompactEvent(event: CompactEvent): void {
  for (const listener of compactListeners) {
    try { listener(event); } catch {}
  }
}

export interface RunResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

const RATE_LIMIT_PATTERN = /you.ve hit your limit|out of extra usage/i;

// Serial queue — prevents concurrent --resume on the same session
let queue: Promise<unknown> = Promise.resolve();

function enqueue<T>(fn: () => Promise<T>): Promise<T> {
  const task = queue.then(fn, fn);
  queue = task.catch(() => {});
  return task;
}

function extractRateLimitMessage(stdout: string, stderr: string): string | null {
  const candidates = [stdout, stderr];
  for (const text of candidates) {
    const trimmed = text.trim();
    if (trimmed && RATE_LIMIT_PATTERN.test(trimmed)) return trimmed;
  }
  return null;
}

function sameModelConfig(a: ModelConfig, b: ModelConfig): boolean {
  return a.model.trim().toLowerCase() === b.model.trim().toLowerCase() && a.api.trim() === b.api.trim();
}

function hasModelConfig(value: ModelConfig): boolean {
  return value.model.trim().length > 0 || value.api.trim().length > 0;
}

function isNotFoundError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const code = (error as { code?: unknown }).code;
  if (code === "ENOENT") return true;
  const message = String((error as { message?: unknown }).message ?? "");
  return /enoent|no such file or directory/i.test(message);
}

function buildChildEnv(baseEnv: Record<string, string>, model: string, api: string): Record<string, string> {
  const childEnv: Record<string, string> = { ...baseEnv };
  const normalizedModel = model.trim().toLowerCase();

  if (api.trim()) childEnv.ANTHROPIC_AUTH_TOKEN = api.trim();

  if (normalizedModel === "glm") {
    childEnv.ANTHROPIC_BASE_URL = "https://api.z.ai/api/anthropic";
    childEnv.API_TIMEOUT_MS = "3000000";
  }

  return childEnv;
}

/** Default timeout for a single Claude Code invocation (5 minutes). */
const CLAUDE_TIMEOUT_MS = 5 * 60 * 1000;

async function runClaudeOnce(
  baseArgs: string[],
  model: string,
  api: string,
  baseEnv: Record<string, string>,
  timeoutMs: number = CLAUDE_TIMEOUT_MS
): Promise<{ rawStdout: string; stderr: string; exitCode: number }> {
  const args = [...baseArgs];
  const normalizedModel = model.trim().toLowerCase();
  if (model.trim() && normalizedModel !== "glm") args.push("--model", model.trim());

  const proc = Bun.spawn(args, {
    stdout: "pipe",
    stderr: "pipe",
    env: buildChildEnv(baseEnv, model, api),
  });

  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error(`Claude session timed out after ${timeoutMs / 1000}s`)), timeoutMs);
  });

  try {
    const [rawStdout, stderr] = await Promise.race([
      Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
      ]),
      timeoutPromise,
    ]) as [string, string];
    await proc.exited;

    return {
      rawStdout,
      stderr,
      exitCode: proc.exitCode ?? 1,
    };
  } catch (err) {
    // Kill the hung process
    try { proc.kill("SIGTERM"); } catch {}
    setTimeout(() => { try { proc.kill("SIGKILL"); } catch {} }, 5000);

    const message = err instanceof Error ? err.message : String(err);
    console.error(`[${new Date().toLocaleTimeString()}] ${message}`);

    return {
      rawStdout: "",
      stderr: message,
      exitCode: 124,
    };
  }
}

const PROJECT_DIR = process.cwd();

const DIR_SCOPE_PROMPT = [
  `CRITICAL SECURITY CONSTRAINT: You are scoped to the project directory: ${PROJECT_DIR}`,
  "You MUST NOT read, write, edit, or delete any file outside this directory.",
  "You MUST NOT run bash commands that modify anything outside this directory (no cd /, no /etc, no ~/, no ../.. escapes).",
  "If a request requires accessing files outside the project, refuse and explain why.",
].join("\n");

export async function ensureProjectClaudeMd(): Promise<void> {
  // Preflight-only initialization: never rewrite an existing project CLAUDE.md.
  if (existsSync(PROJECT_CLAUDE_MD)) return;

  const promptContent = (await loadPrompts()).trim();
  const managedBlock = [
    CLAUDECLAW_BLOCK_START,
    promptContent,
    CLAUDECLAW_BLOCK_END,
  ].join("\n");

  let content = "";

  if (existsSync(LEGACY_PROJECT_CLAUDE_MD)) {
    try {
      const legacy = await readFile(LEGACY_PROJECT_CLAUDE_MD, "utf8");
      content = legacy.trim();
    } catch (e) {
      console.error(`[${new Date().toLocaleTimeString()}] Failed to read legacy .claude/CLAUDE.md:`, e);
      return;
    }
  }

  const normalized = content.trim();
  const hasManagedBlock =
    normalized.includes(CLAUDECLAW_BLOCK_START) && normalized.includes(CLAUDECLAW_BLOCK_END);
  const managedPattern = new RegExp(
    `${CLAUDECLAW_BLOCK_START}[\\s\\S]*?${CLAUDECLAW_BLOCK_END}`,
    "m"
  );

  const merged = hasManagedBlock
    ? `${normalized.replace(managedPattern, managedBlock)}\n`
    : normalized
      ? `${normalized}\n\n${managedBlock}\n`
      : `${managedBlock}\n`;

  try {
    await writeFile(PROJECT_CLAUDE_MD, merged, "utf8");
  } catch (e) {
    console.error(`[${new Date().toLocaleTimeString()}] Failed to write project CLAUDE.md:`, e);
  }
}

function buildSecurityArgs(security: SecurityConfig): string[] {
  const args: string[] = ["--dangerously-skip-permissions"];

  switch (security.level) {
    case "locked":
      args.push("--tools", "Read,Grep,Glob");
      break;
    case "strict":
      args.push("--disallowedTools", "Bash,WebSearch,WebFetch");
      break;
    case "moderate":
      // all tools available, scoped to project dir via system prompt
      break;
    case "unrestricted":
      // all tools, no directory restriction
      break;
  }

  if (security.allowedTools.length > 0) {
    args.push("--allowedTools", security.allowedTools.join(" "));
  }
  if (security.disallowedTools.length > 0) {
    args.push("--disallowedTools", security.disallowedTools.join(" "));
  }

  return args;
}

/** Load and concatenate all prompt files from the prompts/ directory. */
async function loadPrompts(): Promise<string> {
  const selectedPromptFiles = [
    join(PROMPTS_DIR, "IDENTITY.md"),
    join(PROMPTS_DIR, "USER.md"),
    join(PROMPTS_DIR, "SOUL.md"),
  ];
  const parts: string[] = [];

  for (const file of selectedPromptFiles) {
    try {
      const content = await Bun.file(file).text();
      if (content.trim()) parts.push(content.trim());
    } catch (e) {
      console.error(`[${new Date().toLocaleTimeString()}] Failed to read prompt file ${file}:`, e);
    }
  }

  return parts.join("\n\n");
}

/**
 * Load the heartbeat prompt template.
 * Project-level override takes precedence: place a file at
 * .claude/claudeclaw/prompts/HEARTBEAT.md to fully replace the built-in template.
 */
export async function loadHeartbeatPromptTemplate(): Promise<string> {
  const projectOverride = join(PROJECT_PROMPTS_DIR, "HEARTBEAT.md");
  for (const file of [projectOverride, HEARTBEAT_PROMPT_FILE]) {
    try {
      const content = await Bun.file(file).text();
      if (content.trim()) return content.trim();
    } catch (e) {
      if (!isNotFoundError(e)) {
        console.warn(`[${new Date().toLocaleTimeString()}] Failed to read heartbeat prompt file ${file}:`, e);
      }
    }
  }
  return "";
}

/** Run /compact on the current session to reduce context size. */
export async function runCompact(
  sessionId: string,
  model: string,
  api: string,
  baseEnv: Record<string, string>,
  securityArgs: string[],
  timeoutMs: number
): Promise<boolean> {
  const compactArgs = [
    "claude", "-p", "/compact",
    "--output-format", "text",
    "--resume", sessionId,
    ...securityArgs,
  ];
  console.log(`[${new Date().toLocaleTimeString()}] Running /compact on session ${sessionId.slice(0, 8)}...`);
  const result = await runClaudeOnce(compactArgs, model, api, baseEnv, timeoutMs);
  const success = result.exitCode === 0;
  console.log(`[${new Date().toLocaleTimeString()}] Compact ${success ? "succeeded" : `failed (exit ${result.exitCode})`}`);
  return success;
}

/**
 * High-level compact: resolves session + settings internally.
 * Returns { success, message }.
 */
export async function compactCurrentSession(): Promise<{ success: boolean; message: string }> {
  const existing = await getSession();
  if (!existing) return { success: false, message: "No active session to compact." };

  const settings = getSettings();
  const securityArgs = buildSecurityArgs(settings.security);
  const { CLAUDECODE: _, ...cleanEnv } = process.env;
  const baseEnv = { ...cleanEnv } as Record<string, string>;
  const timeoutMs = (settings as any).sessionTimeoutMs || CLAUDE_TIMEOUT_MS;

  const ok = await runCompact(
    existing.sessionId,
    settings.model,
    settings.api,
    baseEnv,
    securityArgs,
    timeoutMs
  );

  return ok
    ? { success: true, message: `✅ Session compact complete (${existing.sessionId.slice(0, 8)})` }
    : { success: false, message: `❌ Compact failed (${existing.sessionId.slice(0, 8)})` };
}

/**
 * Build memory context for injection into the prompt.
 * Scores memory entries by relevance to the current prompt.
 * Includes corrections readback.
 */
async function buildMemoryContext(currentPrompt: string): Promise<string> {
  const parts: string[] = [];
  const now = new Date();
  const todayStr = now.toISOString().slice(0, 10);
  const yesterday = new Date(now.getTime() - 86400000);
  const yesterdayStr = yesterday.toISOString().slice(0, 10);

  // Load daily logs (today + yesterday always included)
  for (const dateStr of [yesterdayStr, todayStr]) {
    try {
      const content = await Bun.file(join(MEMORY_DIR, `${dateStr}.md`)).text();
      if (content.trim()) parts.push(content.trim());
    } catch { /* doesn't exist yet */ }
  }

  // Load and score any other memory files (not date-based logs)
  try {
    const { readdir, stat } = await import("fs/promises");
    const files = await readdir(MEMORY_DIR);
    const promptLower = currentPrompt.toLowerCase();
    const promptWords = promptLower.split(/\s+/).filter(w => w.length > 2);

    const scored: Array<{ content: string; score: number }> = [];

    for (const file of files) {
      if (!file.endsWith(".md")) continue;
      // Skip daily logs (already loaded above)
      if (/^\d{4}-\d{2}-\d{2}\.md$/.test(file)) continue;

      try {
        const filePath = join(MEMORY_DIR, file);
        const content = await Bun.file(filePath).text();
        if (!content.trim()) continue;

        const fileStat = await stat(filePath);
        const ageDays = (now.getTime() - fileStat.mtimeMs) / (1000 * 60 * 60 * 24);
        const contentLower = content.toLowerCase();

        // Score by keyword overlap + recency
        let score = 0;
        for (const word of promptWords) {
          if (contentLower.includes(word)) score += 2;
        }
        if (ageDays < 1) score += 3;
        else if (ageDays < 7) score += 2;
        else if (ageDays < 30) score += 1;

        if (score > 0) scored.push({ content: content.trim(), score });
      } catch { /* skip unreadable */ }
    }

    // Top 5 by relevance
    scored.sort((a, b) => b.score - a.score);
    for (const entry of scored.slice(0, 5)) {
      parts.push(entry.content);
    }
  } catch { /* memory dir doesn't exist */ }

  // Corrections readback -- last 10 factual updates
  try {
    const lines = (await Bun.file(CORRECTIONS_FILE).text()).trim().split("\n").filter(Boolean);
    const last10 = lines.slice(-10).join("\n");
    if (last10) parts.push("Recent corrections:\n" + last10);
  } catch { /* no corrections file */ }

  if (parts.length === 0) return "";
  return "DAEMON MEMORY (context that survives compaction):\n" + parts.join("\n\n");
}

/**
 * Auto-save a conversation summary to the daily memory log.
 * Called by runner.ts after every non-trivial response -- no prompting needed.
 */
async function autoSaveMemory(name: string, prompt: string, output: string): Promise<void> {
  const trimmed = output.trim();
  if (!trimmed || trimmed === "HEARTBEAT_OK" || trimmed.startsWith("HEARTBEAT_OK")) return;

  try {
    await mkdir(MEMORY_DIR, { recursive: true });
    const now = new Date();
    const dateStr = now.toISOString().slice(0, 10);
    const timeStr = now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
    const memFile = join(MEMORY_DIR, `${dateStr}.md`);

    const shortPrompt = prompt.length > 200 ? prompt.slice(0, 200) + "..." : prompt;
    const shortOutput = trimmed.length > 300 ? trimmed.slice(0, 300) + "..." : trimmed;

    const entry = `\n### ${timeStr} - ${name.charAt(0).toUpperCase() + name.slice(1)}\n` +
      `Prompt: ${shortPrompt}\n` +
      `Response: ${shortOutput}\n`;

    let existing = "";
    try { existing = await Bun.file(memFile).text(); } catch { existing = `## ${dateStr}\n`; }
    await Bun.write(memFile, existing + entry);
  } catch (e) {
    console.error(`[${new Date().toLocaleTimeString()}] Failed to write memory log:`, e);
  }
}

/**
 * Extract [remember: ...] tags from output, save each as a memory file.
 * Returns the output with tags stripped.
 */
async function extractAndSaveRememberTags(output: string): Promise<string> {
  const pattern = /\[remember:\s*(.+?)\]/g;
  let match;
  const memories: string[] = [];

  while ((match = pattern.exec(output)) !== null) {
    memories.push(match[1].trim());
  }

  if (memories.length === 0) return output;

  await mkdir(MEMORY_DIR, { recursive: true });
  for (const mem of memories) {
    const key = "remember_" + Date.now().toString(36) + Math.random().toString(36).slice(2, 4);
    await Bun.write(join(MEMORY_DIR, `${key}.md`), mem);
    console.log(`[${new Date().toLocaleTimeString()}] Saved memory: ${key}`);
  }

  // Strip tags from output before delivery
  return output.replace(/\[remember:\s*.+?\]/g, "").replace(/\n{3,}/g, "\n\n").trim();
}

/**
 * Strip internal notes that should never reach the user.
 * Belt-and-suspenders: the prompt tells Claude not to output these,
 * but if it slips through, catch it here before delivery.
 */
function stripInternalNotes(output: string): string {
  const patterns = [
    // "Memory log for this exchange: ..." (full line)
    /^Memory log[^\n]*$/gm,
    // "Response sent and logged to daily memory" style
    /^Response sent[^\n]*$/gm,
    // "Let me update the memory file" / "Let me log this"
    /^Let me (?:update|log|write|save|append)[^\n]*(?:memory|log|file|daily)[^\n]*$/gim,
    // "The daily memory already has..." / "Memory log already captured..."
    /^(?:The daily memory|Memory log) already[^\n]*$/gm,
    // "I've logged this..." / "Logged to daily memory"
    /^(?:I've logged|Logged to)[^\n]*$/gm,
    // "HEARTBEAT_OK" fragments mixed with real content
    /^HEARTBEAT_OK\s*$/gm,
  ];

  let cleaned = output;
  for (const pat of patterns) {
    cleaned = cleaned.replace(pat, "");
  }
  // Collapse multiple blank lines left behind
  cleaned = cleaned.replace(/\n{3,}/g, "\n\n").trim();
  return cleaned;
}

async function execClaude(name: string, prompt: string): Promise<RunResult> {
  await mkdir(LOGS_DIR, { recursive: true });

  const existing = await getSession();
  const isNew = !existing;
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const logFile = join(LOGS_DIR, `${name}-${timestamp}.log`);

  const settings = getSettings();
  const { security, model, api, fallback, agentic } = settings;

  // Determine which model to use based on agentic routing
  let primaryConfig: ModelConfig;
  let taskType = "unknown";
  let routingReasoning = "";

  if (agentic.enabled) {
    const routing = selectModel(prompt, agentic.modes, agentic.defaultMode);
    primaryConfig = { model: routing.model, api };
    taskType = routing.taskType;
    routingReasoning = routing.reasoning;
    console.log(
      `[${new Date().toLocaleTimeString()}] Agentic routing: ${routing.taskType} → ${routing.model} (${routing.reasoning})`
    );
  } else {
    primaryConfig = { model, api };
  }

  const fallbackConfig: ModelConfig = {
    model: fallback?.model ?? "",
    api: fallback?.api ?? "",
  };
  const securityArgs = buildSecurityArgs(security);
  const timeoutMs = (settings as any).sessionTimeoutMs || CLAUDE_TIMEOUT_MS;

  console.log(
    `[${new Date().toLocaleTimeString()}] Running: ${name} (${isNew ? "new session" : `resume ${existing.sessionId.slice(0, 8)}`}, security: ${security.level})`
  );

  // New session: use json output to capture Claude's session_id
  // Resumed session: use text output with --resume
  const outputFormat = isNew ? "json" : "text";
  const args = ["claude", "-p", prompt, "--output-format", outputFormat, ...securityArgs];

  if (!isNew) {
    args.push("--resume", existing.sessionId);
  }

  // Build the appended system prompt: prompt files + directory scoping
  // This is passed on EVERY invocation (not just new sessions) because
  // --append-system-prompt does not persist across --resume.
  const promptContent = await loadPrompts();
  const appendParts: string[] = [
    "You are running inside ClaudeClaw.",
  ];
  if (promptContent) appendParts.push(promptContent);

  // NOTE: Project CLAUDE.md is NOT loaded here -- Claude Code loads it
  // natively on every -p invocation (confirmed via testing). Loading it
  // again via --append-system-prompt was duplicating ~300 lines per tick.

  // --- Memory injection: load relevant context into the prompt ---
  const memoryCtx = await buildMemoryContext(prompt);
  if (memoryCtx) appendParts.push(memoryCtx);

  // Tell the model it can use [remember: ...] tags to save important info
  appendParts.push(
    "If you learn something important that should persist across sessions, " +
    "include [remember: <fact>] anywhere in your output. It will be saved automatically and stripped before delivery. " +
    "Use sparingly -- only for durable facts, preferences, or decisions, not routine information."
  );

  if (security.level !== "unrestricted") appendParts.push(DIR_SCOPE_PROMPT);
  if (appendParts.length > 0) {
    args.push("--append-system-prompt", appendParts.join("\n\n"));
  }

  // Strip CLAUDECODE env var so child claude processes don't think they're nested
  const { CLAUDECODE: _, ...cleanEnv } = process.env;
  const baseEnv = { ...cleanEnv } as Record<string, string>;

  let exec = await runClaudeOnce(args, primaryConfig.model, primaryConfig.api, baseEnv, timeoutMs);
  const primaryRateLimit = extractRateLimitMessage(exec.rawStdout, exec.stderr);
  let usedFallback = false;

  if (primaryRateLimit && hasModelConfig(fallbackConfig) && !sameModelConfig(primaryConfig, fallbackConfig)) {
    console.warn(
      `[${new Date().toLocaleTimeString()}] Claude limit reached; retrying with fallback${fallbackConfig.model ? ` (${fallbackConfig.model})` : ""}...`
    );
    exec = await runClaudeOnce(args, fallbackConfig.model, fallbackConfig.api, baseEnv, timeoutMs);
    usedFallback = true;
  }

  const rawStdout = exec.rawStdout;
  const stderr = exec.stderr;
  const exitCode = exec.exitCode;
  let stdout = rawStdout;
  let sessionId = existing?.sessionId ?? "unknown";
  const rateLimitMessage = extractRateLimitMessage(rawStdout, stderr);

  if (rateLimitMessage) {
    stdout = rateLimitMessage;
  }

  // For new sessions, parse the JSON to extract session_id and result text
  if (!rateLimitMessage && isNew && exitCode === 0) {
    try {
      const json = JSON.parse(rawStdout);
      sessionId = json.session_id;
      stdout = json.result ?? "";
      // Save the real session ID from Claude Code
      await createSession(sessionId);
      console.log(`[${new Date().toLocaleTimeString()}] Session created: ${sessionId}`);
    } catch (e) {
      console.error(`[${new Date().toLocaleTimeString()}] Failed to parse session from Claude output:`, e);
    }
  }

  const result: RunResult = {
    stdout,
    stderr,
    exitCode,
  };

  const output = [
    `# ${name}`,
    `Date: ${new Date().toISOString()}`,
    `Session: ${sessionId} (${isNew ? "new" : "resumed"})`,
    `Model config: ${usedFallback ? "fallback" : "primary"}`,
    ...(agentic.enabled ? [`Task type: ${taskType}`, `Routing: ${routingReasoning}`] : []),
    `Prompt: ${prompt}`,
    `Exit code: ${result.exitCode}`,
    "",
    "## Output",
    stdout,
    ...(stderr ? ["## Stderr", stderr] : []),
  ].join("\n");

  await Bun.write(logFile, output);
  console.log(`[${new Date().toLocaleTimeString()}] Done: ${name} → ${logFile}`);

  // --- Auto-save memory + extract [remember:] tags ---
  if (exitCode === 0 && !rateLimitMessage) {
    // Auto-save conversation summary to daily log
    await autoSaveMemory(name, prompt, stdout);

    // Extract and save [remember: ...] tags, strip from output
    const cleaned = await extractAndSaveRememberTags(stdout);
    if (cleaned !== stdout) {
      stdout = cleaned;
      result.stdout = cleaned;
    }

    // Strip internal notes that should never reach Telegram
    const stripped = stripInternalNotes(stdout);
    if (stripped !== stdout) {
      stdout = stripped;
      result.stdout = stripped;
    }
  }

  // --- Auto-compact on timeout (exit 124) ---
  if (COMPACT_TIMEOUT_ENABLED && exitCode === 124 && !isNew && existing) {
    emitCompactEvent({ type: "auto-compact-start" });
    const compactOk = await runCompact(
      existing.sessionId,
      primaryConfig.model,
      primaryConfig.api,
      baseEnv,
      securityArgs,
      timeoutMs
    );
    emitCompactEvent({ type: "auto-compact-done", success: compactOk });

    if (compactOk) {
      console.log(`[${new Date().toLocaleTimeString()}] Retrying ${name} after compact...`);
      const retryExec = await runClaudeOnce(args, primaryConfig.model, primaryConfig.api, baseEnv, timeoutMs);
      const retryResult: RunResult = {
        stdout: retryExec.rawStdout,
        stderr: retryExec.stderr,
        exitCode: retryExec.exitCode,
      };
      emitCompactEvent({
        type: "auto-compact-retry",
        success: retryExec.exitCode === 0,
        stdout: retryResult.stdout,
        stderr: retryResult.stderr,
        exitCode: retryResult.exitCode,
      });

      if (retryExec.exitCode === 0) {
        const count = await incrementTurn();
        console.log(`[${new Date().toLocaleTimeString()}] Turn count: ${count} (after compact + retry)`);
      }
      return retryResult;
    }
  }

  // --- Turn tracking & compact warning ---
  if (exitCode === 0 && !isNew) {
    const turnCount = await incrementTurn();
    console.log(`[${new Date().toLocaleTimeString()}] Turn count: ${turnCount}`);

    if (turnCount >= COMPACT_WARN_THRESHOLD && existing && !existing.compactWarned) {
      await markCompactWarned();
      emitCompactEvent({ type: "warn", turnCount });
    }
  }

  return result;
}

export async function run(name: string, prompt: string): Promise<RunResult> {
  return enqueue(() => execClaude(name, prompt));
}

async function streamClaude(
  name: string,
  prompt: string,
  onChunk: (text: string) => void,
  onUnblock: () => void
): Promise<void> {
  await mkdir(LOGS_DIR, { recursive: true });

  const existing = await getSession();
  const { security, model, api } = getSettings();
  const securityArgs = buildSecurityArgs(security);

  // stream-json gives us events as they happen — text before tool calls,
  // so we can unblock the UI as soon as Claude acknowledges, not after sub-agents finish.
  // --verbose is required for stream-json to produce output in -p (print) mode.
  const args = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose", ...securityArgs];

  if (existing) args.push("--resume", existing.sessionId);

  const promptContent = await loadPrompts();
  const appendParts: string[] = ["You are running inside ClaudeClaw."];
  if (promptContent) appendParts.push(promptContent);

  if (existsSync(PROJECT_CLAUDE_MD)) {
    try {
      const claudeMd = await Bun.file(PROJECT_CLAUDE_MD).text();
      if (claudeMd.trim()) appendParts.push(claudeMd.trim());
    } catch {}
  }

  if (security.level !== "unrestricted") appendParts.push(DIR_SCOPE_PROMPT);
  if (appendParts.length > 0) {
    args.push("--append-system-prompt", appendParts.join("\n\n"));
  }

  const normalizedModel = model.trim().toLowerCase();
  if (model.trim() && normalizedModel !== "glm") args.push("--model", model.trim());

  const { CLAUDECODE: _, ...cleanEnv } = process.env;
  const childEnv = buildChildEnv(cleanEnv as Record<string, string>, model, api);

  console.log(`[${new Date().toLocaleTimeString()}] Running: ${name} (stream-json, session: ${existing?.sessionId?.slice(0, 8) ?? "new"})`);

  const proc = Bun.spawn(args, {
    stdout: "pipe",
    stderr: "pipe",
    env: childEnv,
  });

  const reader = proc.stdout.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let unblocked = false;
  let textEmitted = false;

  const maybeUnblock = () => {
    if (!unblocked) {
      unblocked = true;
      onUnblock();
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    // Parse complete newline-delimited JSON events
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const event = JSON.parse(trimmed) as Record<string, unknown>;

        if (event.type === "system" && (event.subtype === "init" || event.session_id)) {
          // Capture session ID for new sessions
          const sid = event.session_id as string | undefined;
          if (sid && !existing) {
            await createSession(sid);
            console.log(`[${new Date().toLocaleTimeString()}] Session created (stream-json): ${sid}`);
          }
        } else if (event.type === "assistant") {
          // Text and tool_use blocks from the assistant
          type ContentBlock = { type: string; text?: string; id?: string; name?: string; input?: Record<string, unknown> };
          const msg = event.message as { content?: ContentBlock[] } | undefined;
          const blocks = msg?.content ?? [];
          let hasActivity = false;
          for (const block of blocks) {
            if (block.type === "text" && block.text) {
              onChunk(block.text);
              textEmitted = true;
              hasActivity = true;
            } else if (block.type === "tool_use") {
              hasActivity = true;
            }
          }
          if (hasActivity) maybeUnblock();
        } else if (event.type === "tool_use") {
          // Top-level tool_use event (some stream-json versions) — unblock the UI
          maybeUnblock();
        } else if (event.type === "result") {
          // Final result event — emit text as fallback if no assistant text was seen
          const resultText = (event as Record<string, unknown>).result as string | undefined;
          if (resultText && !textEmitted) {
            onChunk(resultText);
          }
          maybeUnblock();
        }
      } catch {}
    }
  }

  await proc.exited;
  // Ensure unblock fires even if something unexpected happened
  maybeUnblock();

  console.log(`[${new Date().toLocaleTimeString()}] Done: ${name}`);
}

export async function streamUserMessage(
  name: string,
  prompt: string,
  onChunk: (text: string) => void,
  onUnblock: () => void
): Promise<void> {
  return enqueue(() => streamClaude(name, prefixUserMessageWithClock(prompt), onChunk, onUnblock));
}

function prefixUserMessageWithClock(prompt: string): string {
  try {
    const settings = getSettings();
    const prefix = buildClockPromptPrefix(new Date(), settings.timezoneOffsetMinutes);
    return `${prefix}\n${prompt}`;
  } catch {
    const prefix = buildClockPromptPrefix(new Date(), 0);
    return `${prefix}\n${prompt}`;
  }
}

export async function runUserMessage(name: string, prompt: string): Promise<RunResult> {
  return run(name, prefixUserMessageWithClock(prompt));
}

/**
 * Bootstrap the session: fires Claude with the system prompt so the
 * session is created immediately. No-op if a session already exists.
 */
export async function bootstrap(): Promise<void> {
  const existing = await getSession();
  if (existing) return;

  console.log(`[${new Date().toLocaleTimeString()}] Bootstrapping new session...`);
  await execClaude("bootstrap", "Wakeup, my friend!");
  console.log(`[${new Date().toLocaleTimeString()}] Bootstrap complete — session is live.`);
}
