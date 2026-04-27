import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const rootDir = path.resolve(process.cwd(), "..", "..");
const workerScript = path.join(rootDir, "scripts", "test_model.py");
const venvPython = path.join(rootDir, ".venv", "Scripts", "python.exe");
const pythonPath = existsSync(venvPython) ? venvPython : "python";
const requestTimeoutMs = 10 * 60 * 1000;
const maxPromptChars = 8000;

function getWorkerState() {
  if (!globalThis.__jarvisModelWorkerState) {
    globalThis.__jarvisModelWorkerState = {
      process: null,
      buffer: "",
      pending: [],
      logs: [],
      startedAt: null,
      lastError: null,
      stopping: false,
    };
  }

  return globalThis.__jarvisModelWorkerState;
}

function appendLog(state, value) {
  const text = String(value || "").trim();
  if (!text) return;
  state.logs = [...state.logs, ...text.split(/\r?\n/)].slice(-80);
}

function rejectPending(state, error) {
  for (const pending of state.pending.splice(0)) {
    clearTimeout(pending.timer);
    pending.reject(error);
  }
}

function handleWorkerLine(state, line) {
  if (!line) return;

  let message;
  try {
    message = JSON.parse(line);
  } catch {
    appendLog(state, line);
    return;
  }

  const pending = state.pending.shift();
  if (!pending) {
    appendLog(state, `Risposta worker senza richiesta: ${line}`);
    return;
  }

  clearTimeout(pending.timer);
  if (message.ok === false) {
    pending.reject(new Error(message.error || "Generazione fallita"));
    return;
  }

  pending.resolve(message);
}

function startWorker() {
  const state = getWorkerState();
  if (state.process && !state.process.killed) {
    return state;
  }

  if (!existsSync(workerScript)) {
    throw new Error(`Worker Python non trovato: ${workerScript}`);
  }

  const worker = spawn(pythonPath, [workerScript, "--stdio-server"], {
    cwd: rootDir,
    env: {
      ...process.env,
      PYTHONPATH: rootDir,
      PYTHONUTF8: "1",
      PYTHONIOENCODING: "utf-8",
    },
    stdio: ["pipe", "pipe", "pipe"],
    windowsHide: true,
  });

  state.process = worker;
  state.buffer = "";
  state.pending = [];
  state.logs = [];
  state.startedAt = new Date().toISOString();
  state.lastError = null;
  state.stopping = false;

  worker.stdout.setEncoding("utf8");
  worker.stdout.on("data", (chunk) => {
    state.buffer += chunk;

    let newlineIndex = state.buffer.indexOf("\n");
    while (newlineIndex >= 0) {
      const line = state.buffer.slice(0, newlineIndex).trim();
      state.buffer = state.buffer.slice(newlineIndex + 1);
      handleWorkerLine(state, line);
      newlineIndex = state.buffer.indexOf("\n");
    }
  });

  worker.stderr.setEncoding("utf8");
  worker.stderr.on("data", (chunk) => appendLog(state, chunk));

  worker.on("error", (error) => {
    state.lastError = error.message;
    state.process = null;
    rejectPending(state, error);
  });

  worker.on("exit", (code, signal) => {
    const reason = signal || (code === null || code === undefined ? "unknown" : code);
    let message = "Modello chiuso manualmente";
    if (state.stopping) {
      appendLog(state, "Modello chiuso manualmente");
      state.lastError = null;
    } else {
      message = `Worker modello terminato (${reason})`;
      state.lastError = message;
    }
    state.process = null;
    state.buffer = "";
    state.startedAt = null;
    state.stopping = false;
    rejectPending(state, new Error(message));
  });

  return state;
}

function stopWorker() {
  const state = getWorkerState();
  const worker = state.process;

  if (!worker) {
    state.startedAt = null;
    state.stopping = false;
    state.lastError = null;
    return { stopped: true, wasRunning: false };
  }

  state.stopping = true;
  state.lastError = null;
  rejectPending(state, new Error("Modello chiuso manualmente"));

  try {
    worker.stdin?.end();
  } catch {
    // The process may already be closing.
  }

  try {
    worker.kill();
  } catch (error) {
    state.stopping = false;
    state.lastError = error.message;
    throw error;
  }

  setTimeout(() => {
    if (state.process === worker) {
      try {
        worker.kill("SIGKILL");
      } catch {
        // Best effort shutdown.
      }
    }
  }, 3000);

  return { stopped: true, wasRunning: true };
}

function sendToWorker(payload) {
  const state = startWorker();

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      const index = state.pending.findIndex((item) => item.timer === timer);
      if (index >= 0) state.pending.splice(index, 1);
      reject(new Error("Timeout generazione modello"));
    }, requestTimeoutMs);

    const pending = { resolve, reject, timer };
    state.pending.push(pending);

    state.process.stdin.write(`${JSON.stringify(payload)}\n`, "utf8", (error) => {
      if (!error) return;

      const index = state.pending.indexOf(pending);
      if (index >= 0) state.pending.splice(index, 1);
      clearTimeout(timer);
      reject(error);
    });
  });
}

function clampNumber(value, min, max, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(min, Math.min(max, number));
}

export async function POST(request) {
  try {
    const body = await request.json();
    const prompt = String(body.prompt || "").trim().slice(0, maxPromptChars);

    if (!prompt) {
      return Response.json({ ok: false, error: "Prompt vuoto" }, { status: 400 });
    }

    const result = await sendToWorker({
      command: "generate",
      prompt,
      max_new_tokens: Math.round(clampNumber(body.maxNewTokens, 1, 512, 100)),
      temperature: clampNumber(body.temperature, 0.01, 2, 0.8),
      top_p: clampNumber(body.topP, 0.05, 1, 0.9),
      repetition_penalty: clampNumber(body.repetitionPenalty, 1, 2, 1.1),
    });

    return Response.json(result, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    const state = getWorkerState();
    return Response.json(
      {
        ok: false,
        error: error.message || "Errore generazione",
        logs: state.logs.slice(-12),
      },
      { status: 500, headers: { "Cache-Control": "no-store" } },
    );
  }
}

export async function GET() {
  const state = getWorkerState();

  return Response.json(
    {
      running: Boolean(state.process && !state.stopping),
      stopping: Boolean(state.stopping),
      startedAt: state.startedAt,
      pending: state.pending.length,
      lastError: state.lastError,
      logs: state.logs.slice(-20),
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}

export async function DELETE() {
  try {
    const result = stopWorker();
    return Response.json(
      {
        ok: true,
        ...result,
      },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch (error) {
    return Response.json(
      {
        ok: false,
        error: error.message || "Chiusura modello fallita",
      },
      { status: 500, headers: { "Cache-Control": "no-store" } },
    );
  }
}
