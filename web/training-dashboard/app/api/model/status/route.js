import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const rootDir = path.resolve(process.cwd(), "..", "..");
const checkpointsDir = path.join(rootDir, "checkpoints");
const officialDir = path.join(checkpointsDir, "official");
const tokenizerPath = path.join(rootDir, "data", "tokenizer", "jarvis.model");

function readJson(filePath) {
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function modelFileFor(fullPath) {
  if (existsSync(path.join(fullPath, "model.safetensors"))) return "model.safetensors";
  if (existsSync(path.join(fullPath, "pytorch_model.bin"))) return "pytorch_model.bin";
  return null;
}

function checkpointComplete(fullPath) {
  return Boolean(
    modelFileFor(fullPath) &&
      existsSync(path.join(fullPath, "optimizer.bin")) &&
      existsSync(path.join(fullPath, "scheduler.bin")),
  );
}

function checkpointStep(name) {
  const step = Number(name.match(/^step_(\d+)$/)?.[1]);
  return Number.isFinite(step) ? step : null;
}

function checkpointPayload(fullPath, source) {
  const name = path.basename(fullPath);
  const meta = readJson(path.join(fullPath, "jarvis_checkpoint_meta.json")) || {};
  const pipelineStep = meta && meta.pipeline_step && typeof meta.pipeline_step === "object" ? meta.pipeline_step : {};
  const warnings = [];

  if (source !== "official") {
    warnings.push("Snapshot attivo: non e' un checkpoint ufficiale pubblicato.");
  }
  if (!checkpointComplete(fullPath)) {
    warnings.push("Checkpoint incompleto: valido solo per test veloce, non per ripresa training.");
  }
  if (pipelineStep?.id === "step_1_italian_corpus") {
    warnings.push("Stage 1 LM: completa testo, non risponde ancora come assistente.");
  }
  if (source !== "official" && !pipelineStep?.id) {
    warnings.push("Metadata stage assente: snapshot vecchio o parziale.");
  }

  return {
    name,
    step: checkpointStep(name),
    path: path.relative(rootDir, fullPath),
    modelFile: modelFileFor(fullPath),
    updatedAt: statSync(fullPath).mtime.toISOString(),
    source,
    complete: checkpointComplete(fullPath),
    pipelineStep,
    warnings,
  };
}

function latestOfficialCheckpoint() {
  const latest = readJson(path.join(officialDir, "latest.json"));
  if (!latest || latest.kind !== "train" || !latest.artifact_path) return null;

  const fullPath = path.resolve(latest.artifact_path);
  if (!existsSync(fullPath) || !modelFileFor(fullPath)) return null;

  return checkpointPayload(fullPath, "official");
}

function latestActiveCheckpoint() {
  if (!existsSync(checkpointsDir)) return null;

  const candidates = readdirSync(checkpointsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && /^step_\d+$/.test(entry.name))
    .map((entry) => {
      const fullPath = path.join(checkpointsDir, entry.name);
      return checkpointPayload(fullPath, "active");
    })
    .filter((entry) => Number.isFinite(entry.step) && entry.modelFile)
    .sort((a, b) => b.step - a.step);

  const complete = candidates.find((entry) => entry.complete);
  return complete || candidates[0] || null;
}

function selectedCheckpoint() {
  return latestOfficialCheckpoint() || latestActiveCheckpoint();
}

export async function GET() {
  const worker = globalThis.__jarvisModelWorkerState;

  return Response.json(
    {
      checkpoint: selectedCheckpoint(),
      tokenizer: {
        available: existsSync(tokenizerPath),
        path: path.relative(rootDir, tokenizerPath),
      },
      worker: {
        running: Boolean(worker?.process && !worker?.stopping),
        stopping: Boolean(worker?.stopping),
        startedAt: worker?.startedAt || null,
        pending: worker?.pending?.length || 0,
        lastError: worker?.lastError || null,
        logs: worker?.logs?.slice(-12) || [],
      },
    },
    { headers: { "Cache-Control": "no-store" } },
  );
}
