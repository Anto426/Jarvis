import { statfsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { execSync } from "node:child_process";

const rootDir = path.resolve(process.cwd(), "..", "..");
const metricsPath = path.join(rootDir, "logs", "training_metrics.json");
const zoomPath = path.join(rootDir, "logs", "zoom_state.json");

let previousCpuSample = null;

function round(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return null;
  const factor = 10 ** digits;
  return Math.round(Number(value) * factor) / factor;
}

function bytesToGb(value) {
  return value / (1024 ** 3);
}

function sampleCpuPercent() {
  const cpus = os.cpus();
  const sample = cpus.reduce(
    (acc, cpu) => {
      const times = cpu.times;
      const idle = times.idle;
      const total = Object.values(times).reduce((sum, value) => sum + value, 0);
      return {
        idle: acc.idle + idle,
        total: acc.total + total,
      };
    },
    { idle: 0, total: 0 },
  );

  if (!previousCpuSample) {
    previousCpuSample = sample;
    return null;
  }

  const idleDelta = sample.idle - previousCpuSample.idle;
  const totalDelta = sample.total - previousCpuSample.total;
  previousCpuSample = sample;

  if (totalDelta <= 0) return null;
  return round(Math.max(0, Math.min(100, (1 - idleDelta / totalDelta) * 100)), 1);
}

function getGpuTelemetry() {
  try {
    const output = execSync(
      [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
      ].join(" "),
      { encoding: "utf8", timeout: 1500 },
    ).trim();

    const firstLine = output.split(/\r?\n/).find(Boolean);
    if (!firstLine) return {};

    const [name, util, memUsed, memTotal, temp, powerDraw, powerLimit] = firstLine
      .split(",")
      .map((item) => item.trim());
    const memoryUsedGb = Number(memUsed) / 1024;
    const memoryTotalGb = Number(memTotal) / 1024;

    return {
      gpu_name: name || null,
      gpu_util_percent: round(Number(util), 1),
      gpu_memory_used_gb: round(memoryUsedGb, 2),
      gpu_memory_total_gb: round(memoryTotalGb, 2),
      gpu_memory_percent: memoryTotalGb > 0 ? round((memoryUsedGb / memoryTotalGb) * 100, 1) : null,
      gpu_temp_c: round(Number(temp), 0),
      gpu_power_w: round(Number(powerDraw), 1),
      gpu_power_limit_w: round(Number(powerLimit), 1),
    };
  } catch {
    return {};
  }
}

function getDiskTelemetry() {
  try {
    const stats = statfsSync(rootDir);
    const totalBytes = stats.blocks * stats.bsize;
    const freeBytes = stats.bavail * stats.bsize;
    const usedBytes = Math.max(0, totalBytes - freeBytes);

    return {
      disk_used_gb: round(bytesToGb(usedBytes), 1),
      disk_total_gb: round(bytesToGb(totalBytes), 1),
      disk_free_gb: round(bytesToGb(freeBytes), 1),
      disk_percent: totalBytes > 0 ? round((usedBytes / totalBytes) * 100, 1) : null,
    };
  } catch {
    return {};
  }
}

function getSystemTelemetry() {
  const totalRamBytes = os.totalmem();
  const freeRamBytes = os.freemem();
  const usedRamBytes = Math.max(0, totalRamBytes - freeRamBytes);

  return {
    captured_at: new Date().toISOString(),
    cpu_percent: sampleCpuPercent(),
    ram_used_gb: round(bytesToGb(usedRamBytes), 1),
    ram_total_gb: round(bytesToGb(totalRamBytes), 1),
    ram_free_gb: round(bytesToGb(freeRamBytes), 1),
    ram_percent: totalRamBytes > 0 ? round((usedRamBytes / totalRamBytes) * 100, 1) : null,
    ...getGpuTelemetry(),
    ...getDiskTelemetry(),
  };
}

function getSystemHardware() {
  const cpus = os.cpus();
  const cpuModel = cpus.length > 0 ? cpus[0].model : "Unknown CPU";
  const totalRam = round(os.totalmem() / (1024 ** 3), 1) + "GB";
  
  const gpuName = getGpuTelemetry().gpu_name || "GPU non rilevata";

  return {
    cpu: cpuModel.replace(/\(R\)|\(TM\)|Core|Processor/g, "").trim(),
    ram: totalRam,
    gpu: gpuName
  };
}

function waitingPayload() {
  const system = getSystemTelemetry();
  return {
    status: "waiting",
    updated_at: null,
    total_steps: null,
    current: {},
    history: [],
    epochs: [],
    hardware: getSystemHardware(),
    system,
  };
}

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const raw = await readFile(metricsPath, "utf8");
    const payload = JSON.parse(raw);
    
    payload.hardware = getSystemHardware();
    payload.system = getSystemTelemetry();

    // Inject zoom state if available
    try {
      const zoomRaw = await readFile(zoomPath, "utf8");
      const zoomState = JSON.parse(zoomRaw);
      if (zoomState.lossDomain !== undefined) payload.lossDomain = zoomState.lossDomain;
      if (zoomState.lrDomain !== undefined) payload.lrDomain = zoomState.lrDomain;
    } catch (e) {
      // Zoom file might not exist yet, that's fine
    }

    return Response.json(payload, {
      headers: { "Cache-Control": "no-store" }
    });
  } catch (error) {
    if (error.code === "ENOENT") {
      return Response.json(waitingPayload(), {
        headers: { "Cache-Control": "no-store" }
      });
    }

    if (error instanceof SyntaxError) {
      return Response.json(
        { status: "reading", history: [], epochs: [] },
        { status: 503, headers: { "Cache-Control": "no-store" } }
      );
    }

    return Response.json(
      { status: "error", history: [], epochs: [] },
      { status: 500, headers: { "Cache-Control": "no-store" } }
    );
  }
}
