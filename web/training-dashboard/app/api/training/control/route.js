import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const rootDir = path.resolve(process.cwd(), "..", "..");
const logsDir = path.join(rootDir, "logs");
const metricsPath = path.join(logsDir, "training_metrics.json");
const controlPath = path.join(logsDir, "training_control.json");

export const dynamic = "force-dynamic";

async function readJson(filePath) {
  try {
    return JSON.parse(await readFile(filePath, "utf8"));
  } catch {
    return null;
  }
}

async function writeControl(data) {
  await mkdir(logsDir, { recursive: true });
  const payload = {
    ...data,
    updated_at: new Date().toISOString(),
  };
  await writeFile(controlPath, JSON.stringify(payload, null, 2), "utf8");
  return payload;
}

function powershellJson(script, timeout = 4000) {
  try {
    const output = execFileSync(
      "powershell.exe",
      ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
      { cwd: rootDir, encoding: "utf8", timeout, windowsHide: true },
    ).trim();
    if (!output) return [];
    const parsed = JSON.parse(output);
    return Array.isArray(parsed) ? parsed : [parsed];
  } catch {
    return [];
  }
}

function trainingProcesses() {
  const script = `
$items = Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and
  ($_.CommandLine -match 'training[\\\\/]+train\\.py' -or $_.CommandLine -match 'jarvis_pipeline\\.ps1') -and
  $_.CommandLine -notmatch 'Get-CimInstance Win32_Process'
} | Select-Object ProcessId, ParentProcessId, Name, CommandLine
$items | ConvertTo-Json -Depth 3
`;

  return powershellJson(script).map((item) => ({
    pid: Number(item.ProcessId),
    parentPid: Number(item.ParentProcessId),
    name: item.Name,
    commandLine: item.CommandLine,
  })).filter((item) => Number.isFinite(item.pid));
}

function processIds(processes) {
  return [...new Set(processes.map((item) => Number(item.pid)).filter(Number.isFinite))];
}

function applyPauseState(processes, action) {
  const ids = processIds(processes);
  if (!ids.length) return false;
  const method = action === "pause" ? "NtSuspendProcess" : "NtResumeProcess";
  const idList = ids.join(",");
  const script = `
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class JarvisNtProcess {
  [DllImport("ntdll.dll")] public static extern int NtSuspendProcess(IntPtr processHandle);
  [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr processHandle);
}
"@ -ErrorAction SilentlyContinue
$Ids = @(${idList})
foreach ($Id in $Ids) {
  try {
    $Process = [System.Diagnostics.Process]::GetProcessById([int]$Id)
    [JarvisNtProcess]::${method}($Process.Handle) | Out-Null
  } catch {}
}
`;

  try {
    execFileSync(
      "powershell.exe",
      ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
      { cwd: rootDir, encoding: "utf8", timeout: 8000, windowsHide: true },
    );
    return true;
  } catch {
    return false;
  }
}

async function controlPayload(extra = {}) {
  const processes = trainingProcesses();
  const metrics = (await readJson(metricsPath)) || {};
  const control = await readJson(controlPath);
  const paused = processes.length > 0 && control?.action === "pause";

  return {
    running: processes.length > 0,
    paused,
    processes,
    pids: processIds(processes),
    control,
    metricsStatus: metrics.status || "waiting",
    stage: (metrics.pipeline_step || {}).id || (metrics.config || {}).pipeline_step || "",
    currentStep: Number(metrics.current?.global_step ?? metrics.start_global_step ?? 0) || 0,
    totalSteps: Number(metrics.total_steps || metrics.current?.total_steps || 0) || 0,
    ...extra,
  };
}

export async function GET() {
  return Response.json(await controlPayload(), {
    headers: { "Cache-Control": "no-store" },
  });
}

export async function POST(request) {
  try {
    const body = await request.json();
    const action = String(body.action || "").toLowerCase();
    if (!["pause", "resume", "save"].includes(action)) {
      return Response.json(
        { ok: false, error: "Comando supportato solo: pause/resume/save." },
        { status: 400, headers: { "Cache-Control": "no-store" } },
      );
    }

    const processes = trainingProcesses();
    if (!processes.length) {
      return Response.json(
        await controlPayload({ ok: false, action, error: "Nessun training attivo da controllare." }),
        { status: 409, headers: { "Cache-Control": "no-store" } },
      );
    }

    if (action === "save") {
      const control = await writeControl({
        action,
        request_id: String(Date.now()),
        requested_at: new Date().toISOString(),
        pids: processIds(processes),
      });
      return Response.json(await controlPayload({ ok: true, action, requested: true, control }), {
        headers: { "Cache-Control": "no-store" },
      });
    }

    const applied = applyPauseState(processes, action);
    const control = await writeControl({
      action,
      requested_at: new Date().toISOString(),
      pids: processIds(processes),
    });

    return Response.json(await controlPayload({ ok: applied, action, applied, control }), {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    return Response.json(
      { ok: false, error: error.message },
      { status: 500, headers: { "Cache-Control": "no-store" } },
    );
  }
}
