"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowRight,
  Bot,
  Cpu,
  Database,
  Gauge,
  LineChart as LineChartIcon,
  MemoryStick,
  Monitor,
  Pause,
  Play,
  RefreshCcw,
  Save,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";

function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatFixed(value, digits = 1) {
  const number = numberOrNull(value);
  return number === null ? "-" : number.toFixed(digits);
}

function formatLoss(value) {
  const number = numberOrNull(value);
  return number === null ? "-" : number.toFixed(4);
}

function formatCompact(value) {
  const number = numberOrNull(value);
  return number === null ? "-" : number.toExponential(2);
}

function formatGb(value, digits = 1) {
  const number = numberOrNull(value);
  return number === null ? "-" : `${number.toFixed(digits)} GB`;
}

function formatUnit(value, unit, digits = 0) {
  const number = numberOrNull(value);
  return number === null ? "-" : `${number.toFixed(digits)}${unit}`;
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleTimeString();
}

function formatDuration(seconds) {
  const value = numberOrNull(seconds);
  if (value === null || value <= 0) return "-";
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function formatSigned(value, digits = 1, suffix = "") {
  const number = numberOrNull(value);
  if (number === null) return "-";
  const sign = number > 0 ? "+" : "";
  return `${sign}${number.toFixed(digits)}${suffix}`;
}

function formatInteger(value) {
  const number = numberOrNull(value);
  return number === null ? "-" : Math.round(number).toLocaleString("it-IT");
}

function metricRows(history) {
  const rows = new Map();
  for (const item of Array.isArray(history) ? history : []) {
    if (!item?.global_step) continue;
    const step = Number(item.global_step);
    const row = rows.get(step) || { step };
    if (item.type === "train") {
      row.trainLoss = numberOrNull(item.loss);
      row.perplexity = numberOrNull(item.perplexity);
      row.learningRate = numberOrNull(item.learning_rate);
    }
    if (item.type === "eval") {
      row.evalLoss = numberOrNull(item.loss);
      row.evalPerplexity = numberOrNull(item.perplexity);
    }
    rows.set(step, row);
  }
  return Array.from(rows.values()).sort((a, b) => a.step - b.step);
}

function latestFrom(data) {
  const history = Array.isArray(data.history) ? data.history : [];
  return data.current?.loss !== undefined ? data.current : history[history.length - 1] || {};
}

function computeEta(history, currentStep, totalSteps) {
  const rows = (Array.isArray(history) ? history : [])
    .filter((item) => item?.timestamp && item?.global_step)
    .slice(-30);

  if (rows.length < 2 || !totalSteps || currentStep >= totalSteps) return null;

  const first = rows[0];
  const last = rows[rows.length - 1];
  const seconds = (new Date(last.timestamp).getTime() - new Date(first.timestamp).getTime()) / 1000;
  const steps = Number(last.global_step) - Number(first.global_step);
  if (seconds <= 0 || steps <= 0) return null;
  return ((totalSteps - currentStep) / steps) * seconds;
}

function average(values) {
  const clean = values.filter((value) => Number.isFinite(Number(value)));
  if (!clean.length) return null;
  return clean.reduce((sum, value) => sum + Number(value), 0) / clean.length;
}

function standardDeviation(values) {
  const avg = average(values);
  if (avg === null) return null;
  const clean = values.filter((value) => Number.isFinite(Number(value)));
  const variance = average(clean.map((value) => (Number(value) - avg) ** 2));
  return variance === null ? null : Math.sqrt(variance);
}

function movingAverageRows(rows, windowSize) {
  const window = Math.max(1, Number(windowSize) || 1);
  return rows.map((row, index) => {
    const start = Math.max(0, index - window + 1);
    const subset = rows.slice(start, index + 1);
    return {
      ...row,
      trainLossSmooth: average(subset.map((item) => item.trainLoss)),
      perplexitySmooth: average(subset.map((item) => item.perplexity)),
      learningRateSmooth: average(subset.map((item) => item.learningRate)),
    };
  });
}

function computeAnalysis(history, rows, currentStep, totalSteps) {
  const trainRows = rows.filter((row) => Number.isFinite(Number(row.trainLoss)));
  const recent = trainRows.slice(-120);
  const previous = trainRows.slice(-240, -120);
  const recentAvg = average(recent.map((row) => row.trainLoss));
  const previousAvg = average(previous.map((row) => row.trainLoss));
  const trendPct =
    recentAvg !== null && previousAvg !== null && previousAvg !== 0
      ? ((recentAvg - previousAvg) / previousAvg) * 100
      : null;
  const volatility = standardDeviation(recent.map((row) => row.trainLoss));
  const best = trainRows.reduce((bestRow, row) => {
    if (!bestRow || row.trainLoss < bestRow.trainLoss) return row;
    return bestRow;
  }, null);

  const timed = (Array.isArray(history) ? history : [])
    .filter((item) => item?.timestamp && item?.global_step && item.type === "train")
    .slice(-80);
  let stepsPerMinute = null;
  if (timed.length >= 2) {
    const first = timed[0];
    const last = timed[timed.length - 1];
    const seconds = (new Date(last.timestamp).getTime() - new Date(first.timestamp).getTime()) / 1000;
    const steps = Number(last.global_step) - Number(first.global_step);
    if (seconds > 0 && steps > 0) stepsPerMinute = (steps / seconds) * 60;
  }

  const latestEval = [...rows].reverse().find((row) => Number.isFinite(Number(row.evalLoss)));
  const trainNearEval = latestEval
    ? [...trainRows].reverse().find((row) => row.step <= latestEval.step)
    : null;
  const evalGap =
    latestEval && trainNearEval && Number.isFinite(Number(trainNearEval.trainLoss))
      ? latestEval.evalLoss - trainNearEval.trainLoss
      : null;

  let verdict = "raccolgo dati";
  if (trendPct !== null) {
    if (trendPct < -3) verdict = "in miglioramento";
    else if (trendPct > 3) verdict = "peggiora";
    else verdict = "stabile";
  }
  if (volatility !== null && volatility > 0.85) verdict += " / rumoroso";

  return {
    recentAvg,
    previousAvg,
    trendPct,
    volatility,
    bestLoss: best?.trainLoss,
    bestStep: best?.step,
    stepsPerMinute,
    evalGap,
    etaSeconds: computeEta(history, currentStep, totalSteps),
    verdict,
  };
}

function linearTrend(rows, key) {
  const points = rows
    .map((row) => ({ x: Number(row.step), y: numberOrNull(row[key]) }))
    .filter((point) => Number.isFinite(point.x) && point.y !== null);
  if (points.length < 2) return { slope: null, projectedDelta: null, projectedPct: null };

  const xAvg = average(points.map((point) => point.x));
  const yAvg = average(points.map((point) => point.y));
  const denominator = points.reduce((sum, point) => sum + (point.x - xAvg) ** 2, 0);
  if (!denominator) return { slope: null, projectedDelta: null, projectedPct: null };

  const numerator = points.reduce((sum, point) => sum + (point.x - xAvg) * (point.y - yAvg), 0);
  const slope = numerator / denominator;
  const span = points[points.length - 1].x - points[0].x;
  const projectedDelta = slope * span;
  const projectedPct = points[0].y !== 0 ? (projectedDelta / points[0].y) * 100 : null;
  return { slope, projectedDelta, projectedPct };
}

function computePeriodAnalysis(history, rows, domain) {
  const left = domain ? Math.min(Number(domain[0]), Number(domain[1])) : -Infinity;
  const right = domain ? Math.max(Number(domain[0]), Number(domain[1])) : Infinity;
  const inRange = (step) => Number.isFinite(Number(step)) && Number(step) >= left && Number(step) <= right;
  const scopedRows = rows.filter((row) => inRange(row.step));
  const trainRows = scopedRows.filter((row) => Number.isFinite(Number(row.trainLoss)));
  if (!trainRows.length) return null;

  const first = trainRows[0];
  const last = trainRows[trainRows.length - 1];
  const losses = trainRows.map((row) => row.trainLoss);
  const avgLoss = average(losses);
  const volatility = standardDeviation(losses);
  const best = trainRows.reduce((bestRow, row) => (row.trainLoss < bestRow.trainLoss ? row : bestRow), trainRows[0]);
  const worst = trainRows.reduce((worstRow, row) => (row.trainLoss > worstRow.trainLoss ? row : worstRow), trainRows[0]);
  const deltaLoss = last.trainLoss - first.trainLoss;
  const deltaPct = first.trainLoss !== 0 ? (deltaLoss / first.trainLoss) * 100 : null;
  const trend = linearTrend(trainRows, "trainLoss");
  const movement = trend.projectedPct ?? deltaPct;
  const noiseRatio = avgLoss ? (volatility || 0) / avgLoss : 0;

  let behavior = "stabile";
  if (movement !== null) {
    if (movement < -5) behavior = "migliora netto";
    else if (movement < -1) behavior = "migliora leggero";
    else if (movement > 5) behavior = "peggiora netto";
    else if (movement > 1) behavior = "peggiora leggero";
  }
  if (noiseRatio > 0.12) behavior += " / instabile";

  const rawTrain = (Array.isArray(history) ? history : [])
    .filter((item) => item?.type === "train" && inRange(item.global_step))
    .sort((a, b) => Number(a.global_step) - Number(b.global_step));
  const timed = rawTrain.filter((item) => item.timestamp);
  let durationSeconds = null;
  let stepsPerMinute = null;
  if (timed.length >= 2) {
    const startTime = new Date(timed[0].timestamp).getTime();
    const endTime = new Date(timed[timed.length - 1].timestamp).getTime();
    const stepSpan = Number(timed[timed.length - 1].global_step) - Number(timed[0].global_step);
    durationSeconds = (endTime - startTime) / 1000;
    if (durationSeconds > 0 && stepSpan > 0) stepsPerMinute = (stepSpan / durationSeconds) * 60;
  }

  const tokens = rawTrain.map((item) => numberOrNull(item.tokens)).filter((value) => value !== null);
  const throughput = rawTrain
    .map((item) => numberOrNull(item.tokens_per_second ?? item.tokens_per_sec ?? item.tok_per_sec ?? item.tok_s))
    .filter((value) => value !== null);
  const evalRows = scopedRows.filter((row) => Number.isFinite(Number(row.evalLoss)));
  const latestEval = evalRows[evalRows.length - 1];
  const trainNearEval = latestEval ? [...trainRows].reverse().find((row) => row.step <= latestEval.step) : null;
  const evalGap =
    latestEval && trainNearEval && Number.isFinite(Number(trainNearEval.trainLoss))
      ? latestEval.evalLoss - trainNearEval.trainLoss
      : null;
  const lrFirst = trainRows.find((row) => Number.isFinite(Number(row.learningRate)));
  const lrLast = [...trainRows].reverse().find((row) => Number.isFinite(Number(row.learningRate)));
  const cpuOptimizerShare = rawTrain.length
    ? (rawTrain.filter((item) => item.cpu_optimizer_active === true).length / rawTrain.length) * 100
    : null;

  return {
    startStep: first.step,
    endStep: last.step,
    points: trainRows.length,
    behavior,
    firstLoss: first.trainLoss,
    lastLoss: last.trainLoss,
    deltaLoss,
    deltaPct,
    trendPct: trend.projectedPct,
    avgLoss,
    volatility,
    bestLoss: best.trainLoss,
    bestStep: best.step,
    worstLoss: worst.trainLoss,
    worstStep: worst.step,
    avgPerplexity: average(trainRows.map((row) => row.perplexity)),
    lrStart: lrFirst?.learningRate,
    lrEnd: lrLast?.learningRate,
    durationSeconds,
    stepsPerMinute,
    tokensAvg: average(tokens),
    tokensTotal: tokens.length ? tokens.reduce((sum, value) => sum + value, 0) : null,
    throughputAvg: average(throughput),
    evalCount: evalRows.length,
    evalGap,
    cpuOptimizerShare,
  };
}

function clampDomain(domain, rows) {
  if (!domain || !rows.length) return null;
  const minStep = rows[0].step;
  const maxStep = rows[rows.length - 1].step;
  const left = Math.max(minStep, Math.min(maxStep, Number(domain[0])));
  const right = Math.max(minStep, Math.min(maxStep, Number(domain[1])));
  if (!Number.isFinite(left) || !Number.isFinite(right) || left >= right) return null;
  return [left, right];
}

function Panel({ title, icon: Icon, action, children, className = "" }) {
  return (
    <section className={`relative rounded-3xl border border-white/5 bg-white/[0.02] backdrop-blur-md overflow-hidden ${className}`}>
      <div className="relative flex flex-col sm:flex-row sm:min-h-14 items-start sm:items-center justify-between border-b border-white/5 px-6 py-4 gap-4">
        <div className="flex items-center gap-3">
          {Icon ? <Icon size={18} className="text-white/50" /> : null}
          <h2 className="text-xs font-bold uppercase tracking-wider text-white/80">{title}</h2>
        </div>
        {action && <div className="flex items-center gap-2">{action}</div>}
      </div>
      <div className="relative p-6">{children}</div>
    </section>
  );
}

function HeaderInfo({ label, value }) {
  return (
    <div className="min-w-0 rounded-2xl border border-white/5 bg-white/[0.02] px-5 py-3.5 transition-colors duration-300 hover:bg-white/5">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-white/40">{label}</div>
      <div className="mt-1 truncate text-sm font-bold text-white/90">{value || "-"}</div>
    </div>
  );
}

function MetricTile({ icon: Icon, label, value, sub }) {
  return (
    <div className="group relative overflow-hidden rounded-3xl border border-white/5 bg-white/[0.02] backdrop-blur-md p-6 transition-colors duration-300 hover:bg-white/[0.04]">
      <div className="relative z-10 flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <Icon size={16} className="text-white/40" />
          <span className="text-[11px] font-semibold uppercase tracking-wider text-white/50">{label}</span>
        </div>
      </div>
      <div className="relative z-10 text-3xl sm:text-4xl font-bold tracking-tight text-white">{value}</div>
      {sub ? <div className="relative z-10 mt-2 text-xs font-medium text-white/40">{sub}</div> : null}
    </div>
  );
}

function ResourceRow({ icon: Icon, label, value, sub, percent, tone = "bg-white", glowColor = "transparent" }) {
  const width = Math.max(0, Math.min(100, Number.isFinite(percent) ? percent : 0));
  return (
    <div className="group py-2.5">
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2.5">
          <Icon size={16} className="text-white/40 group-hover:text-white/60 transition-colors" />
          <span className="text-[11px] font-semibold uppercase tracking-wider text-white/50 group-hover:text-white/70 transition-colors">{label}</span>
        </div>
        <div className="text-right">
          <span className="text-xs font-bold text-white/90">{value}</span>
          {sub && <span className="ml-2 text-[10px] font-medium text-white/40">{sub}</span>}
        </div>
      </div>
      <div className="relative h-1.5 rounded-full bg-black/40 shadow-inner overflow-hidden border border-white/5">
        <div 
          className={`absolute top-0 left-0 h-full rounded-full ${tone} transition-all duration-1000 shadow-[0_0_10px_rgba(255,255,255,0.3)]`} 
          style={{ width: `${width}%` }} 
        />
      </div>
    </div>
  );
}

function DetailRow({ label, value, highlight = false }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-white/5 py-3 last:border-b-0 group">
      <span className="text-[11px] font-semibold uppercase tracking-wider text-white/40 group-hover:text-white/60 transition-colors">{label}</span>
      <span className={`text-right text-xs font-medium ${highlight ? "text-white" : "text-white/70"}`}>{value || "-"}</span>
    </div>
  );
}

function SmallButton({ children, onClick, active = false, disabled = false }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`h-7 rounded-md border px-3 text-[10px] font-bold uppercase tracking-wider transition-all duration-300 ${
        disabled
          ? "cursor-not-allowed border-transparent bg-transparent text-white/20"
          : active
            ? "border-white/20 bg-white/10 text-white shadow-[0_0_15px_rgba(255,255,255,0.1)]"
            : "border-transparent bg-transparent text-white/50 hover:bg-white/5 hover:text-white"
      }`}
    >
      {children}
    </button>
  );
}

function ZoomStrip({ rows, domain, onDomainChange }) {
  const ref = useRef(null);
  const dragging = useRef(null);
  const minStep = rows[0]?.step ?? 0;
  const maxStep = rows[rows.length - 1]?.step ?? 1;
  const range = Math.max(1, maxStep - minStep);
  const effective = useMemo(() => domain || [minStep, maxStep], [domain, minStep, maxStep]);
  const leftPct = ((effective[0] - minStep) / range) * 100;
  const rightPct = ((effective[1] - minStep) / range) * 100;

  const stepFromEvent = useCallback(
    (event) => {
      const rect = ref.current?.getBoundingClientRect();
      if (!rect) return null;
      const clientX = event.touches?.[0]?.clientX ?? event.clientX;
      const percent = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      return minStep + percent * range;
    },
    [minStep, range]
  );

  useEffect(() => {
    const onMove = (event) => {
      if (!dragging.current) return;
      const step = stepFromEvent(event);
      if (step === null) return;
      const minWidth = Math.max(10, range * 0.02);
      if (dragging.current === "left") {
        onDomainChange([Math.min(step, effective[1] - minWidth), effective[1]]);
      } else if (dragging.current === "right") {
        onDomainChange([effective[0], Math.max(step, effective[0] + minWidth)]);
      } else {
        const width = effective[1] - effective[0];
        const nextLeft = Math.max(minStep, Math.min(maxStep - width, step - width / 2));
        onDomainChange([nextLeft, nextLeft + width]);
      }
    };
    const onUp = () => {
      dragging.current = null;
      document.body.style.cursor = "";
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [effective, maxStep, minStep, onDomainChange, range, stepFromEvent]);

  if (rows.length < 2) return null;

  return (
    <div className="mt-6">
      <div
        ref={ref}
        className="relative h-4 rounded-full border border-white/10 bg-black/40 shadow-inner cursor-pointer"
        onMouseDown={(event) => {
          dragging.current = "move";
          document.body.style.cursor = "ew-resize";
          const step = stepFromEvent(event);
          if (step !== null) {
            const width = effective[1] - effective[0];
            const nextLeft = Math.max(minStep, Math.min(maxStep - width, step - width / 2));
            onDomainChange([nextLeft, nextLeft + width]);
          }
        }}
      >
        <div
          className="absolute inset-y-0 rounded-full bg-gradient-to-r from-white/10 via-white/30 to-white/10 shadow-[0_0_10px_rgba(255,255,255,0.15)] transition-all duration-75"
          style={{ left: `${leftPct}%`, width: `${Math.max(2, rightPct - leftPct)}%` }}
        />
        <button
          type="button"
          className="absolute top-1/2 h-5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/20 bg-white shadow-[0_0_10px_rgba(255,255,255,0.8)] flex items-center justify-center hover:scale-110 transition-transform"
          style={{ left: `${leftPct}%` }}
          onMouseDown={(event) => {
            event.stopPropagation();
            dragging.current = "left";
            document.body.style.cursor = "ew-resize";
          }}
        >
          <div className="w-0.5 h-2.5 bg-black/30 rounded-full" />
        </button>
        <button
          type="button"
          className="absolute top-1/2 h-5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/20 bg-white shadow-[0_0_10px_rgba(255,255,255,0.8)] flex items-center justify-center hover:scale-110 transition-transform"
          style={{ left: `${rightPct}%` }}
          onMouseDown={(event) => {
            event.stopPropagation();
            dragging.current = "right";
            document.body.style.cursor = "ew-resize";
          }}
        >
          <div className="w-0.5 h-2.5 bg-black/30 rounded-full" />
        </button>
      </div>
      <div className="mt-2.5 flex justify-between text-[10px] font-bold text-white/40 px-1">
        <span>{Math.round(effective[0])}</span>
        <span>{Math.round(effective[1])}</span>
      </div>
    </div>
  );
}

function ChartTooltip({ active, payload, label }) {
  const rows = (payload || []).filter((item) => Number.isFinite(Number(item.value)));
  if (!active || rows.length === 0) return null;

  return (
    <div className="rounded-2xl border border-white/10 bg-black/80 backdrop-blur-xl px-5 py-4 shadow-xl">
      <div className="mb-3 flex items-center gap-2 border-b border-white/10 pb-2">
        <div className="h-2 w-2 rounded-full bg-white animate-pulse" />
        <div className="text-[10px] font-bold uppercase tracking-wider text-white/70">
          Step {label}
        </div>
      </div>
      <div className="space-y-2">
        {rows.map((item) => (
          <div key={item.dataKey} className="flex items-center justify-between gap-8 text-[11px]">
            <div className="flex items-center gap-2">
              <div className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: item.stroke || item.fill || "white" }} />
              <span className="font-semibold text-white/60">{item.name || item.dataKey}</span>
            </div>
            <span className="font-mono font-bold text-white">{formatFixed(item.value, item.dataKey === "learningRate" ? 8 : 4)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function chartStep(state) {
  return numberOrNull(state?.activeLabel ?? state?.activePayload?.[0]?.payload?.step);
}

function HighlightArea({ domain }) {
  const left = numberOrNull(domain?.[0]);
  const right = numberOrNull(domain?.[1]);
  if (left === null || right === null || left === right) return null;
  return (
    <ReferenceArea
      x1={Math.min(left, right)}
      x2={Math.max(left, right)}
      stroke="rgba(255,255,255,0.4)"
      strokeOpacity={0.8}
      fill="#ffffff"
      fillOpacity={0.08}
    />
  );
}

function TrainingChart({ rows, selectionDomain, selectionAnalysis, isDragging, onApplyZoom, onClearSelection, onSelectionStart, onSelectionMove, onSelectionEnd }) {
  return (
    <div className="h-[380px] select-none cursor-crosshair relative">
      {selectionAnalysis && (
        <div className="absolute top-2 left-1/2 -translate-x-1/2 z-50 pointer-events-auto">
          <div className="bg-black/80 backdrop-blur-md border border-white/10 px-4 py-2 rounded-xl flex items-center gap-4 text-xs font-medium text-white shadow-xl">
            <span className="text-white/50">Punti:</span> {selectionAnalysis.points}
            <div className="w-px h-3 bg-white/20" />
            <span className="text-white/50">Δ Loss:</span> {formatSigned(selectionAnalysis.deltaLoss, 4)} ({formatSigned(selectionAnalysis.deltaPct, 1, "%")})
            <div className="w-px h-3 bg-white/20" />
            <span className={selectionAnalysis.behavior?.includes("migliora") ? "text-emerald-400" : selectionAnalysis.behavior?.includes("peggiora") ? "text-red-400" : "text-white"}>{selectionAnalysis.behavior}</span>
            {!isDragging && (
              <>
                <div className="w-px h-3 bg-white/20" />
                <button onClick={(e) => { e.stopPropagation(); onApplyZoom?.(); }} className="px-2 py-0.5 bg-white/10 hover:bg-white/20 rounded text-white font-bold transition-colors">Zoom</button>
                <button onClick={(e) => { e.stopPropagation(); onClearSelection?.(); }} className="px-2 py-0.5 bg-red-500/20 text-red-300 hover:bg-red-500/40 rounded font-bold transition-colors">X</button>
              </>
            )}
          </div>
        </div>
      )}
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={rows}
          margin={{ top: 20, right: 20, left: -10, bottom: 0 }}
          onMouseDown={(state) => onSelectionStart?.(state)}
          onMouseMove={(state) => onSelectionMove?.(state)}
          onMouseUp={(state) => onSelectionEnd?.(state)}
          onMouseLeave={() => onSelectionEnd?.()}
        >
          <defs>
            <linearGradient id="colorLossSmooth" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ffffff" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#ffffff" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorLossRaw" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ffffff" stopOpacity={0.05} />
              <stop offset="95%" stopColor="#ffffff" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorEval" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#60a5fa" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#60a5fa" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(255,255,255,0.03)" vertical={false} strokeDasharray="4 4" />
          <XAxis dataKey="step" tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10, fontWeight: "600" }} tickLine={false} axisLine={false} minTickGap={30} />
          <YAxis tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10, fontWeight: "600" }} tickLine={false} axisLine={false} width={45} />
          <Tooltip content={<ChartTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.2)', strokeWidth: 1, strokeDasharray: '4 4' }} />
          <HighlightArea domain={selectionDomain} />
          <Area name="Loss" type="monotone" dataKey="trainLossSmooth" stroke="#ffffff" strokeWidth={2.5} fill="url(#colorLossSmooth)" connectNulls dot={false} isAnimationActive={false} activeDot={{ r: 5, fill: "#fff", stroke: "rgba(255,255,255,0.3)", strokeWidth: 6 }} />
          <Area name="Raw" type="monotone" dataKey="trainLoss" stroke="rgba(255,255,255,0.15)" strokeWidth={1} fill="url(#colorLossRaw)" connectNulls dot={false} isAnimationActive={false} />
          <Area name="Eval" type="monotone" dataKey="evalLoss" stroke="#60a5fa" strokeWidth={2.5} fill="url(#colorEval)" connectNulls dot={false} isAnimationActive={false} activeDot={{ r: 5, fill: "#60a5fa", stroke: "rgba(96,165,250,0.3)", strokeWidth: 6 }} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function LearningChart({ rows, selectionDomain, selectionAnalysis, isDragging, onApplyZoom, onClearSelection, onSelectionStart, onSelectionMove, onSelectionEnd }) {
  return (
    <div className="h-[220px] select-none cursor-crosshair relative">
      {selectionAnalysis && (
        <div className="absolute top-2 left-1/2 -translate-x-1/2 z-50 pointer-events-auto">
          <div className="bg-black/80 backdrop-blur-md border border-white/10 px-4 py-2 rounded-xl flex items-center gap-4 text-xs font-medium text-white shadow-xl">
            <span className="text-white/50">Punti:</span> {selectionAnalysis.points}
            <div className="w-px h-3 bg-white/20" />
            <span className="text-white/50">Δ LR:</span> {formatCompact(selectionAnalysis.lrEnd - selectionAnalysis.lrStart)}
            {!isDragging && (
              <>
                <div className="w-px h-3 bg-white/20" />
                <button onClick={(e) => { e.stopPropagation(); onApplyZoom?.(); }} className="px-2 py-0.5 bg-white/10 hover:bg-white/20 rounded text-white font-bold transition-colors">Zoom</button>
                <button onClick={(e) => { e.stopPropagation(); onClearSelection?.(); }} className="px-2 py-0.5 bg-red-500/20 text-red-300 hover:bg-red-500/40 rounded font-bold transition-colors">X</button>
              </>
            )}
          </div>
        </div>
      )}
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={rows}
          margin={{ top: 20, right: 20, left: -10, bottom: 0 }}
          onMouseDown={(state) => onSelectionStart?.(state)}
          onMouseMove={(state) => onSelectionMove?.(state)}
          onMouseUp={(state) => onSelectionEnd?.(state)}
          onMouseLeave={() => onSelectionEnd?.()}
        >
          <defs>
            <linearGradient id="colorLR" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(255,255,255,0.03)" vertical={false} strokeDasharray="4 4" />
          <XAxis dataKey="step" tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10, fontWeight: "600" }} tickLine={false} axisLine={false} minTickGap={30} />
          <YAxis tickFormatter={(value) => Number(value).toExponential(1)} tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10, fontWeight: "600" }} tickLine={false} axisLine={false} width={55} />
          <Tooltip content={<ChartTooltip />} cursor={{ stroke: 'rgba(16,185,129,0.3)', strokeWidth: 1, strokeDasharray: '4 4' }} />
          <HighlightArea domain={selectionDomain} />
          <Area name="LR" type="monotone" dataKey="learningRateSmooth" stroke="#10b981" strokeWidth={2.5} fill="url(#colorLR)" connectNulls dot={false} isAnimationActive={false} activeDot={{ r: 5, fill: "#10b981", stroke: "rgba(16,185,129,0.3)", strokeWidth: 6 }} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function ChartControls({ rows, domain, onDomainChange, smoothing, onSmoothingChange, onReset }) {
  const maxStep = rows[rows.length - 1]?.step ?? 0;
  const setLast = (count) => {
    if (!maxStep) return;
    onDomainChange([Math.max(rows[0]?.step ?? 0, maxStep - count), maxStep]);
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <div className="flex items-center gap-1 bg-white/[0.02] rounded-lg p-1 border border-white/5 shadow-inner">
        <SmallButton onClick={() => setLast(100)}>100</SmallButton>
        <SmallButton onClick={() => setLast(300)}>300</SmallButton>
        <SmallButton onClick={() => onDomainChange(null)} active={!domain}>Tutto</SmallButton>
      </div>
      <SmallButton onClick={onReset}>Reset</SmallButton>
      <div className="ml-0 flex h-9 items-center gap-3 rounded-lg border border-white/5 bg-white/[0.02] px-4 lg:ml-2 shadow-inner">
        <span className="text-[10px] font-bold uppercase tracking-wider text-white/40">Smooth</span>
        <input
          type="range"
          min="1"
          max="80"
          value={smoothing}
          onChange={(event) => onSmoothingChange(Number(event.target.value))}
          className="h-1.5 w-24 cursor-pointer appearance-none rounded-full bg-black/50 shadow-inner outline-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(255,255,255,0.8)] [&::-webkit-slider-thumb]:hover:scale-110 [&::-webkit-slider-thumb]:transition-transform"
        />
        <span className="w-6 text-right font-mono text-[10px] font-bold text-white/70">{smoothing}</span>
      </div>
    </div>
  );
}

function ControlButton({ icon: Icon, label, onClick, disabled = false, busy = false, tone = "default" }) {
  const tones = {
    default: "border-white/5 bg-white/[0.02] text-white/70 hover:bg-white/10 hover:text-white",
    good: "border-emerald-500/20 bg-emerald-500/5 text-emerald-300 hover:bg-emerald-500/20",
    warn: "border-amber-500/20 bg-amber-500/5 text-amber-300 hover:bg-amber-500/20",
    danger: "border-red-500/20 bg-red-500/5 text-red-300 hover:bg-red-500/20",
  };

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled || busy}
      className={`inline-flex h-10 min-w-32 items-center justify-center gap-2 rounded-xl border px-5 text-[11px] font-bold uppercase tracking-wider transition-colors ${
        disabled || busy ? "cursor-not-allowed border-white/5 bg-white/[0.02] text-white/20" : tones[tone]
      }`}
    >
      <Icon size={16} className={busy ? "animate-spin" : ""} />
      {label}
    </button>
  );
}

export default function Home() {
  const [data, setData] = useState({ status: "waiting", current: {}, history: [], epochs: [] });
  const [refreshing, setRefreshing] = useState(false);
  const [lossDomain, setLossDomain] = useState(null);
  const [lrDomain, setLrDomain] = useState(null);
  const [lossSmoothing, setLossSmoothing] = useState(16);
  const [lrSmoothing, setLrSmoothing] = useState(4);
  const [zoomInitialized, setZoomInitialized] = useState(false);
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [dragSelection, setDragSelection] = useState(null);
  const [control, setControl] = useState({ running: null, processes: null });
  const [controlBusy, setControlBusy] = useState(null);
  const [controlMessage, setControlMessage] = useState("");
  const selectionRef = useRef({ start: null, end: null });

  const refreshData = useCallback(async () => {
    setRefreshing(true);
    try {
      const response = await fetch("/api/metrics", { cache: "no-store" });
      if (!response.ok) throw new Error("metrics unavailable");
      const payload = await response.json();
      const controlResponse = await fetch("/api/training/control", { cache: "no-store" });
      if (controlResponse.ok) {
        setControl(await controlResponse.json());
      }
      setData(payload);
      if (!zoomInitialized) {
        setLossDomain(payload.lossDomain || null);
        setLrDomain(payload.lrDomain || null);
        setZoomInitialized(true);
      }
    } catch {
      setData((current) => ({ ...current, status: "offline" }));
    } finally {
      setRefreshing(false);
    }
  }, [zoomInitialized]);

  useEffect(() => {
    const initial = setTimeout(refreshData, 0);
    const timer = setInterval(refreshData, 3000);
    return () => {
      clearTimeout(initial);
      clearInterval(timer);
    };
  }, [refreshData]);

  const history = useMemo(() => (Array.isArray(data.history) ? data.history : []), [data.history]);
  const latest = latestFrom(data);
  const allRows = useMemo(() => metricRows(history), [history]);
  const lossRows = useMemo(() => movingAverageRows(allRows, lossSmoothing), [allRows, lossSmoothing]);
  const lrRowsAll = useMemo(() => movingAverageRows(allRows, lrSmoothing).filter((row) => row.learningRate !== null), [allRows, lrSmoothing]);

  const currentStep = Number(data.current?.global_step ?? latest.global_step ?? data.start_global_step ?? 0);
  const totalSteps = Number(data.total_steps || latest.total_steps || 0);
  const progress = totalSteps > 0 ? Math.max(0, Math.min(100, (currentStep / totalSteps) * 100)) : 0;
  const etaSeconds = computeEta(history, currentStep, totalSteps);
  const analysis = useMemo(() => computeAnalysis(history, allRows, currentStep, totalSteps), [history, allRows, currentStep, totalSteps]);

  const system = data.system || {};
  const config = data.config || {};
  const stage = data.pipeline_step?.id || config.pipeline_step || "manual";
  const stageTitle = data.pipeline_step?.title || stage;
  const gpuName = system.gpu_name || data.hardware?.gpu || "GPU";
  const currentVram = numberOrNull(system.gpu_memory_used_gb ?? latest.vram_reserved_gb ?? latest.vram_allocated_gb);
  const totalVram = numberOrNull(system.gpu_memory_total_gb);
  const vramPercent = totalVram ? (currentVram / totalVram) * 100 : null;
  const isRunning = data.status === "running";
  const controlHasProcessList = Array.isArray(control.processes);
  const trainingRunning = controlHasProcessList ? Boolean(control.running) : isRunning;
  const trainingPaused = Boolean(control.paused);
  const displayStatus = trainingPaused ? "paused" : trainingRunning ? "running" : data.status || "waiting";
  const controlAction = control.control?.action || "-";
  const controlPidText = Array.isArray(control.pids) && control.pids.length ? control.pids.join(", ") : "-";
  const latestTokens = Number.isFinite(Number(latest.tokens))
    ? Number(latest.tokens).toLocaleString("it-IT")
    : "-";
  const batchText = `${config.per_device_batch_size || "-"} x ${config.gradient_accumulation_steps || "-"}`;
  const saveText = config.save_steps ? `ogni ${config.save_steps} step` : "-";
  const cpuOptText = latest.cpu_optimizer_active ? "attivo" : config.cpu_optimizer || "auto";
  const historyText = history.length.toLocaleString("it-IT");
  const safeLossDomain = clampDomain(lossDomain, lossRows);
  const safeLrDomain = clampDomain(lrDomain, lrRowsAll);
  const safeSelectedDomain = clampDomain(selectedDomain, allRows);
  const isDragging = !!dragSelection;
  const activeSelectionDomain = dragSelection || safeSelectedDomain;
  const periodDomain = safeSelectedDomain || safeLossDomain;
  const periodMode = safeSelectedDomain ? "selezione" : safeLossDomain ? "zoom visibile" : "run completo";
  const periodAnalysis = useMemo(() => computePeriodAnalysis(history, allRows, periodDomain), [history, allRows, periodDomain]);
  const selectionAnalysis = useMemo(() => activeSelectionDomain ? computePeriodAnalysis(history, allRows, activeSelectionDomain) : null, [history, allRows, activeSelectionDomain]);

  const visibleLossRows = safeLossDomain
    ? lossRows.filter((row) => row.step >= safeLossDomain[0] && row.step <= safeLossDomain[1])
    : lossRows;
  const visibleLrRows = safeLrDomain
    ? lrRowsAll.filter((row) => row.step >= safeLrDomain[0] && row.step <= safeLrDomain[1])
    : lrRowsAll;

  function persistZoom(patch) {
    fetch("/api/zoom", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }).catch(() => {});
  }

  function applyLossDomain(domain) {
    const next = domain ? clampDomain(domain, lossRows) : null;
    setLossDomain(next);
    persistZoom({ lossDomain: next });
  }

  function applyLrDomain(domain) {
    const next = domain ? clampDomain(domain, lrRowsAll) : null;
    setLrDomain(next);
    persistZoom({ lrDomain: next });
  }

  function beginAreaSelection(state) {
    const step = chartStep(state);
    if (step === null) return;
    selectionRef.current = { start: step, end: step };
    setDragSelection([step, step]);
  }

  function moveAreaSelection(state) {
    if (selectionRef.current.start === null) return;
    const step = chartStep(state);
    if (step === null) return;
    selectionRef.current = { ...selectionRef.current, end: step };
    setDragSelection([selectionRef.current.start, step].sort((a, b) => a - b));
  }

  function finishAreaSelection(state) {
    const step = chartStep(state);
    if (step !== null && selectionRef.current.start !== null) {
      selectionRef.current = { ...selectionRef.current, end: step };
    }
    const { start, end } = selectionRef.current;
    selectionRef.current = { start: null, end: null };
    setDragSelection(null);
    if (start === null || end === null || Math.abs(end - start) < 1) return;
    const next = clampDomain([Math.min(start, end), Math.max(start, end)], allRows);
    if (next) setSelectedDomain(next);
  }

  function zoomToSelectedPeriod() {
    if (!safeSelectedDomain) return;
    applyLossDomain(safeSelectedDomain);
    applyLrDomain(safeSelectedDomain);
  }

  async function sendTrainingControl(action) {
    setControlBusy(action);
    setControlMessage("");
    try {
      const response = await fetch("/api/training/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, stage, currentStep }),
      });
      const payload = await response.json();
      if (!response.ok || payload.ok === false) {
        setControlMessage(payload.error || "Comando non riuscito");
      } else {
        setControl(payload);
        const labels = {
          pause: "Training in pausa",
          resume: "Training ripreso",
          save: "Salvataggio richiesto",
        };
        setControlMessage(labels[action] || "Comando inviato");
      }
      await refreshData();
    } catch (error) {
      setControlMessage(error.message || "Errore comando");
    } finally {
      setControlBusy(null);
    }
  }

  return (
    <main className="min-h-screen w-full bg-[#0a0a0a] text-white overflow-x-hidden font-sans pb-24 selection:bg-white/20">
      <div className="relative z-10 mx-auto max-w-screen-2xl px-4 py-8 sm:px-6 lg:px-8 space-y-8">
        
        <header className="relative flex flex-col md:flex-row items-start md:items-center justify-between gap-6 pb-4 border-b border-white/5">
          <div className="relative min-w-0 flex-1">
            <div className="mb-3 flex flex-wrap items-center gap-3">
              <Badge className={`rounded-lg px-3 py-1 text-[10px] font-bold uppercase tracking-wider ${trainingRunning ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-white/5 text-white/50 border border-white/10"}`}>
                <span className="flex items-center gap-2">
                  {trainingRunning && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />}
                  {displayStatus}
                </span>
              </Badge>
              <span className="text-xs font-medium text-white/50">{stageTitle}</span>
            </div>
            <h1 className="text-3xl font-bold tracking-tight text-white md:text-4xl">Jarvis Dashboard</h1>
          </div>

          <div className="relative flex flex-wrap items-center gap-3 mt-2 md:mt-0">
            <button
              type="button"
              onClick={refreshData}
              className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-white/5 bg-white/[0.02] text-white/60 transition-colors hover:bg-white/10 hover:text-white"
              aria-label="Aggiorna dashboard"
            >
              <RefreshCcw size={16} className={refreshing ? "animate-spin" : ""} />
            </button>
            <Link
              href="/test-model"
              className="inline-flex h-10 items-center gap-2 rounded-xl bg-white px-5 text-[11px] font-bold uppercase tracking-wider text-black transition-colors hover:bg-white/90"
            >
              <Bot size={16} />
              Test Model
              <ArrowRight size={14} className="text-black/50" />
            </Link>
          </div>
        </header>

        <section className="grid grid-cols-1 xl:grid-cols-[1fr_auto] gap-6">
          <div className="rounded-3xl border border-white/5 bg-white/[0.02] backdrop-blur-md p-6 sm:p-8 relative overflow-hidden group">
            <div className="relative mb-5 flex flex-wrap items-end justify-between gap-6">
              <div>
                <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wider text-white/40 mb-2">
                  <Play size={14} /> Progresso
                </div>
                <div className="text-xl font-bold text-white/90">
                  Step <span className="text-white text-2xl">{currentStep.toLocaleString("it-IT")}</span> <span className="text-white/30 text-lg">/ {totalSteps ? totalSteps.toLocaleString("it-IT") : "-"}</span>
                </div>
              </div>
              <div className="text-right">
                <div className="text-5xl font-bold tracking-tighter text-white">
                  {formatFixed(progress, 1)}%
                </div>
                <div className="text-xs font-medium text-white/40 mt-1">ETA <span className="text-white/70">{formatDuration(etaSeconds)}</span></div>
              </div>
            </div>
            <div className="relative h-2.5 rounded-full bg-black/40 shadow-inner overflow-hidden mb-6 border border-white/5">
              <div 
                className="absolute top-0 left-0 h-full rounded-full bg-gradient-to-r from-white/40 via-white to-white shadow-[0_0_15px_rgba(255,255,255,0.5)] transition-all duration-1000" 
                style={{ width: `${progress}%` }} 
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-4 pt-4 border-t border-white/5">
              <div className="flex flex-wrap gap-2">
                <ControlButton
                  icon={Pause}
                  label="Pausa"
                  tone="warn"
                  disabled={!trainingRunning || trainingPaused}
                  busy={controlBusy === "pause"}
                  onClick={() => sendTrainingControl("pause")}
                />
                <ControlButton
                  icon={Play}
                  label="Riprendi"
                  tone="good"
                  disabled={!trainingRunning || !trainingPaused}
                  busy={controlBusy === "resume"}
                  onClick={() => sendTrainingControl("resume")}
                />
                <ControlButton
                  icon={Save}
                  label="Salva ora"
                  disabled={!trainingRunning || trainingPaused}
                  busy={controlBusy === "save"}
                  onClick={() => sendTrainingControl("save")}
                />
              </div>
              {controlMessage && (
                <div className="text-[11px] font-bold uppercase tracking-wider text-white/50 animate-pulse">
                  {controlMessage}
                </div>
              )}
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-2 gap-4 w-full xl:w-[460px]">
            <HeaderInfo label="Stage" value={stage} />
            <HeaderInfo label="Batch" value={batchText} />
            <HeaderInfo label="Save" value={saveText} />
            <HeaderInfo label="CPU opt" value={cpuOptText} />
            <HeaderInfo label="History" value={historyText} />
            <HeaderInfo label="PID" value={controlPidText} />
          </div>
        </section>

        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricTile icon={Gauge} label="Loss" value={formatLoss(latest.loss)} sub="Ultimo batch" />
          <MetricTile icon={LineChartIcon} label="Perplexity" value={formatFixed(latest.perplexity, 1)} sub="Perplexity corrente" />
          <MetricTile icon={Zap} label="Learning Rate" value={formatCompact(latest.learning_rate)} sub="Scheduler attivo" />
          <MetricTile icon={Activity} label="Token" value={latestTokens} sub="Ultimo optimizer step" />
        </section>

        <div className="grid grid-cols-1 xl:grid-cols-[1fr_380px] 2xl:grid-cols-[1fr_420px] gap-8">
          
          <div className="space-y-8 min-w-0">
            <Panel
              title="Andamento Loss Globale"
              icon={LineChartIcon}
              action={
                <ChartControls
                  rows={lossRows}
                  domain={safeLossDomain}
                  onDomainChange={applyLossDomain}
                  smoothing={lossSmoothing}
                  onSmoothingChange={setLossSmoothing}
                  onReset={() => applyLossDomain(null)}
                />
              }
            >
              <TrainingChart
                rows={visibleLossRows}
                selectionDomain={activeSelectionDomain}
                selectionAnalysis={selectionAnalysis}
                isDragging={isDragging}
                onApplyZoom={zoomToSelectedPeriod}
                onClearSelection={() => setSelectedDomain(null)}
                onSelectionStart={beginAreaSelection}
                onSelectionMove={moveAreaSelection}
                onSelectionEnd={finishAreaSelection}
              />
              <ZoomStrip rows={lossRows} domain={safeLossDomain} onDomainChange={applyLossDomain} />
            </Panel>

            <Panel
              title="Traiettoria Learning Rate"
              icon={Zap}
              action={
                <ChartControls
                  rows={lrRowsAll}
                  domain={safeLrDomain}
                  onDomainChange={applyLrDomain}
                  smoothing={lrSmoothing}
                  onSmoothingChange={setLrSmoothing}
                  onReset={() => applyLrDomain(null)}
                />
              }
            >
              <LearningChart
                rows={visibleLrRows}
                selectionDomain={activeSelectionDomain}
                selectionAnalysis={selectionAnalysis}
                isDragging={isDragging}
                onApplyZoom={zoomToSelectedPeriod}
                onClearSelection={() => setSelectedDomain(null)}
                onSelectionStart={beginAreaSelection}
                onSelectionMove={moveAreaSelection}
                onSelectionEnd={finishAreaSelection}
              />
              <ZoomStrip rows={lrRowsAll} domain={safeLrDomain} onDomainChange={applyLrDomain} />
            </Panel>
          </div>

          <aside className="space-y-6">
            
            <Panel title="Risorse" icon={Monitor}>
              <div className="space-y-1">
                <ResourceRow icon={Cpu} label="CPU" value={`${formatFixed(system.cpu_percent, 1)}%`} percent={system.cpu_percent} />
                <ResourceRow icon={Monitor} label="GPU" value={`${formatFixed(system.gpu_util_percent, 1)}%`} percent={system.gpu_util_percent} tone="bg-white" />
                <ResourceRow icon={Database} label="VRAM" value={`${formatGb(currentVram)} / ${formatGb(totalVram)}`} percent={vramPercent} tone="bg-white/70" />
                <ResourceRow icon={MemoryStick} label="RAM" value={`${formatGb(system.ram_used_gb)}`} percent={system.ram_percent} tone="bg-white/40" />
              </div>
              <div className="mt-5 pt-4 border-t border-white/5 space-y-1">
                <DetailRow label="Temp GPU" value={formatUnit(system.gpu_temp_c, " C", 0)} />
                <DetailRow label="Power" value={formatUnit(system.gpu_power_w, " W", 1)} />
              </div>
            </Panel>

            <Panel title="Trend Generale" icon={Activity}>
              <div className="mb-4 rounded-xl bg-white/[0.02] p-4 border border-white/5 flex items-center justify-between">
                <span className="text-[10px] font-bold uppercase tracking-wider text-white/40">Verdetto</span>
                <span className={`text-sm font-bold ${
                  analysis.verdict?.includes('miglioramento') ? 'text-emerald-400' :
                  analysis.verdict?.includes('peggiora') ? 'text-red-400' : 'text-white'
                }`}>{analysis.verdict}</span>
              </div>
              <div className="space-y-1">
                <DetailRow label="Trend 120" value={analysis.trendPct === null ? "-" : `${formatFixed(analysis.trendPct, 1)}%`} />
                <DetailRow label="Loss media" value={formatLoss(analysis.recentAvg)} />
                <DetailRow label="Volatilità" value={formatLoss(analysis.volatility)} />
                <DetailRow label="Velocità" value={`${formatFixed(analysis.stepsPerMinute, 1)} step/min`} />
                <DetailRow label="Eval Gap" value={analysis.evalGap === null ? "-" : formatLoss(analysis.evalGap)} />
              </div>
            </Panel>

          </aside>
        </div>
      </div>
    </main>
  );
}
