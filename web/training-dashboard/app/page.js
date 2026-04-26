"use client";

import { useEffect, useMemo, useState, useRef, useCallback } from "react";
import {
  Activity,
  Clock3,
  Gauge,
  LineChart as LineChartIcon,
  Radio,
  Server,
  Zap,
} from "lucide-react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis, ResponsiveContainer } from "recharts";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

function useMediaQuery(query) {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const media = window.matchMedia(query);
    const update = () => setMatches(media.matches);

    update();
    media.addEventListener("change", update);

    return () => media.removeEventListener("change", update);
  }, [query]);

  return matches;
}

const chartConfig = {
  trainLoss: {
    label: "Train loss",
    color: "var(--chart-1)",
  },
  evalLoss: {
    label: "Eval loss",
    color: "var(--chart-2)",
  },
  learningRate: {
    label: "Learning rate",
    color: "var(--chart-3)",
  },
};

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }

  return Number(value).toLocaleString("it-IT", {
    maximumFractionDigits: digits,
  });
}

function formatCompact(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }

  return Number(value).toExponential(2);
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "-";

  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;

  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${rest}s`;

  return `${rest}s`;
}

function lastItem(items, predicate = () => true) {
  for (let index = items.length - 1; index >= 0; index -= 1) {
    if (predicate(items[index])) return items[index];
  }

  return {};
}

// Gentle downsample: keeps max 3000 points to ensure smooth zoom without losing detail
function gentleDownsample(data, threshold = 3000) {
  if (!data || data.length <= threshold) return data;
  const step = Math.ceil(data.length / threshold);
  return data.filter((_, index) => index % step === 0);
}

function MetricTile({ icon: Icon, label, value, detail, tone = "cyan" }) {
  const tones = {
    cyan: {
      icon: "text-cyan-300 bg-cyan-400/10 ring-cyan-400/30",
      glow: "bg-cyan-500",
      border: "hover:border-cyan-500/30",
    },
    amber: {
      icon: "text-amber-300 bg-amber-400/10 ring-amber-400/30",
      glow: "bg-amber-500",
      border: "hover:border-amber-500/30",
    },
    violet: {
      icon: "text-violet-300 bg-violet-400/10 ring-violet-400/30",
      glow: "bg-violet-500",
      border: "hover:border-violet-500/30",
    },
    emerald: {
      icon: "text-emerald-300 bg-emerald-400/10 ring-emerald-400/30",
      glow: "bg-emerald-500",
      border: "hover:border-emerald-500/30",
    },
  };

  const style = tones[tone];

  return (
    <Card className={`group relative overflow-hidden rounded-2xl border border-white/10 bg-black/40 py-0 shadow-2xl backdrop-blur-xl transition-all duration-500 hover:-translate-y-1 hover:bg-black/60 hover:shadow-[0_8px_30px_rgb(0,0,0,0.5)] ${style.border}`}>
      <div className={`absolute -right-12 -top-12 size-32 rounded-full opacity-10 blur-[40px] transition-all duration-700 group-hover:opacity-40 group-hover:scale-150 ${style.glow}`} />
      <CardContent className="relative grid min-h-24 grid-cols-[minmax(0,1fr)_auto] items-center gap-4 p-5 sm:p-6">
        <div className="min-w-0">
          <div className="text-[11px] font-bold tracking-widest uppercase text-zinc-400">
            {label}
          </div>
          <div className="mt-1 truncate text-3xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-br from-white to-white/60 sm:text-4xl drop-shadow-sm">
            {value}
          </div>
          <div className="mt-1.5 truncate text-[12px] font-medium text-zinc-500">{detail}</div>
        </div>

        <div className={`flex size-12 items-center justify-center rounded-2xl ring-1 shadow-inner transition-all duration-500 group-hover:scale-110 group-hover:rotate-3 sm:size-14 ${style.icon}`}>
          <Icon className="size-6 drop-shadow-[0_0_8px_rgba(255,255,255,0.3)]" />
        </div>
      </CardContent>
    </Card>
  );
}

function EmptyChart({ text }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center text-sm font-medium text-zinc-500 tracking-wide">
      {text}
    </div>
  );
}

function ChartCard({ title, description, children, empty }) {
  return (
    <Card className="group relative overflow-hidden rounded-2xl border border-white/10 bg-black/40 py-0 shadow-2xl backdrop-blur-xl transition-all duration-500 hover:border-white/20">
      <div className="absolute inset-0 bg-gradient-to-b from-white/[0.03] to-transparent opacity-0 transition-opacity duration-500 group-hover:opacity-100" />
      <CardHeader className="relative border-b border-white/5 px-5 py-5">
        <CardTitle className="text-base font-bold tracking-tight text-white drop-shadow-sm">
          {title}
        </CardTitle>
        <CardDescription className="text-xs font-medium text-zinc-400 mt-1">{description}</CardDescription>
      </CardHeader>

      <CardContent className="relative overflow-hidden p-3 sm:p-5">
        {empty ? <EmptyChart text={empty} /> : children}
      </CardContent>
    </Card>
  );
}

function ZoomMinimap({ data, dataKey, domain, onDomainChange, color, height = 36 }) {
  const containerRef = useRef(null);
  const dragging = useRef(null);
  const dragOrigin = useRef(null);
  const domainAtDragStart = useRef(null);

  const steps = useMemo(() => data.map((d) => d.step), [data]);
  const values = useMemo(() => data.map((d) => d[dataKey] ?? 0), [data, dataKey]);
  const minStep = steps[0] ?? 0;
  const maxStep = steps[steps.length - 1] ?? 1;
  const range = maxStep - minStep || 1;

  // Selection bounds as fractions [0..1]
  const selLeft = domain ? (domain[0] - minStep) / range : 0;
  const selRight = domain ? (domain[1] - minStep) / range : 1;

  // Build SVG polyline points for sparkline
  const polyline = useMemo(() => {
    if (!values.length) return "";
    const vMin = Math.min(...values);
    const vMax = Math.max(...values);
    const vRange = vMax - vMin || 1;
    return data
      .map((d, i) => {
        const x = ((steps[i] - minStep) / range) * 100;
        const y = (1 - (values[i] - vMin) / vRange) * 100;
        return `${x},${y}`;
      })
      .join(" ");
  }, [data, steps, values, minStep, range]);

  // Extract clientX from either mouse or touch event
  const getClientX = useCallback((e) => {
    if (e.touches && e.touches.length > 0) return e.touches[0].clientX;
    if (e.changedTouches && e.changedTouches.length > 0) return e.changedTouches[0].clientX;
    return e.clientX;
  }, []);

  const getStepFromX = useCallback(
    (clientX) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect || !rect.width) return minStep;
      const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      return minStep + frac * range;
    },
    [minStep, range],
  );

  const snapToStep = useCallback(
    (val) => {
      if (!steps.length) return val;
      let best = steps[0];
      for (const s of steps) {
        if (Math.abs(s - val) < Math.abs(best - val)) best = s;
      }
      return best;
    },
    [steps],
  );

  const handlePointerMove = useCallback(
    (clientX) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const frac = (clientX - rect.left) / rect.width;
      const cur = domainAtDragStart.current || [minStep, maxStep];

      if (dragging.current === "left") {
        const raw = minStep + Math.max(0, Math.min(frac, 1)) * range;
        const s = snapToStep(Math.min(raw, cur[1] - range * 0.01));
        onDomainChange([s, cur[1]]);
      } else if (dragging.current === "right") {
        const raw = minStep + Math.max(0, Math.min(frac, 1)) * range;
        const s = snapToStep(Math.max(raw, cur[0] + range * 0.01));
        onDomainChange([cur[0], s]);
      } else if (dragging.current === "move") {
        const dx = (clientX - dragOrigin.current) / rect.width * range;
        const width = cur[1] - cur[0];
        let newStart = cur[0] + dx;
        let newEnd = cur[1] + dx;
        if (newStart < minStep) { newStart = minStep; newEnd = minStep + width; }
        if (newEnd > maxStep) { newEnd = maxStep; newStart = maxStep - width; }
        onDomainChange([snapToStep(newStart), snapToStep(newEnd)]);
      }
    },
    [minStep, maxStep, range, snapToStep, onDomainChange],
  );

  const stopDrag = useCallback(() => {
    dragging.current = null;
    dragOrigin.current = null;
    domainAtDragStart.current = null;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  useEffect(() => {
    function onMouseMove(e) {
      if (!dragging.current) return;
      e.preventDefault();
      handlePointerMove(e.clientX);
    }
    function onTouchMove(e) {
      if (!dragging.current) return;
      e.preventDefault();
      if (e.touches.length > 0) handlePointerMove(e.touches[0].clientX);
    }
    function onEnd() {
      stopDrag();
    }

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onEnd);
    window.addEventListener("touchmove", onTouchMove, { passive: false });
    window.addEventListener("touchend", onEnd);
    window.addEventListener("touchcancel", onEnd);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onEnd);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", onEnd);
      window.removeEventListener("touchcancel", onEnd);
    };
  }, [handlePointerMove, stopDrag]);

  function startDrag(type, e) {
    e.preventDefault();
    e.stopPropagation();
    dragging.current = type;
    dragOrigin.current = getClientX(e);
    domainAtDragStart.current = domain ? [...domain] : [minStep, maxStep];
    document.body.style.cursor = type === "move" ? "grabbing" : "ew-resize";
    document.body.style.userSelect = "none";
  }

  const hasZoom = domain !== null;
  const leftPct = `${selLeft * 100}%`;
  const widthPct = `${(selRight - selLeft) * 100}%`;

  return (
    <div
      ref={containerRef}
      className="relative mt-3 rounded-lg border border-white/10 bg-black/40 overflow-hidden select-none touch-none"
      style={{ height }}
    >
      {/* Sparkline */}
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        className="absolute inset-0 h-full w-full"
      >
        <polyline
          points={polyline}
          fill="none"
          stroke={color}
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
          opacity="0.4"
        />
      </svg>

      {/* Dim overlays for non-selected regions */}
      {hasZoom && (
        <>
          <div
            className="absolute inset-y-0 left-0 bg-black/50"
            style={{ width: leftPct }}
          />
          <div
            className="absolute inset-y-0 right-0 bg-black/50"
            style={{ width: `${(1 - selRight) * 100}%` }}
          />
        </>
      )}

      {/* Selection box — draggable center */}
      <div
        className="absolute inset-y-0 cursor-grab active:cursor-grabbing"
        style={{ left: leftPct, width: widthPct }}
        onMouseDown={(e) => startDrag("move", e)}
        onTouchStart={(e) => startDrag("move", e)}
      >
        <div className="absolute inset-0 border-x-2 border-white/30 bg-white/[0.04]" />
      </div>

      {/* Left handle — wider on mobile for touch */}
      {hasZoom && (
        <div
          className="absolute inset-y-0 w-5 sm:w-3 cursor-ew-resize group/handle flex items-center justify-center z-10"
          style={{ left: `calc(${leftPct} - 10px)` }}
          onMouseDown={(e) => startDrag("left", e)}
          onTouchStart={(e) => startDrag("left", e)}
        >
          <div className="h-5 w-1.5 sm:h-4 sm:w-1 rounded-full bg-white/60 group-hover/handle:bg-white transition-colors" />
        </div>
      )}

      {/* Right handle */}
      {hasZoom && (
        <div
          className="absolute inset-y-0 w-5 sm:w-3 cursor-ew-resize group/handle flex items-center justify-center z-10"
          style={{ left: `calc(${leftPct} + ${widthPct} - 10px)` }}
          onMouseDown={(e) => startDrag("right", e)}
          onTouchStart={(e) => startDrag("right", e)}
        >
          <div className="h-5 w-1.5 sm:h-4 sm:w-1 rounded-full bg-white/60 group-hover/handle:bg-white transition-colors" />
        </div>
      )}

      {/* Click/tap on empty area to create new zoom */}
      {!hasZoom && (
        <div
          className="absolute inset-0 cursor-crosshair"
          onMouseDown={(e) => {
            const step = getStepFromX(getClientX(e));
            const snap = snapToStep(step);
            const w = range * 0.15;
            onDomainChange([
              snapToStep(Math.max(minStep, snap - w / 2)),
              snapToStep(Math.min(maxStep, snap + w / 2)),
            ]);
          }}
          onTouchStart={(e) => {
            e.preventDefault();
            const step = getStepFromX(getClientX(e));
            const snap = snapToStep(step);
            const w = range * 0.15;
            onDomainChange([
              snapToStep(Math.max(minStep, snap - w / 2)),
              snapToStep(Math.min(maxStep, snap + w / 2)),
            ]);
          }}
        />
      )}
    </div>
  );
}

export default function Home() {
  const isSmallScreen = useMediaQuery("(max-width: 640px)");

  const [data, setData] = useState({
    status: "waiting",
    current: {},
    history: [],
    epochs: [],
  });

  const [lossDomain, setLossDomain] = useState(null);
  const [lrDomain, setLrDomain] = useState(null);
  const [isInitialized, setIsInitialized] = useState(false);

  // Use refs to debounce the API calls so we don't spam the server while dragging
  const saveTimeoutRef = useRef(null);

  function saveZoomToServer(payload) {
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }
    
    saveTimeoutRef.current = setTimeout(() => {
      fetch("/api/zoom", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }).catch(console.error);
    }, 500); // Wait 500ms after the last change before saving
  }

  // Persist zoom states to backend when they change
  function handleSetLossDomain(domain) {
    setLossDomain(domain);
    saveZoomToServer({ lossDomain: domain });
  }

  function handleSetLrDomain(domain) {
    setLrDomain(domain);
    saveZoomToServer({ lrDomain: domain });
  }

  useEffect(() => {
    let alive = true;
    let initialized = false;

    async function refresh() {
      try {
        const response = await fetch("/api/metrics", { cache: "no-store" });

        if (!response.ok) {
          throw new Error("metrics unavailable");
        }

        const payload = await response.json();

        if (alive) {
          if (!initialized) {
            initialized = true;
            // Batch all state updates into a single render
            setLossDomain(payload.lossDomain ?? null);
            setLrDomain(payload.lrDomain ?? null);
            setData(payload);
            setIsInitialized(true);
          } else {
            setData(payload);
          }
        }
      } catch {
        if (alive) {
          setData((current) => ({
            ...current,
            status: "offline",
          }));
        }
      }
    }

    refresh();

    const timer = setInterval(refresh, 3000);

    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const history = useMemo(() => {
    return Array.isArray(data.history) ? data.history : [];
  }, [data.history]);

  const epochs = useMemo(() => {
    return Array.isArray(data.epochs) ? data.epochs : [];
  }, [data.epochs]);

  const latestTrain = lastItem(history, (item) => item.type === "train");

  const latestMetric =
    data.current?.loss !== undefined ? data.current : latestTrain;

  const totalSteps = Number(data.total_steps || latestMetric.total_steps || 0);

  const currentStep = Number(
    data.current?.global_step ??
      latestMetric.global_step ??
      data.start_global_step ??
      0,
  );

  const progress =
    totalSteps > 0 ? Math.min(100, (currentStep / totalSteps) * 100) : 0;

  const status = data.status || "waiting";
  const isRunning = status === "running";

  const updatedAt = data.updated_at
    ? new Date(data.updated_at).toLocaleString("it-IT")
    : "in attesa dei primi dati";

  const chartHeight = isSmallScreen ? 260 : 360;
  const yAxisWidth = isSmallScreen ? 30 : 52;
  const lrAxisWidth = isSmallScreen ? 38 : 64;
  const chartMargin = isSmallScreen
    ? { left: -8, right: 4, top: 8, bottom: 0 }
    : { left: 0, right: 10, top: 10, bottom: 0 };

  const compactTick = (value) =>
    Number(value).toLocaleString("it-IT", {
      notation: "compact",
      maximumFractionDigits: 1,
    });

  const lossRowsRaw = useMemo(() => {
    const rows = new Map();

    for (const item of history) {
      if (!item.global_step) continue;

      const row = rows.get(item.global_step) || {
        step: Number(item.global_step),
      };

      if (item.type === "train") {
        row.trainLoss = item.loss;
      }

      if (item.type === "eval") {
        row.evalLoss = item.loss;
      }

      rows.set(item.global_step, row);
    }

    return Array.from(rows.values()).sort((a, b) => a.step - b.step);
  }, [history]);

  const lrRowsRaw = useMemo(() => {
    return history
      .filter((item) => item.global_step && item.learning_rate !== undefined)
      .map((item) => ({
        step: Number(item.global_step),
        learningRate: item.learning_rate,
      }))
      .sort((a, b) => a.step - b.step);
  }, [history]);

  // Gentle downsampling guarantees brush works well without losing zooming fidelity
  const lossRows = useMemo(() => gentleDownsample(lossRowsRaw), [lossRowsRaw]);
  const lrRows = useMemo(() => gentleDownsample(lrRowsRaw), [lrRowsRaw]);

  // Derived filtered data for dynamic dot logic
  const zoomedLossRows = useMemo(() => {
    if (!lossDomain) return lossRows;
    return lossRows.filter(r => r.step >= lossDomain[0] && r.step <= lossDomain[1]);
  }, [lossRows, lossDomain]);

  const zoomedLrRows = useMemo(() => {
    if (!lrDomain) return lrRows;
    return lrRows.filter(r => r.step >= lrDomain[0] && r.step <= lrDomain[1]);
  }, [lrRows, lrDomain]);



  // Dynamic dots based on currently visible points
  const showLossDots = zoomedLossRows.length < 80;
  const showLrDots = zoomedLrRows.length < 80;

  return (
    <main className="relative min-h-screen overflow-x-hidden bg-[#050505] text-zinc-100 selection:bg-cyan-500/30">
      {/* Dynamic ambient backgrounds */}
      <div className="pointer-events-none absolute -left-1/4 top-0 size-[900px] rounded-full bg-cyan-900/10 blur-[150px] animate-pulse duration-[8000ms]" />
      <div className="pointer-events-none absolute -right-1/4 top-1/3 size-[800px] rounded-full bg-violet-900/10 blur-[150px] animate-pulse duration-[12000ms]" />
      <div className="pointer-events-none absolute left-1/4 bottom-0 size-[600px] rounded-full bg-emerald-900/5 blur-[120px]" />

      <div className="relative z-10 border-b border-white/5 bg-black/30 backdrop-blur-2xl">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-4 px-3 py-4 sm:gap-6 sm:px-6 sm:py-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
          <div className="flex min-w-0 items-center gap-5">
            <div className="hidden size-14 shrink-0 items-center justify-center rounded-2xl border border-cyan-400/30 bg-gradient-to-br from-cyan-400/20 to-cyan-600/5 shadow-[0_0_20px_rgba(34,211,238,0.2)] backdrop-blur-md min-[420px]:flex">
              <Server className="size-7 text-cyan-300 drop-shadow-md" />
            </div>

            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-4">
                <h1 className="truncate text-2xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-white via-zinc-100 to-zinc-400 sm:text-4xl drop-shadow-sm">
                  Jarvis Dashboard
                </h1>

                <Badge
                  variant={isRunning ? "default" : "secondary"}
                  className={`rounded-full px-3 py-1 text-xs font-bold uppercase tracking-wider border shadow-sm ${isRunning ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30 shadow-[0_0_10px_rgba(34,211,238,0.2)]' : 'bg-white/5 text-zinc-400 border-white/10'}`}
                >
                  <Radio className={`mr-1.5 size-3.5 ${isRunning ? 'animate-pulse' : ''}`} />
                  {status}
                </Badge>
              </div>

              <p className="mt-1.5 flex min-w-0 items-center gap-2 truncate text-[13px] font-medium text-zinc-400 sm:text-sm">
                <Clock3 className="size-4 shrink-0 text-zinc-500" />
                <span className="truncate">Last updated: <span className="text-zinc-300">{updatedAt}</span></span>
              </p>
            </div>
          </div>

          <div className="w-full rounded-2xl border border-white/10 bg-black/40 p-5 shadow-2xl lg:w-96 backdrop-blur-xl">
            <div className="mb-4 flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-zinc-400">Progress</span>
              <span className="text-lg font-black tabular-nums text-white drop-shadow-sm">
                {formatNumber(progress, 1)}%
              </span>
            </div>

            <Progress value={progress} className="h-3 rounded-full bg-white/5 overflow-hidden shadow-inner [&>div]:bg-gradient-to-r [&>div]:from-cyan-400 [&>div]:via-blue-500 [&>div]:to-violet-500" />

            <div className="mt-4 text-[13px] font-medium tabular-nums text-zinc-400 flex justify-between">
              <span>Step <span className="text-white">{currentStep || 0}</span></span>
              <span>Total <span className="text-white">{totalSteps || "-"}</span></span>
            </div>
          </div>
        </div>
      </div>

      <div className="relative z-10 mx-auto grid w-full max-w-7xl gap-5 sm:gap-6 px-3 sm:px-6 py-6 sm:py-8 lg:px-8">
        <section className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4 animate-in fade-in slide-in-from-bottom-8 duration-700 fill-mode-both">
          <MetricTile
            icon={Activity}
            label="Current Step"
            value={`${currentStep || 0}`}
            detail={`Epoch ${latestMetric.epoch || "-"}`}
          />

          <MetricTile
            icon={Gauge}
            label="Training Loss"
            value={formatNumber(latestMetric.loss)}
            detail="Latest data point"
            tone="amber"
          />

          <MetricTile
            icon={LineChartIcon}
            label="Perplexity"
            value={formatNumber(latestMetric.perplexity, 2)}
            detail="Derived from loss"
            tone="violet"
          />

          <MetricTile
            icon={Zap}
            label="Learning Rate"
            value={formatCompact(latestMetric.learning_rate)}
            detail="Scheduler state"
            tone="emerald"
          />
        </section>

        <section className="grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_minmax(0,0.55fr)] animate-in fade-in slide-in-from-bottom-12 duration-1000 fill-mode-both delay-100">
          <ChartCard
            title="Loss Trajectory"
            description="Training and evaluation loss across global steps"
            empty={!isInitialized || !lossRows.length ? "Waiting for loss data points..." : null}
          >
            <div className="mb-4 flex items-center justify-end">
              <button
                type="button"
                onClick={() => handleSetLossDomain(null)}
                className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-bold uppercase tracking-wider text-zinc-300 transition-all hover:bg-white/15 hover:text-white hover:scale-105 active:scale-95 shadow-sm"
                aria-label="Reset zoom loss"
              >
                Reset Zoom
              </button>
            </div>

            <div className="overflow-hidden min-w-0" style={{ width: '100%', height: chartHeight }}>
              <ChartContainer config={chartConfig} className="h-full w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                  data={zoomedLossRows}
                  margin={chartMargin}
                >
                  <defs>
                    <filter id="glowLoss">
                      <feGaussianBlur stdDeviation="3.5" result="coloredBlur"/>
                      <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                      </feMerge>
                    </filter>
                  </defs>
                  <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />

                  <XAxis
                    dataKey="step"
                    tickLine={false}
                    axisLine={false}
                    type="number"
                    domain={["dataMin", "dataMax"]}
                    tickFormatter={compactTick}
                    tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12, fontWeight: 500 }}
                    dy={12}
                  />

                  <YAxis 
                    tickLine={false} 
                    axisLine={false} 
                    width={yAxisWidth} 
                    tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12, fontWeight: 500 }}
                    dx={-12}
                  />

                  <ChartTooltip 
                    cursor={{ stroke: 'rgba(255,255,255,0.15)', strokeWidth: 2, strokeDasharray: '4 4' }}
                    content={<ChartTooltipContent className="bg-black/90 border border-white/20 backdrop-blur-xl shadow-2xl rounded-xl p-3 font-medium text-sm" />} 
                  />

                  <Line
                    type="monotone"
                    dataKey="trainLoss"
                    stroke="var(--color-trainLoss)"
                    strokeWidth={1.5}
                    dot={false}
                    activeDot={{ r: 6, stroke: "rgba(255,255,255,0.8)", strokeWidth: 2, fill: "var(--color-trainLoss)" }}
                    connectNulls
                    isAnimationActive={false}
                    filter="url(#glowLoss)"
                  />

                  <Line
                    type="monotone"
                    dataKey="evalLoss"
                    stroke="var(--color-evalLoss)"
                    strokeWidth={2.5}
                    dot={showLossDots ? { r: 4, fill: "var(--color-evalLoss)", strokeWidth: 0 } : false}
                    activeDot={{ r: 7, stroke: "rgba(255,255,255,0.8)", strokeWidth: 2, fill: "var(--color-evalLoss)" }}
                    connectNulls
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
              </ChartContainer>
            </div>

            <ZoomMinimap
              data={lossRows}
              dataKey="trainLoss"
              domain={lossDomain}
              onDomainChange={handleSetLossDomain}
              color="var(--color-trainLoss)"
            />
          </ChartCard>

          <ChartCard
            title="Learning Rate Scheduler"
            description="LR decay curve over global steps"
            empty={!isInitialized || !lrRows.length ? "Waiting for LR data points..." : null}
          >
            <div className="mb-4 flex items-center justify-end">
              <button
                type="button"
                onClick={() => handleSetLrDomain(null)}
                className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-bold uppercase tracking-wider text-zinc-300 transition-all hover:bg-white/15 hover:text-white hover:scale-105 active:scale-95 shadow-sm"
                aria-label="Reset zoom learning rate"
              >
                Reset Zoom
              </button>
            </div>

            <div className="overflow-hidden min-w-0" style={{ width: '100%', height: chartHeight }}>
              <ChartContainer config={chartConfig} className="h-full w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={zoomedLrRows}
                    margin={chartMargin}
                  >
                    <defs>
                      <filter id="glowLr">
                        <feGaussianBlur stdDeviation="3.5" result="coloredBlur"/>
                        <feMerge>
                          <feMergeNode in="coloredBlur"/>
                          <feMergeNode in="SourceGraphic"/>
                        </feMerge>
                      </filter>
                    </defs>
                    <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />

                    <XAxis
                      dataKey="step"
                      tickLine={false}
                      axisLine={false}
                      type="number"
                      domain={["dataMin", "dataMax"]}
                      tickFormatter={compactTick}
                      tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12, fontWeight: 500 }}
                      dy={12}
                    />

                    <YAxis
                      tickLine={false}
                      axisLine={false}
                      width={lrAxisWidth}
                      tickFormatter={formatCompact}
                      tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12, fontWeight: 500 }}
                      dx={-12}
                    />

                    <ChartTooltip 
                      cursor={{ stroke: 'rgba(255,255,255,0.15)', strokeWidth: 2, strokeDasharray: '4 4' }}
                      content={<ChartTooltipContent className="bg-black/90 border border-white/20 backdrop-blur-xl shadow-2xl rounded-xl p-3 font-medium text-sm" formatter={(value) => formatCompact(value)} />} 
                    />

                    <Line
                      type="monotone"
                      dataKey="learningRate"
                      stroke="var(--color-learningRate)"
                      strokeWidth={2.5}
                      dot={showLrDots ? { r: 3, fill: "var(--color-learningRate)", strokeWidth: 0 } : false}
                      activeDot={{ r: 6, stroke: "rgba(255,255,255,0.8)", strokeWidth: 2, fill: "var(--color-learningRate)" }}
                      isAnimationActive={false}
                      filter="url(#glowLr)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </ChartContainer>
            </div>

            <ZoomMinimap
              data={lrRows}
              dataKey="learningRate"
              domain={lrDomain}
              onDomainChange={handleSetLrDomain}
              color="var(--color-learningRate)"
            />
          </ChartCard>
        </section>

        <Accordion
          defaultValue={["epochs"]}
          type="multiple"
          className="rounded-2xl border border-white/10 bg-black/40 px-4 sm:px-6 shadow-2xl backdrop-blur-xl animate-in fade-in slide-in-from-bottom-16 duration-1000 fill-mode-both delay-200 mt-6"
        >
          <AccordionItem value="epochs" className="border-white/10 border-b">
            <AccordionTrigger className="py-5 text-base font-bold tracking-tight text-white hover:no-underline">
              Recorded Epochs
            </AccordionTrigger>

            <AccordionContent>
              <div className="overflow-auto pb-4">
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead className="text-zinc-400 font-bold uppercase tracking-wider text-xs">Epoch</TableHead>
                      <TableHead className="text-zinc-400 font-bold uppercase tracking-wider text-xs">Start Step</TableHead>
                      <TableHead className="text-zinc-400 font-bold uppercase tracking-wider text-xs">End Step</TableHead>
                      <TableHead className="text-zinc-400 font-bold uppercase tracking-wider text-xs">Avg Loss</TableHead>
                      <TableHead className="text-zinc-400 font-bold uppercase tracking-wider text-xs">Perplexity</TableHead>
                      <TableHead className="text-zinc-400 font-bold uppercase tracking-wider text-xs">Duration</TableHead>
                      <TableHead className="text-zinc-400 font-bold uppercase tracking-wider text-xs">Status</TableHead>
                    </TableRow>
                  </TableHeader>

                  <TableBody>
                    {epochs.length ? (
                      epochs.map((epoch) => (
                        <TableRow
                          key={`${epoch.epoch}-${epoch.end_global_step}`}
                          className="border-white/5 hover:bg-white/[0.04] transition-colors"
                        >
                          <TableCell className="font-bold text-white text-base">{epoch.epoch}</TableCell>

                          <TableCell className="tabular-nums text-zinc-300 font-medium">
                            {epoch.start_global_step}
                          </TableCell>

                          <TableCell className="tabular-nums text-zinc-300 font-medium">
                            {epoch.end_global_step}
                          </TableCell>

                          <TableCell className="tabular-nums text-zinc-300 font-medium">
                            {formatNumber(epoch.train_loss)}
                          </TableCell>

                          <TableCell className="tabular-nums text-zinc-300 font-medium">
                            {formatNumber(epoch.train_perplexity, 2)}
                          </TableCell>

                          <TableCell className="text-zinc-300 font-medium">
                            {formatDuration(epoch.duration_seconds)}
                          </TableCell>

                          <TableCell>
                            <Badge
                              variant={epoch.complete ? "default" : "secondary"}
                              className={epoch.complete ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30 font-bold shadow-sm' : 'bg-white/10 text-zinc-300 border-white/20 font-bold'}
                            >
                              {epoch.complete ? "completa" : "parziale"}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow className="border-white/5 hover:bg-transparent">
                        <TableCell
                          colSpan={7}
                          className="h-32 text-center text-zinc-500 font-medium"
                        >
                          Nessuna epoca registrata.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </div>
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="details" className="border-0">
            <AccordionTrigger className="py-5 text-base font-bold tracking-tight text-white hover:no-underline">
              Session Details
            </AccordionTrigger>

            <AccordionContent>
              <div className="grid gap-6 pb-6 sm:grid-cols-3">
                <div className="rounded-xl bg-black/40 p-5 border border-white/10 shadow-inner">
                  <p className="text-xs font-bold uppercase tracking-wider text-zinc-500">Collected Metrics</p>
                  <p className="mt-2.5 text-2xl font-black tabular-nums text-white">
                    {history.length}
                  </p>
                </div>

                <div className="rounded-xl bg-black/40 p-5 border border-white/10 shadow-inner">
                  <p className="text-xs font-bold uppercase tracking-wider text-zinc-500">Initial Step</p>
                  <p className="mt-2.5 text-2xl font-black tabular-nums text-white">
                    {data.start_global_step ?? "-"}
                  </p>
                </div>

                <div className="rounded-xl bg-black/40 p-5 border border-white/10 shadow-inner">
                  <p className="text-xs font-bold uppercase tracking-wider text-zinc-500">Created At</p>
                  <p className="mt-3 text-[15px] font-bold text-zinc-200">
                    {data.created_at
                      ? new Date(data.created_at).toLocaleString("it-IT")
                      : "-"}
                  </p>
                </div>
              </div>

              <div className="rounded-xl bg-[#0a0a0a] border border-white/10 shadow-inner overflow-hidden">
                <div className="bg-white/[0.03] border-b border-white/10 px-4 py-2.5">
                  <p className="text-xs font-bold uppercase tracking-wider text-zinc-500">Configuration</p>
                </div>
                <pre className="p-5 max-h-72 overflow-auto text-[13px] text-zinc-300 font-mono leading-relaxed">
                  {JSON.stringify(data.config || {}, null, 2)}
                </pre>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </main>
  );
}
