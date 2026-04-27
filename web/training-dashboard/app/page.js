"use client";

import { useEffect, useMemo, useState, useRef, useCallback } from "react";
import Link from "next/link";
import {
  Activity,
  Gauge,
  LineChart as LineChartIcon,
  Server,
  Zap,
  RefreshCcw,
  LayoutDashboard,
  Cpu,
  Database,
  Terminal,
  ShieldCheck,
  Globe,
  Clock,
  ChevronRight,
  Monitor,
  Box,
  HardDrive,
  Waves,
  RotateCcw,
  Activity as PulseIcon,
} from "lucide-react";
import { CartesianGrid, Area, AreaChart, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";

import { Badge } from "@/components/ui/badge";

/* --- Hooks --- */

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

function useClientClock() {
  const [time, setTime] = useState(null);

  useEffect(() => {
    const update = () => setTime(new Date().toLocaleTimeString());
    const initial = setTimeout(update, 0);
    const timer = setInterval(update, 1000);

    return () => {
      clearTimeout(initial);
      clearInterval(timer);
    };
  }, []);

  return time || "--:--:--";
}

function numericOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function pushSeries(series, value, size = 40) {
  return [...series.slice(-(size - 1)), numericOrNull(value)];
}

function emptySeries(size = 40) {
  return Array.from({ length: size }, () => null);
}

function emptyTelemetrySeries() {
  return {
    cpu: emptySeries(),
    gpu: emptySeries(),
    vram: emptySeries(),
    ram: emptySeries(),
    disk: emptySeries(),
    power: emptySeries(),
  };
}

function appendSystemTelemetry(previousSeries, payload) {
  const base = previousSeries || emptyTelemetrySeries();
  const system = payload.system || {};
  const history = Array.isArray(payload.history) ? payload.history : [];
  const latestMetric = payload.current?.loss !== undefined ? payload.current : (history[history.length - 1] || {});

  return {
    cpu: pushSeries(base.cpu || emptySeries(), system.cpu_percent),
    gpu: pushSeries(base.gpu || emptySeries(), system.gpu_util_percent),
    vram: pushSeries(base.vram || emptySeries(), system.gpu_memory_used_gb ?? latestMetric.vram_reserved_gb ?? latestMetric.vram_allocated_gb),
    ram: pushSeries(base.ram || emptySeries(), system.ram_percent),
    disk: pushSeries(base.disk || emptySeries(), system.disk_percent),
    power: pushSeries(base.power || emptySeries(), system.gpu_power_w),
  };
}

/* --- Formatting --- */

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatCompact(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toExponential(2);
}

function formatFixed(value, digits = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return number.toFixed(digits);
}

function formatGb(value, digits = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${number.toFixed(digits)}GB`;
}

function formatUnit(value, unit, digits = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${number.toFixed(digits)}${unit}`;
}

function metricLabel(dataKey) {
  const labels = {
    trainLoss: "Train loss",
    evalLoss: "Eval loss",
    learningRate: "Learning rate",
  };

  return labels[dataKey] || dataKey;
}

function formatMetricTooltipValue(dataKey, value) {
  if (dataKey === "learningRate") {
    return formatCompact(value);
  }

  return formatNumber(value);
}

/* --- Components --- */

function ZoomMinimap({ data, dataKey, domain, onDomainChange, color }) {
  const containerRef = useRef(null);
  const dragging = useRef(null);
  const steps = useMemo(() => data.map((d) => d.step), [data]);
  const minStep = steps[0] ?? 0;
  const maxStep = steps[steps.length - 1] ?? 1;
  const range = maxStep - minStep || 1;
  const selLeft = domain ? (domain[0] - minStep) / range : 0;
  const selRight = domain ? (domain[1] - minStep) / range : 1;

  const startDrag = (type, e) => {
    e.preventDefault();
    dragging.current = type;
    document.body.style.cursor = "ew-resize";
  };

  useEffect(() => {
    const handleMove = (e) => {
      if (!dragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
      const frac = Math.max(0, Math.min(1, x / rect.width));
      const s = minStep + frac * range;
      if (dragging.current === "left") onDomainChange([Math.min(s, (domain?.[1] || maxStep) - range * 0.01), domain?.[1] || maxStep]);
      if (dragging.current === "right") onDomainChange([domain?.[0] || minStep, Math.max(s, (domain?.[0] || minStep) + range * 0.01)]);
    };
    const stop = () => { dragging.current = null; document.body.style.cursor = ""; };
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", stop);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", stop);
    };
  }, [domain, minStep, range, maxStep, onDomainChange]);

  return (
    <div ref={containerRef} className="relative mt-10 h-3 bg-white/[0.02] rounded-full border border-white/5 overflow-visible">
      {domain && (
        <>
          <div className="absolute inset-y-0 bg-white/5 rounded-full border-x border-white/10" style={{ left: `${selLeft * 100}%`, right: `${(1 - selRight) * 100}%` }} />
          <div className="absolute top-1/2 -translate-y-1/2 size-7 bg-white rounded-xl cursor-ew-resize -translate-x-1/2 shadow-xl border-4 border-black" style={{ left: `${selLeft * 100}%` }} onMouseDown={(e) => startDrag("left", e)} />
          <div className="absolute top-1/2 -translate-y-1/2 size-7 bg-white rounded-xl cursor-ew-resize -translate-x-1/2 shadow-xl border-4 border-black" style={{ left: `${selRight * 100}%` }} onMouseDown={(e) => startDrag("right", e)} />
        </>
      )}
      {!domain && (
        <div className="absolute inset-0 cursor-pointer flex items-center justify-center group/init" onClick={() => onDomainChange([minStep, maxStep])}>
           <span className="text-[8px] font-black uppercase tracking-[0.4em] text-white/10 group-hover/init:text-white/40 transition-colors">Initialize Zoom Controller</span>
        </div>
      )}
    </div>
  );
}

function StatusBar({ vram, vramTotal, status }) {
  const clock = useClientClock();
  const vramText = Number.isFinite(Number(vram))
    ? `${Number(vram).toFixed(1)}GB${Number.isFinite(Number(vramTotal)) ? ` / ${Number(vramTotal).toFixed(1)}GB` : ""}`
    : "-";

  return (
    <div className="w-full bg-white/[0.02] border-b border-white/[0.05] py-2 px-6 sm:px-10 flex items-center justify-between text-[9px] font-black uppercase tracking-[0.2em] text-white/30 backdrop-blur-3xl">
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2.5">
          <div className="size-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_10px_#10b981]" />
          <span className="text-white/50">Core Sync: {status || "waiting"}</span>
        </div>
        <div className="hidden md:flex items-center gap-2">
          <Globe size={10} className="text-white/10" />
          <span>Jarvis-Protocol-01</span>
        </div>
      </div>
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2 font-mono">
          <Monitor size={10} className="text-white/10" />
          <span className="text-white/40">VRAM: {vramText}</span>
        </div>
        <div className="flex items-center gap-2 font-mono">
          <Clock size={10} className="text-white/10" />
          <span className="text-white/40">{clock}</span>
        </div>
      </div>
    </div>
  );
}

function PageHeader({ eyebrow, title, badge, actions }) {
  return (
    <div className="mb-12 admin-fade-up sticky top-0 z-20 bg-background/20 backdrop-blur-3xl py-6 border-b border-white/[0.03] -mx-6 px-6 sm:-mx-10 sm:px-10 group/header">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-8 w-full">
        <div className="flex items-center gap-6">
           <div className="size-12 rounded-2xl bg-white flex items-center justify-center text-black shadow-xl">
              <Cpu size={24} />
           </div>
           <div>
            <p className="text-[9px] text-white/15 font-black uppercase tracking-[0.4em] mb-1">{eyebrow}</p>
            <div className="flex items-center gap-4">
              <h1 className="text-4xl font-black tracking-tighter text-white text-glow leading-none">{title}</h1>
              {badge}
            </div>
           </div>
        </div>
        {actions && <div className="flex items-center gap-4">{actions}</div>}
      </div>
    </div>
  );
}

function TelemetryCard({ label, value, unit, data, color = "#FFFFFF", icon: Icon, domain = [0, 100] }) {
  const chartData = (Array.isArray(data) ? data : []).map(v => ({ v: numericOrNull(v) }));

  return (
    <div className="glass-card rounded-[1.5rem] p-5 border border-white/[0.04] relative overflow-hidden group/tel">
      <div className="relative z-10 flex flex-col justify-between h-full">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2.5">
             <div className="p-1.5 rounded-lg bg-white/[0.02] text-white/20 group-hover/tel:text-white transition-colors">
                <Icon size={14} />
             </div>
             <span className="text-[9px] font-black uppercase tracking-[0.15em] text-white/20">{label}</span>
          </div>
          <span className="text-xl font-black tabular-nums tracking-tighter text-white/80">{value}<span className="text-[9px] text-white/20 ml-1 uppercase">{unit}</span></span>
        </div>
        <div className="h-10 w-full -mb-1">
           <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                 <YAxis domain={domain} hide />
                 <Area type="monotone" dataKey="v" stroke={color} strokeWidth={1.5} fill={color} fillOpacity={0.05} isAnimationActive={false} />
              </AreaChart>
           </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, icon: Icon, caption, trend, data, dataKey, smoothing, onSmoothingChange }) {
  return (
    <div className="surface-elevated rounded-[2rem] p-6 transition-all hover:translate-y-[-4px] admin-fade-up group relative overflow-hidden">
      <div className="relative z-10">
        <div className="flex items-start justify-between mb-4">
          <div className="p-4 rounded-2xl bg-white/[0.03] border border-white/[0.06] group-hover:bg-white group-hover:text-black transition-all duration-500">
            <Icon size={22} />
          </div>
          {onSmoothingChange && (
             <div className="flex items-center gap-2.5 bg-black/40 px-3 py-1.5 rounded-xl border border-white/5 opacity-0 group-hover:opacity-100 transition-all backdrop-blur-xl">
               <input type="range" min="1" max="50" value={smoothing} onChange={(e) => onSmoothingChange(parseInt(e.target.value))} className="w-12 accent-white h-1 bg-white/5 rounded-full appearance-none cursor-pointer" />
               <span className="text-[9px] font-black tabular-nums text-white/40">{smoothing}</span>
             </div>
          )}
        </div>
        <div className="space-y-1">
          <p className="text-[10px] font-black uppercase tracking-[0.3em] text-white/15">{label}</p>
          <h3 className="text-4xl font-black tabular-nums text-white tracking-tighter text-glow">{value}</h3>
        </div>
      </div>
      {data && data.length > 1 && (
        <div className="absolute inset-x-0 bottom-0 h-24 opacity-5 pointer-events-none group-hover:opacity-15 transition-opacity duration-1000">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data}>
              <Area type="monotone" dataKey={dataKey} stroke="#FFFFFF" strokeWidth={2} fill="#FFFFFF" fillOpacity={0.1} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function SurfaceCard({ title, description, children, badge, className = "" }) {
  return (
    <div className={`surface-elevated rounded-[2.5rem] admin-fade-up group/card ${className}`}>
      <div className="px-10 py-7 border-b border-white/[0.04] flex items-center justify-between relative z-10">
        <div className="flex items-center gap-4">
          <div className="size-3 rounded-full bg-white shadow-[0_0_15px_rgba(255,255,255,0.6)] animate-pulse" />
          <div>
            <h2 className="text-xs font-black uppercase tracking-[0.4em] text-white/80">{title}</h2>
            {description && <p className="text-[9px] font-bold text-white/10 uppercase tracking-widest mt-1">{description}</p>}
          </div>
        </div>
        {badge}
      </div>
      <div className={`p-10 relative z-10 ${className.includes('flex-col') ? 'flex-1 flex flex-col' : ''}`}>
        {children}
      </div>
    </div>
  );
}

function SummaryRow({ label, value, colorClass = "text-white", icon: Icon }) {
  return (
    <div className="flex items-center justify-between py-5 border-b border-white/[0.02] last:border-0 group/row">
      <div className="flex items-center gap-5">
        <div className="size-10 rounded-xl bg-white/[0.02] border border-white/[0.05] flex items-center justify-center group-hover/row:bg-white group-hover/row:text-black transition-all duration-500 shadow-xl">
          {Icon && <Icon size={16} />}
        </div>
        <span className="text-[10px] font-black text-white/20 uppercase tracking-[0.3em]">{label}</span>
      </div>
      <span className={`text-lg font-black tabular-nums tracking-tighter ${colorClass}`}>{value}</span>
    </div>
  );
}

function ChartTooltip({ active, label, payload }) {
  const rows = (payload || []).filter((item) => Number.isFinite(Number(item.value)));

  if (!active || rows.length === 0) {
    return null;
  }

  return (
    <div className="rounded-2xl border border-white/10 bg-black/95 px-5 py-4 shadow-2xl backdrop-blur-xl">
      <div className="mb-3 flex items-center justify-between gap-8 border-b border-white/5 pb-3">
        <span className="text-[9px] font-black uppercase tracking-[0.3em] text-white/20">Step</span>
        <span className="font-mono text-xs font-black tabular-nums text-white">{label ?? rows[0]?.payload?.step ?? "-"}</span>
      </div>
      <div className="space-y-2">
        {rows.map((item) => (
          <div key={item.dataKey} className="flex items-center justify-between gap-8">
            <span className="text-[10px] font-black uppercase tracking-[0.2em] text-white/30">{metricLabel(item.dataKey)}</span>
            <span className="font-mono text-sm font-black tabular-nums text-white">{formatMetricTooltipValue(item.dataKey, item.value)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* --- Main View --- */

export default function Home() {
  const [data, setData] = useState({ status: "IDLE", current: {}, history: [], epochs: [], systemSeries: emptyTelemetrySeries() });
  const [lossDomain, setLossDomain] = useState(null);
  const [lrDomain, setLrDomain] = useState(null);
  const [smoothingTrain, setSmoothingTrain] = useState(15);
  const [smoothingEval, setSmoothingEval] = useState(15);
  const [smoothingPpl, setSmoothingPpl] = useState(15);
  const [smoothingLr, setSmoothingLr] = useState(1);
  const [isInitialized, setIsInitialized] = useState(false);
  const isRunning = data.status === "running";
  const telemetry = data.systemSeries || emptyTelemetrySeries();

  const refreshData = useCallback(async () => {
    try {
      const response = await fetch("/api/metrics", { cache: "no-store" });
      if (!response.ok) throw new Error();
      const payload = await response.json();
      setData(previous => ({
        ...payload,
        systemSeries: appendSystemTelemetry(previous.systemSeries, payload),
      }));
      if (!isInitialized) {
        setLossDomain(payload.lossDomain ?? null);
        setLrDomain(payload.lrDomain ?? null);
        setIsInitialized(true);
      }
    } catch {
      setData(c => ({ ...c, status: "OFFLINE" }));
    }
  }, [isInitialized]);

  useEffect(() => {
    const initial = setTimeout(refreshData, 0);
    const timer = setInterval(refreshData, 3000);
    return () => {
      clearTimeout(initial);
      clearInterval(timer);
    };
  }, [refreshData]);

  const history = useMemo(() => Array.isArray(data.history) ? data.history : [], [data.history]);
  const latestMetric = data.current?.loss !== undefined ? data.current : (history[history.length - 1] || {});
  const currentStep = Number(data.current?.global_step ?? latestMetric.global_step ?? data.start_global_step ?? 0);
  const totalSteps = Number(data.total_steps || latestMetric.total_steps || 0);
  const progress = totalSteps > 0 ? Math.min(100, (currentStep / totalSteps) * 100) : 0;

  const system = data.system || {};
  const currentVram = numericOrNull(system.gpu_memory_used_gb ?? latestMetric.vram_reserved_gb ?? latestMetric.vram_allocated_gb);
  const totalVram = numericOrNull(system.gpu_memory_total_gb);
  const vramDomain = [0, Math.max(1, Math.ceil(totalVram || currentVram || 1))];

  const lossRows = useMemo(() => {
    const rows = new Map();
    history.forEach(item => {
      if (!item.global_step) return;
      const step = Number(item.global_step);
      const row = rows.get(step) || { step };
      if (item.type === "train") { row.trainLoss = item.loss; row.perplexity = item.perplexity; }
      if (item.type === "eval") { row.evalLoss = item.loss; }
      rows.set(step, row);
    });
    const arr = Array.from(rows.values()).sort((a, b) => a.step - b.step);
    const smoothedArr = arr.map((item, index) => {
      const getAvg = (key, window) => {
        const start = Math.max(0, index - window);
        const subset = arr.slice(start, index + 1);
        const values = subset.map(s => s[key]).filter(v => v !== undefined);
        return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : undefined;
      };
      return { ...item, trainLoss: getAvg('trainLoss', smoothingTrain), evalLoss: getAvg('evalLoss', smoothingEval), perplexity: getAvg('perplexity', smoothingPpl), rawTrainLoss: item.trainLoss };
    });
    return smoothedArr;
  }, [history, smoothingTrain, smoothingEval, smoothingPpl]);

  const lrRows = useMemo(() => {
    const arr = history.filter(i => i.global_step && i.learning_rate !== undefined).map(i => ({ step: Number(i.global_step), learningRate: i.learning_rate })).sort((a, b) => a.step - b.step);
    const smoothedArr = arr.map((item, index) => {
      const start = Math.max(0, index - smoothingLr);
      const values = arr.slice(start, index + 1).map(s => s.learningRate);
      return { ...item, learningRate: values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : item.learningRate };
    });
    return smoothedArr;
  }, [history, smoothingLr]);

  const zoomedLossRows = useMemo(() => lossDomain ? lossRows.filter(r => r.step >= lossDomain[0] && r.step <= lossDomain[1]) : lossRows, [lossRows, lossDomain]);
  const zoomedLrRows = useMemo(() => lrDomain ? lrRows.filter(r => r.step >= lrDomain[0] && r.step <= lrDomain[1]) : lrRows, [lrRows, lrDomain]);

  return (
    <main className="min-h-screen w-full pb-32 font-sans overflow-x-hidden">
      <StatusBar vram={currentVram} vramTotal={totalVram} status={data.status} />
      
      <div className="px-6 sm:px-10">
        <PageHeader
          eyebrow="Neural Protocol 4.2.0"
          title="Jarvis Core"
          badge={
            <Badge className={`rounded-full px-4 py-1.5 text-[10px] font-black uppercase tracking-[0.3em] transition-all duration-700 ${isRunning ? 'bg-white text-black animate-pulse' : 'bg-white/5 text-white/30 border-white/10'}`}>
              {data.status || "IDLE"}
            </Badge>
          }
          actions={
            <div className="flex gap-6">
               <button onClick={refreshData} className="size-12 rounded-xl bg-white/[0.02] border border-white/10 flex items-center justify-center transition-all hover:bg-white hover:text-black group">
                  <RefreshCcw size={18} className={`${isRunning ? "animate-spin" : ""} group-hover:rotate-180 transition-transform duration-700`} />
               </button>
               <Link href="/test-model" className="px-8 h-12 rounded-xl bg-white text-black text-[12px] font-black uppercase tracking-[0.24em] hover:bg-white/90 transition-all shadow-xl flex items-center gap-3">
                  <Terminal size={14} /> Test Model <ChevronRight size={14} />
               </Link>
            </div>
          }
        />

        <div className="mb-14 admin-fade-up">
           <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-5 gap-6">
              <TelemetryCard label="CPU Load" value={formatFixed(system.cpu_percent, 1)} unit="%" data={telemetry.cpu} icon={Cpu} color="#FFFFFF" />
              <TelemetryCard label="GPU Compute" value={formatFixed(system.gpu_util_percent, 1)} unit="%" data={telemetry.gpu} icon={Monitor} color="#10b981" />
              <TelemetryCard label="VRAM Used" value={formatFixed(currentVram, 2)} unit="GB" data={telemetry.vram} icon={Database} color="#3b82f6" domain={vramDomain} />
              <TelemetryCard label="RAM Load" value={formatFixed(system.ram_percent, 1)} unit="%" data={telemetry.ram} icon={Globe} color="#f59e0b" />
              
              <div className="glass-card rounded-[1.5rem] p-5 border border-white/[0.04] relative overflow-hidden group/specs">
                 <div className="flex flex-col justify-between h-full relative z-10">
                    <div className="flex items-center gap-2.5 mb-4">
                       <HardDrive size={14} className="text-white/20" />
                       <span className="text-[9px] font-black uppercase tracking-[0.15em] text-white/20">System Specs</span>
                    </div>
                    <div className="space-y-3">
                       <div className="flex justify-between items-center gap-3"><span className="text-[9px] font-bold text-white/10 uppercase">GPU</span><span className="text-[11px] font-black text-white/60 truncate">{system.gpu_name || data.hardware?.gpu || "GPU non rilevata"}</span></div>
                       <div className="flex justify-between items-center gap-3"><span className="text-[9px] font-bold text-white/10 uppercase">CPU</span><span className="text-[11px] font-black text-white/60 truncate">{data.hardware?.cpu || "Detecting..."}</span></div>
                       <div className="flex justify-between items-center"><span className="text-[9px] font-bold text-white/10 uppercase">RAM</span><span className="text-[11px] font-black text-white/60">{formatGb(system.ram_used_gb)} / {formatGb(system.ram_total_gb)}</span></div>
                       <div className="flex justify-between items-center"><span className="text-[9px] font-bold text-white/10 uppercase">Disk</span><span className="text-[11px] font-black text-white/60">{formatGb(system.disk_free_gb)} free</span></div>
                    </div>
                 </div>
              </div>
           </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-5 gap-6 mb-14 items-stretch">
          <MetricCard icon={LayoutDashboard} label="Process Step" value={String(currentStep)} caption={`Target: ${totalSteps || "INF"}`} data={lossRows.slice(-60)} dataKey="step" />
          <MetricCard icon={Gauge} label="Loss Delta" value={formatNumber(latestMetric.loss)} caption="Neural alignment" data={lossRows.slice(-60)} dataKey="trainLoss" smoothing={smoothingTrain} onSmoothingChange={setSmoothingTrain} />
          <MetricCard icon={LineChartIcon} label="Optimization" value={formatNumber(latestMetric.perplexity, 2)} caption="PPL metric" data={lossRows.slice(-60)} dataKey="perplexity" smoothing={smoothingPpl} onSmoothingChange={setSmoothingPpl} />
          <MetricCard icon={Zap} label="Flux Rate" value={formatCompact(latestMetric.learning_rate)} caption="Learning modulation" data={lrRows.slice(-60)} dataKey="learningRate" smoothing={smoothingLr} onSmoothingChange={setSmoothingLr} />

          <SurfaceCard title="System Load" className="h-full">
            <div className="flex flex-col h-full justify-between gap-6">
              <div className="space-y-4">
                <span className="text-5xl font-black tabular-nums text-white tracking-tighter text-glow">{progress.toFixed(1)}%</span>
                <div className="h-3 w-full bg-white/[0.02] rounded-full overflow-hidden border border-white/5 p-1 shadow-inner relative">
                  <div className="h-full bg-white rounded-full transition-all duration-1000 ease-out relative" style={{ width: `${progress}%` }}>
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-pulse" />
                  </div>
                </div>
              </div>
              <div className="space-y-0">
                 <SummaryRow icon={Globe} label="VRAM" value={`${formatGb(currentVram)} / ${formatGb(totalVram)}`} />
                 <SummaryRow icon={Server} label="GPU Temp" value={formatUnit(system.gpu_temp_c, "C", 0)} />
                 <SummaryRow icon={ShieldCheck} label="Power" value={formatUnit(system.gpu_power_w, "W", 1)} colorClass="text-emerald-400" />
              </div>
            </div>
          </SurfaceCard>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 mb-16">
          <SurfaceCard 
            title="Trajectory Analysis"
            badge={
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-6 bg-black/40 px-6 py-3 rounded-[1.25rem] border border-white/[0.05] backdrop-blur-xl">
                  <div className="flex items-center gap-4">
                    <span className="text-[10px] font-black uppercase text-white/20">Train</span>
                    <input type="range" min="1" max="50" value={smoothingTrain} onChange={(e) => setSmoothingTrain(parseInt(e.target.value))} className="w-20 accent-white h-1 bg-white/5 rounded-full appearance-none cursor-pointer" />
                    <span className="text-[10px] font-bold text-white/50">{smoothingTrain}</span>
                  </div>
                  <div className="w-px h-5 bg-white/10" />
                  <div className="flex items-center gap-4">
                    <span className="text-[10px] font-black uppercase text-white/20">Eval</span>
                    <input type="range" min="1" max="50" value={smoothingEval} onChange={(e) => setSmoothingEval(parseInt(e.target.value))} className="w-20 accent-white h-1 bg-white/5 rounded-full appearance-none cursor-pointer" />
                    <span className="text-[10px] font-bold text-white/50">{smoothingEval}</span>
                  </div>
                </div>
                <button onClick={() => setLossDomain(null)} className="p-3 rounded-xl bg-white/[0.05] hover:bg-white hover:text-black transition-all border border-white/5">
                   <RotateCcw size={14} />
                </button>
              </div>
            }
          >
            <div className="h-[400px] relative">
              <ResponsiveContainer>
                <AreaChart data={zoomedLossRows} margin={{ left: -20, right: 0, top: 20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#FFFFFF" stopOpacity={0.2}/><stop offset="95%" stopColor="#FFFFFF" stopOpacity={0}/></linearGradient>
                  </defs>
                  <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.01)" strokeDasharray="12 12" />
                  <XAxis dataKey="step" hide /><YAxis tick={{ fill: "rgba(255,255,255,0.15)", fontSize: 11, fontWeight: 900 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<ChartTooltip />} cursor={{ stroke: "rgba(255,255,255,0.18)", strokeWidth: 1 }} />
                  <Area type="monotone" dataKey="trainLoss" stroke="#FFFFFF" strokeWidth={4} fillOpacity={1} fill="url(#colorLoss)" dot={false} />
                  <Area type="monotone" dataKey="evalLoss" stroke="rgba(255,255,255,0.3)" strokeWidth={2} fillOpacity={0} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <ZoomMinimap data={lossRows} dataKey="trainLoss" domain={lossDomain} onDomainChange={setLossDomain} color="#FFFFFF" />
          </SurfaceCard>

          <SurfaceCard 
            title="Learning Flux"
            badge={
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-6 bg-black/40 px-6 py-3 rounded-[1.25rem] border border-white/[0.05] backdrop-blur-xl">
                  <span className="text-[10px] font-black uppercase text-white/20">Smoothing</span>
                  <input type="range" min="1" max="50" value={smoothingLr} onChange={(e) => setSmoothingLr(parseInt(e.target.value))} className="w-32 accent-white h-1 bg-white/5 rounded-full appearance-none cursor-pointer" />
                  <span className="text-[10px] font-bold text-white/50">{smoothingLr}</span>
                </div>
                <button onClick={() => setLrDomain(null)} className="p-3 rounded-xl bg-white/[0.05] hover:bg-white hover:text-black transition-all border border-white/5">
                   <RotateCcw size={14} />
                </button>
              </div>
            }
          >
            <div className="h-[400px] relative">
              <ResponsiveContainer>
                <AreaChart data={zoomedLrRows} margin={{ left: -20, right: 0, top: 20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorLr" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#FFFFFF" stopOpacity={0.2}/><stop offset="95%" stopColor="#FFFFFF" stopOpacity={0}/></linearGradient>
                  </defs>
                  <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.01)" strokeDasharray="12 12" />
                  <XAxis dataKey="step" hide /><YAxis tickFormatter={formatCompact} tick={{ fill: "rgba(255,255,255,0.15)", fontSize: 11, fontWeight: 900 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<ChartTooltip />} cursor={{ stroke: "rgba(255,255,255,0.18)", strokeWidth: 1 }} />
                  <Area type="monotone" dataKey="learningRate" stroke="#FFFFFF" strokeWidth={4} fillOpacity={1} fill="url(#colorLr)" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <ZoomMinimap data={lrRows} dataKey="learningRate" domain={lrDomain} onDomainChange={setLrDomain} color="#FFFFFF" />
          </SurfaceCard>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">
          <div className="lg:col-span-8 flex flex-col h-full">
            <SurfaceCard title="Event History" description="Neural sequence log" className="flex-1 h-full flex flex-col">
              <div className="overflow-auto h-0 flex-1 relative custom-scrollbar pr-4 -mr-4">
                <table className="w-full text-left text-[10px] border-separate border-spacing-y-4">
                  <thead>
                    <tr className="text-white/10 font-black uppercase tracking-[0.4em]">
                      <th className="py-4 px-8">Epoch</th>
                      <th className="py-4 px-8">Step Range</th>
                      <th className="py-4 px-8 text-right">Metric (PPL)</th>
                      <th className="py-4 px-8 text-right">State</th>
                    </tr>
                  </thead>
                  <tbody className="font-bold">
                    {data.epochs?.length > 0 ? (
                      data.epochs.map((e, i) => (
                        <tr key={i} className="group glass-card rounded-2xl hover:bg-white/[0.03] transition-all border-white/[0.02]">
                          <td className="py-6 px-8 rounded-l-2xl"><div className="size-10 rounded-xl bg-white/[0.05] flex items-center justify-center font-black text-white text-lg">{e.epoch}</div></td>
                          <td className="py-6 px-8"><div className="flex flex-col gap-1"><span className="font-mono text-white/50 text-sm">{e.start_global_step} → {e.end_global_step}</span></div></td>
                          <td className="py-6 px-8 text-right"><span className="font-black text-white text-2xl text-glow">{formatNumber(e.train_perplexity, 2)}</span></td>
                          <td className="py-6 px-8 text-right rounded-r-2xl"><div className="flex justify-end"><span className={`flex items-center gap-3 px-4 py-2 rounded-full text-[9px] font-black uppercase tracking-[0.3em] border ${e.complete ? 'bg-emerald-500/5 text-emerald-400 border-emerald-500/10' : 'bg-white/5 text-white/20 border-white/5'}`}>{e.complete ? "STABLE" : "SYNCING"}</span></div></td>
                        </tr>
                      ))
                    ) : (
                      <tr><td colSpan="4" className="py-40 text-center opacity-10 font-black uppercase tracking-[0.8em]">Awaiting Stream</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </SurfaceCard>
          </div>

          <div className="lg:col-span-4 space-y-8">
             <SurfaceCard title="Memory Pool">
                <div className="space-y-1">
                   <SummaryRow icon={Database} label="History" value={history.length} />
                   <SummaryRow icon={Terminal} label="Disk Used" value={`${formatFixed(system.disk_percent, 1)}%`} colorClass="text-white/60" />
                   <SummaryRow icon={RefreshCcw} label="Updated" value={system.captured_at ? new Date(system.captured_at).toLocaleTimeString() : "-"} colorClass="text-emerald-400" />
                </div>
             </SurfaceCard>
             <SurfaceCard title="Raw Config">
                <div className="p-1.5 bg-white/[0.03] rounded-2xl border border-white/[0.05] overflow-hidden">
                  <pre className="text-[11px] text-white/25 font-mono leading-relaxed p-6 bg-black/60 rounded-xl overflow-auto max-h-[400px] custom-scrollbar">
                    {JSON.stringify(data.config || {}, null, 2)}
                  </pre>
                </div>
             </SurfaceCard>
          </div>
        </div>
      </div>
    </main>
  );
}

const updatedAt = ""; 
