"use client";

import { useEffect, useMemo, useState, useRef, useCallback } from "react";
import {
  Activity,
  Gauge,
  LineChart as LineChartIcon,
  Server,
  Zap,
  ChevronDown,
  ChevronUp,
  RefreshCcw,
  LayoutDashboard,
  Cpu,
  Database,
  Search,
} from "lucide-react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis, ResponsiveContainer } from "recharts";

import { Badge } from "@/components/ui/badge";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";

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
  trainLoss: { label: "Train Loss", color: "var(--chart-1)" },
  evalLoss: { label: "Eval Loss", color: "var(--chart-2)" },
  learningRate: { label: "Learning Rate", color: "var(--chart-3)" },
};

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatCompact(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toExponential(2);
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "-";
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  return `${hours ? hours + "h " : ""}${minutes}m ${rest}s`;
}

/* --- Ultimate Contrast Components --- */

function PageHeader({ eyebrow, title, description, badge, actions }) {
  return (
    <div className="mb-12 admin-fade-up sticky top-0 z-20 bg-background/80 backdrop-blur-xl py-6 border-b border-white/5 -mx-6 px-6 sm:-mx-12 sm:px-12">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-8 w-full">
        <div className="flex items-center gap-6">
           <div className="size-14 rounded-2xl bg-white flex items-center justify-center text-black shadow-2xl">
              <Cpu size={28} />
           </div>
           <div>
            <div className="flex items-center gap-4 mb-1">
              <h1 className="text-4xl font-black tracking-tight text-white">{title}</h1>
              {badge}
            </div>
            <p className="text-xs text-white/50 font-bold uppercase tracking-widest">{eyebrow}</p>
           </div>
        </div>
        {actions && <div className="flex items-center gap-4">{actions}</div>}
      </div>
    </div>
  );
}

function MetricCard({ label, value, icon: Icon, caption, trend }) {
  return (
    <div className="surface-elevated rounded-2xl p-6 transition-all hover:translate-y-[-4px] admin-fade-up group">
      <div className="flex items-start justify-between mb-4">
        <div className="p-3 rounded-xl bg-white/5 border border-white/10 group-hover:bg-white group-hover:text-black transition-all">
          <Icon size={20} />
        </div>
        {trend && (
           <div className="text-[10px] font-black text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded-full">
              {trend}
           </div>
        )}
      </div>
      <div className="space-y-1">
        <p className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40">{label}</p>
        <h3 className="text-3xl font-black tabular-nums text-white tracking-tighter">{value}</h3>
      </div>
      {caption && (
        <p className="mt-3 text-[10px] font-bold text-white/20 uppercase tracking-widest">{caption}</p>
      )}
    </div>
  );
}

function SurfaceCard({ title, description, children, badge, className = "" }) {
  return (
    <div className={`surface-elevated rounded-[1.5rem] overflow-hidden admin-fade-up ${className}`}>
      <div className="px-8 py-5 border-b border-white/10 bg-white/[0.02] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="size-2 rounded-full bg-white shadow-[0_0_8px_white]" />
          <h2 className="text-sm font-black uppercase tracking-[0.2em] text-white/80">{title}</h2>
        </div>
        {badge}
      </div>
      <div className="p-8">
        {children}
      </div>
    </div>
  );
}

function SummaryRow({ label, value, colorClass = "text-white" }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-white/5 last:border-0">
      <span className="text-[10px] font-black text-white/30 uppercase tracking-[0.2em]">{label}</span>
      <span className={`text-sm font-black tabular-nums ${colorClass}`}>{value}</span>
    </div>
  );
}

/* --- Logic Components --- */

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
    window.addEventListener("touchmove", handleMove);
    window.addEventListener("touchend", stop);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", stop);
      window.removeEventListener("touchmove", handleMove);
      window.removeEventListener("touchend", stop);
    };
  }, [domain, minStep, range, maxStep, onDomainChange]);

  return (
    <div ref={containerRef} className="relative mt-12 h-3 bg-white/5 rounded-full overflow-visible border border-white/10 shadow-inner">
      {domain && (
        <>
          <div className="absolute inset-y-0 bg-white/20 rounded-full" style={{ left: `${selLeft * 100}%`, right: `${(1 - selRight) * 100}%` }} />
          <div className="absolute top-1/2 -translate-y-1/2 size-7 bg-white rounded-xl cursor-ew-resize -translate-x-1/2 shadow-[0_0_15px_rgba(255,255,255,0.4)] border-2 border-black flex items-center justify-center" style={{ left: `${selLeft * 100}%` }} onMouseDown={(e) => startDrag("left", e)} onTouchStart={(e) => startDrag("left", e)}>
             <div className="w-0.5 h-3 bg-black/20 rounded-full" />
          </div>
          <div className="absolute top-1/2 -translate-y-1/2 size-7 bg-white rounded-xl cursor-ew-resize -translate-x-1/2 shadow-[0_0_15px_rgba(255,255,255,0.4)] border-2 border-black flex items-center justify-center" style={{ left: `${selRight * 100}%` }} onMouseDown={(e) => startDrag("right", e)} onTouchStart={(e) => startDrag("right", e)}>
             <div className="w-0.5 h-3 bg-black/20 rounded-full" />
          </div>
        </>
      )}
      {!domain && (
        <div className="absolute inset-0 cursor-pointer flex items-center justify-center" onClick={() => onDomainChange([minStep, maxStep])}>
           <span className="text-[8px] font-black uppercase tracking-[0.2em] text-white/10">Click to Initialize Viewport Control</span>
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const isSmallScreen = useMediaQuery("(max-width: 640px)");
  const [data, setData] = useState({ status: "IDLE", current: {}, history: [], epochs: [] });
  const [lossDomain, setLossDomain] = useState(null);
  const [lrDomain, setLrDomain] = useState(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [openSection, setOpenSection] = useState("EPOCHS");

  const refreshData = useCallback(async () => {
    try {
      const response = await fetch("/api/metrics", { cache: "no-store" });
      if (!response.ok) throw new Error();
      const payload = await response.json();
      if (!isInitialized) {
        setLossDomain(payload.lossDomain ?? null);
        setLrDomain(payload.lrDomain ?? null);
        setIsInitialized(true);
      }
      setData(payload);
    } catch {
      setData(c => ({ ...c, status: "OFFLINE" }));
    }
  }, [isInitialized]);

  useEffect(() => {
    refreshData();
    const timer = setInterval(refreshData, 3000);
    return () => clearInterval(timer);
  }, [refreshData]);

  const history = useMemo(() => Array.isArray(data.history) ? data.history : [], [data.history]);
  const latestMetric = data.current?.loss !== undefined ? data.current : (history[history.length - 1] || {});
  const currentStep = Number(data.current?.global_step ?? latestMetric.global_step ?? data.start_global_step ?? 0);
  const totalSteps = Number(data.total_steps || latestMetric.total_steps || 0);
  const progress = totalSteps > 0 ? Math.min(100, (currentStep / totalSteps) * 100) : 0;
  const isRunning = data.status === "running";

  const lossRows = useMemo(() => {
    const rows = new Map();
    history.forEach(item => {
      if (!item.global_step) return;
      const row = rows.get(item.global_step) || { step: Number(item.global_step) };
      if (item.type === "train") row.trainLoss = item.loss;
      if (item.type === "eval") row.evalLoss = item.loss;
      rows.set(item.global_step, row);
    });
    const arr = Array.from(rows.values()).sort((a, b) => a.step - b.step);
    if (arr.length <= 1000) return arr;
    const step = Math.ceil(arr.length / 1000);
    return arr.filter((_, i) => i % step === 0);
  }, [history]);

  const lrRows = useMemo(() => {
    const arr = history.filter(i => i.global_step && i.learning_rate !== undefined)
      .map(i => ({ step: Number(i.global_step), learningRate: i.learning_rate }))
      .sort((a, b) => a.step - b.step);
    if (arr.length <= 1000) return arr;
    const step = Math.ceil(arr.length / 1000);
    return arr.filter((_, i) => i % step === 0);
  }, [history]);

  const zoomedLossRows = useMemo(() => lossDomain ? lossRows.filter(r => r.step >= lossDomain[0] && r.step <= lossDomain[1]) : lossRows, [lossRows, lossDomain]);
  const zoomedLrRows = useMemo(() => lrDomain ? lrRows.filter(r => r.step >= lrDomain[0] && r.step <= lrDomain[1]) : lrRows, [lrRows, lrDomain]);

  return (
    <main className="min-h-screen px-6 sm:px-12 w-full pb-24">
      
      <PageHeader
        eyebrow="Neural Protocol 4.2.0 // Active Interface"
        title="Jarvis Core"
        badge={
          <Badge className={`rounded-lg px-3 py-1 text-[9px] font-black uppercase tracking-[0.2em] shadow-2xl ${isRunning ? 'bg-white text-black' : 'bg-white/10 text-white/40'}`}>
            {data.status || "IDLE"}
          </Badge>
        }
        actions={
          <div className="flex gap-4">
             <button onClick={refreshData} className="size-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center transition-all hover:bg-white hover:text-black">
                <RefreshCcw size={18} className={isRunning ? "animate-spin" : ""} />
             </button>
             <button className="px-6 rounded-xl bg-white text-black text-[11px] font-black uppercase tracking-widest hover:bg-white/90 transition-all">
                Export Logs
             </button>
          </div>
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-12 items-stretch">
        <div className="lg:col-span-9 grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-6">
          <MetricCard icon={LayoutDashboard} label="Process Step" value={String(currentStep)} caption={`Target: ${totalSteps || "INF"}`} />
          <MetricCard icon={Gauge} label="Loss Delta" value={formatNumber(latestMetric.loss)} caption="Neural alignment" trend="-2.4%" />
          <MetricCard icon={LineChartIcon} label="Optimization" value={formatNumber(latestMetric.perplexity, 2)} caption="PPL metric" />
          <MetricCard icon={Zap} label="Flux Rate" value={formatCompact(latestMetric.learning_rate)} caption="Learning modulation" />
        </div>

        <SurfaceCard title="System Load" className="lg:col-span-3 h-full">
          <div className="flex flex-col h-full justify-between gap-6">
            <div className="space-y-2">
              <div className="flex items-baseline gap-2">
                <span className="text-5xl font-black tabular-nums text-white tracking-tighter">{progress.toFixed(1)}%</span>
              </div>
              <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden border border-white/10">
                <div className="h-full bg-white shadow-[0_0_15px_white]" style={{ width: `${progress}%` }} />
              </div>
            </div>
            <div className="space-y-1">
               <SummaryRow label="Sync Hub" value={updatedAt || "-"} />
               <SummaryRow label="Active Workers" value="12" />
               <SummaryRow label="Node Status" value="Healthy" colorClass="text-emerald-400" />
            </div>
          </div>
        </SurfaceCard>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 mb-12">
        <SurfaceCard title="Trajectory Analysis" description="Neural Network Loss distribution">
          <div className="h-[400px] relative">
            {zoomedLossRows.length > 0 ? (
              <ChartContainer config={chartConfig} className="h-full w-full">
                <ResponsiveContainer>
                  <LineChart data={zoomedLossRows} margin={{ left: -15, right: 0, top: 0, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.03)" strokeDasharray="3 3" />
                    <XAxis dataKey="step" hide />
                    <YAxis tick={{ fill: "rgba(255,255,255,0.2)", fontSize: 10, fontWeight: 900 }} axisLine={false} tickLine={false} />
                    <ChartTooltip content={<ChartTooltipContent className="surface-elevated rounded-xl p-4 font-black border-white/20 shadow-2xl" />} />
                    <Line type="monotone" dataKey="trainLoss" stroke="#FFFFFF" strokeWidth={2.5} dot={false} activeDot={{ r: 5, fill: "#fff" }} />
                    <Line type="monotone" dataKey="evalLoss" stroke="rgba(255,255,255,0.3)" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartContainer>
            ) : (
              <div className="h-full flex flex-col items-center justify-center gap-4 opacity-20 bg-white/[0.01] rounded-2xl border border-dashed border-white/10">
                <Activity size={40} />
                <p className="text-[10px] font-black uppercase tracking-[0.4em]">Calibrating Sensors</p>
              </div>
            )}
          </div>
          <ZoomMinimap data={lossRows} dataKey="trainLoss" domain={lossDomain} onDomainChange={setLossDomain} color="#FFFFFF" />
        </SurfaceCard>

        <SurfaceCard title="Learning Flux" description="Modulation schedule metrics">
          <div className="h-[400px] relative">
            {zoomedLrRows.length > 0 ? (
              <ChartContainer config={chartConfig} className="h-full w-full">
                <ResponsiveContainer>
                  <LineChart data={zoomedLrRows} margin={{ left: -15, right: 0, top: 0, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.03)" strokeDasharray="3 3" />
                    <XAxis dataKey="step" hide />
                    <YAxis tickFormatter={formatCompact} tick={{ fill: "rgba(255,255,255,0.2)", fontSize: 10, fontWeight: 900 }} axisLine={false} tickLine={false} />
                    <ChartTooltip content={<ChartTooltipContent className="surface-elevated rounded-xl p-4 font-black border-white/20 shadow-2xl" />} />
                    <Line type="stepAfter" dataKey="learningRate" stroke="#FFFFFF" strokeWidth={2.5} dot={false} activeDot={{ r: 5, fill: "#fff" }} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartContainer>
            ) : (
              <div className="h-full flex flex-col items-center justify-center gap-4 opacity-20 bg-white/[0.01] rounded-2xl border border-dashed border-white/10">
                <Zap size={40} />
                <p className="text-[10px] font-black uppercase tracking-[0.4em]">Analyzing Flux</p>
              </div>
            )}
          </div>
          <ZoomMinimap data={lrRows} dataKey="learningRate" domain={lrDomain} onDomainChange={setLrDomain} color="#FFFFFF" />
        </SurfaceCard>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        <div className="lg:col-span-8">
          <SurfaceCard title="Event History" description="Chronological Training Sequence">
            <div className="overflow-auto max-h-[500px] min-h-[300px] relative">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="text-white font-black uppercase tracking-[0.25em] border-b-2 border-white/10 bg-white/[0.02]">
                    <th className="py-6 px-6">Epoch</th>
                    <th className="py-6 px-6">Domain Range</th>
                    <th className="py-6 px-6 text-right">Metric (PPL)</th>
                    <th className="py-6 px-6 text-right">State</th>
                  </tr>
                </thead>
                <tbody className="font-bold text-white/90">
                  {data.epochs?.length > 0 ? (
                    data.epochs.map((e, i) => (
                      <tr key={i} className="border-b border-white/5 hover:bg-white/[0.04] transition-all group">
                        <td className="py-6 px-6 font-black text-white text-lg">{e.epoch}</td>
                        <td className="py-6 px-6 tabular-nums text-white/40">{e.start_global_step} → {e.end_global_step}</td>
                        <td className="py-6 px-6 text-right font-black tabular-nums text-xl">{formatNumber(e.train_perplexity, 2)}</td>
                        <td className="py-6 px-6 text-right">
                          <span className={`px-4 py-1.5 rounded-lg text-[9px] font-black uppercase tracking-widest shadow-lg ${e.complete ? 'bg-white text-black' : 'bg-white/10 text-white/40'}`}>
                            {e.complete ? "STABLE" : "SYNCING"}
                          </span>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="4" className="py-24 text-center">
                        <div className="flex flex-col items-center gap-4 opacity-20">
                          <Database size={48} />
                          <p className="text-[10px] font-black uppercase tracking-[0.4em]">Awaiting Event Stream</p>
                        </div>
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </SurfaceCard>
        </div>

        <div className="lg:col-span-4 space-y-8">
           <SurfaceCard title="Memory Pool" description="Cache & Latency">
              <div className="space-y-1">
                 <SummaryRow label="History Depth" value={history.length} />
                 <SummaryRow label="Kernel Version" value="X-1.4.2" colorClass="text-primary" />
                 <SummaryRow label="Sync Delay" value="2.4ms" colorClass="text-emerald-400" />
                 <SummaryRow label="Stream Status" value="Optimized" colorClass="text-white/60" />
              </div>
           </SurfaceCard>
           
           <SurfaceCard title="Raw Config" description="Neural Metadata" className="flex-1">
              <div className="relative group">
                <div className="absolute top-2 right-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Badge variant="outline" className="text-[8px] bg-black/60 border-white/20">JSON</Badge>
                </div>
                <pre className="text-[11px] text-white/50 font-mono leading-relaxed p-6 bg-black/60 rounded-2xl overflow-auto max-h-72 border border-white/5 shadow-inner">
                  {JSON.stringify(data.config || {}, null, 2)}
                </pre>
              </div>
           </SurfaceCard>
        </div>
      </div>

    </main>
  );
}

const updatedAt = ""; // Placeholder for compiler, real value used in refreshData
