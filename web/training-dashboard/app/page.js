"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Clock3,
  Cpu,
  Gauge,
  LineChart as LineChartIcon,
  Radio,
  Server,
  Zap
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis
} from "recharts";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from "@/components/ui/card";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent
} from "@/components/ui/chart";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";

const chartConfig = {
  trainLoss: {
    label: "Train loss",
    color: "var(--chart-1)"
  },
  evalLoss: {
    label: "Eval loss",
    color: "var(--chart-2)"
  },
  learningRate: {
    label: "Learning rate",
    color: "var(--chart-3)"
  }
};

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toLocaleString("it-IT", { maximumFractionDigits: digits });
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

function MetricTile({ icon: Icon, label, value, detail, tone = "cyan" }) {
  const tones = {
    cyan: "text-cyan-300 bg-cyan-300/10 ring-cyan-300/20",
    amber: "text-amber-300 bg-amber-300/10 ring-amber-300/20",
    violet: "text-violet-300 bg-violet-300/10 ring-violet-300/20",
    emerald: "text-emerald-300 bg-emerald-300/10 ring-emerald-300/20"
  };

  return (
    <Card className="rounded-lg border-white/10 bg-zinc-950/85 py-0 shadow-none">
      <CardContent className="grid h-24 grid-cols-[1fr_auto] items-center gap-3 p-4">
        <div className="min-w-0">
          <div className="text-[11px] font-medium uppercase text-zinc-500">
            {label}
          </div>
          <div className="mt-1 truncate text-2xl font-semibold tabular-nums text-zinc-50">
            {value}
          </div>
          <div className="mt-1 truncate text-xs text-zinc-500">{detail}</div>
        </div>
        <div className={`flex size-9 items-center justify-center rounded-md ring-1 ${tones[tone]}`}>
          <Icon className="size-4" />
        </div>
      </CardContent>
    </Card>
  );
}

function EmptyChart({ text }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center text-sm text-zinc-500">
      {text}
    </div>
  );
}

function ChartCard({ title, description, children, empty }) {
  return (
    <Card className="rounded-lg border-white/10 bg-zinc-950/85 py-0 shadow-none">
      <CardHeader className="border-b border-white/10 px-4 py-3">
        <CardTitle className="text-sm font-semibold text-zinc-100">{title}</CardTitle>
        <CardDescription className="text-xs">{description}</CardDescription>
      </CardHeader>
      <CardContent className="relative p-3">
        {empty ? <EmptyChart text={empty} /> : null}
        {children}
      </CardContent>
    </Card>
  );
}

export default function Home() {
  const [data, setData] = useState({
    status: "waiting",
    current: {},
    history: [],
    epochs: []
  });

  useEffect(() => {
    let alive = true;

    async function refresh() {
      try {
        const response = await fetch("/api/metrics", { cache: "no-store" });
        if (!response.ok) throw new Error("metrics unavailable");
        const payload = await response.json();
        if (alive) setData(payload);
      } catch {
        if (alive) setData(current => ({ ...current, status: "offline" }));
      }
    }

    refresh();
    const timer = setInterval(refresh, 3000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const history = useMemo(
    () => (Array.isArray(data.history) ? data.history : []),
    [data.history]
  );
  const epochs = useMemo(
    () => (Array.isArray(data.epochs) ? data.epochs : []),
    [data.epochs]
  );

  const latestTrain = lastItem(history, item => item.type === "train");
  const latestMetric = data.current?.loss !== undefined ? data.current : latestTrain;
  const totalSteps = Number(data.total_steps || latestMetric.total_steps || 0);
  const currentStep = Number(
    data.current?.global_step ?? latestMetric.global_step ?? data.start_global_step ?? 0
  );
  const progress = totalSteps > 0 ? Math.min(100, (currentStep / totalSteps) * 100) : 0;
  const status = data.status || "waiting";
  const isRunning = status === "running";
  const updatedAt = data.updated_at
    ? new Date(data.updated_at).toLocaleString("it-IT")
    : "in attesa dei primi dati";

  const lossRows = useMemo(() => {
    const rows = new Map();
    for (const item of history) {
      if (!item.global_step) continue;
      const row = rows.get(item.global_step) || { step: item.global_step };
      if (item.type === "train") row.trainLoss = item.loss;
      if (item.type === "eval") row.evalLoss = item.loss;
      rows.set(item.global_step, row);
    }
    return Array.from(rows.values()).sort((a, b) => a.step - b.step);
  }, [history]);

  const lrRows = useMemo(
    () =>
      history
        .filter(item => item.global_step && item.learning_rate !== undefined)
        .map(item => ({
          step: item.global_step,
          learningRate: item.learning_rate
        })),
    [history]
  );

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="border-b border-white/10 bg-zinc-950/95">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-4 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
          <div className="flex items-center gap-4">
            <div className="flex size-11 items-center justify-center rounded-lg border border-cyan-300/20 bg-cyan-300/10">
              <Server className="size-5 text-cyan-300" />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-2xl font-semibold text-zinc-50">Jarvis Training</h1>
                <Badge
                  variant={isRunning ? "default" : "secondary"}
                  className="rounded-md"
                >
                  <Radio className="size-3" />
                  {status}
                </Badge>
              </div>
              <p className="mt-1 flex items-center gap-2 text-sm text-zinc-400">
                <Clock3 className="size-3.5" />
                Aggiornato: {updatedAt}
              </p>
            </div>
          </div>

          <div className="w-full rounded-lg border border-white/10 bg-white/[0.03] p-3 lg:w-80">
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="text-zinc-400">Avanzamento</span>
              <span className="font-medium tabular-nums text-zinc-100">
                {formatNumber(progress, 1)}%
              </span>
            </div>
            <Progress value={progress} className="h-2" />
            <div className="mt-2 text-xs tabular-nums text-zinc-500">
              Step {currentStep || 0} / {totalSteps || "-"}
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto grid w-full max-w-7xl gap-4 px-4 py-4 sm:px-6 lg:px-8">
        <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <MetricTile
            icon={Activity}
            label="Step"
            value={`${currentStep || 0}/${totalSteps || "-"}`}
            detail={`Epoca ${latestMetric.epoch || "-"}`}
          />
          <MetricTile
            icon={Gauge}
            label="Loss"
            value={formatNumber(latestMetric.loss)}
            detail="Ultimo punto train"
            tone="amber"
          />
          <MetricTile
            icon={LineChartIcon}
            label="Perplexity"
            value={formatNumber(latestMetric.perplexity, 2)}
            detail="Derivata dalla loss"
            tone="violet"
          />
          <MetricTile
            icon={Zap}
            label="Learning rate"
            value={formatCompact(latestMetric.learning_rate)}
            detail="Scheduler"
            tone="emerald"
          />
        </section>

        <section className="grid gap-4 xl:grid-cols-[minmax(0,1.45fr)_minmax(360px,0.55fr)]">
          <ChartCard
            title="Loss"
            description="Andamento train ed eval per step"
            empty={lossRows.length ? null : "In attesa dei punti loss"}
          >
            <ChartContainer config={chartConfig} className="h-72 w-full">
              <LineChart data={lossRows} margin={{ left: 4, right: 14, top: 12, bottom: 0 }}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" />
                <XAxis dataKey="step" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} width={48} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Line
                  type="monotone"
                  dataKey="trainLoss"
                  stroke="var(--color-trainLoss)"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  activeDot={{ r: 4 }}
                  connectNulls
                />
                <Line
                  type="monotone"
                  dataKey="evalLoss"
                  stroke="var(--color-evalLoss)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 4 }}
                  connectNulls
                />
              </LineChart>
            </ChartContainer>
          </ChartCard>

          <ChartCard
            title="Learning rate"
            description="Curva dello scheduler"
            empty={lrRows.length ? null : "In attesa dei punti LR"}
          >
            <ChartContainer config={chartConfig} className="h-72 w-full">
              <LineChart data={lrRows} margin={{ left: 4, right: 14, top: 12, bottom: 0 }}>
                <CartesianGrid vertical={false} strokeDasharray="3 3" />
                <XAxis dataKey="step" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} width={58} tickFormatter={formatCompact} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Line
                  type="monotone"
                  dataKey="learningRate"
                  stroke="var(--color-learningRate)"
                  strokeWidth={2}
                  dot={{ r: 2 }}
                  activeDot={{ r: 4 }}
                />
              </LineChart>
            </ChartContainer>
          </ChartCard>
        </section>

        <Accordion
          defaultValue={["epochs"]}
          type="multiple"
          className="rounded-lg border border-white/10 bg-zinc-950/85 px-4"
        >
          <AccordionItem value="epochs" className="border-white/10">
            <AccordionTrigger className="py-3 text-sm font-semibold text-zinc-100 hover:no-underline">
              Epoche registrate
            </AccordionTrigger>
            <AccordionContent>
              <div className="overflow-auto pb-2">
                <Table>
                  <TableHeader>
                    <TableRow className="border-white/10 hover:bg-transparent">
                      <TableHead>Epoca</TableHead>
                      <TableHead>Step iniziale</TableHead>
                      <TableHead>Step finale</TableHead>
                      <TableHead>Loss media</TableHead>
                      <TableHead>Perplexity</TableHead>
                      <TableHead>Durata</TableHead>
                      <TableHead>Stato</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {epochs.length ? (
                      epochs.map(epoch => (
                        <TableRow
                          key={`${epoch.epoch}-${epoch.end_global_step}`}
                          className="border-white/10"
                        >
                          <TableCell>{epoch.epoch}</TableCell>
                          <TableCell className="tabular-nums">{epoch.start_global_step}</TableCell>
                          <TableCell className="tabular-nums">{epoch.end_global_step}</TableCell>
                          <TableCell className="tabular-nums">{formatNumber(epoch.train_loss)}</TableCell>
                          <TableCell className="tabular-nums">{formatNumber(epoch.train_perplexity, 2)}</TableCell>
                          <TableCell>{formatDuration(epoch.duration_seconds)}</TableCell>
                          <TableCell>
                            <Badge variant={epoch.complete ? "default" : "secondary"}>
                              {epoch.complete ? "completa" : "parziale"}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow className="border-white/10 hover:bg-transparent">
                        <TableCell colSpan={7} className="h-20 text-center text-zinc-500">
                          Nessuna epoca registrata.
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </div>
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="details" className="border-white/10">
            <AccordionTrigger className="py-3 text-sm font-semibold text-zinc-100 hover:no-underline">
              Dettagli sessione
            </AccordionTrigger>
            <AccordionContent>
              <div className="grid gap-4 pb-4 text-sm sm:grid-cols-3">
                <div>
                  <p className="text-zinc-500">Metriche raccolte</p>
                  <p className="mt-1 font-medium tabular-nums text-zinc-100">{history.length}</p>
                </div>
                <div>
                  <p className="text-zinc-500">Step iniziale</p>
                  <p className="mt-1 font-medium tabular-nums text-zinc-100">
                    {data.start_global_step ?? "-"}
                  </p>
                </div>
                <div>
                  <p className="text-zinc-500">Creato</p>
                  <p className="mt-1 font-medium text-zinc-100">
                    {data.created_at ? new Date(data.created_at).toLocaleString("it-IT") : "-"}
                  </p>
                </div>
              </div>
              <Separator className="bg-white/10" />
              <pre className="mt-4 max-h-56 overflow-auto rounded-md bg-black/40 p-3 text-xs text-zinc-300">
                {JSON.stringify(data.config || {}, null, 2)}
              </pre>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </main>
  );
}
