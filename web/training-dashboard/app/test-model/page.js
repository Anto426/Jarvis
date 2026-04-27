"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  AlertTriangle,
  ArrowLeft,
  Bot,
  Cpu,
  Database,
  History,
  Layers,
  Loader2,
  MessageSquare,
  Power,
  RefreshCcw,
  RotateCcw,
  Send,
  Settings2,
  Terminal,
  Zap,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";

const initialMessages = [
  {
    id: "welcome",
    role: "assistant",
    content: "Sistemi online. Pronto per l'analisi del checkpoint caricato. In cosa posso esserti utile?",
    createdAt: new Date().toISOString(),
  },
];

const promptPresets = [
  "Analisi di sistema: continua...",
  "Stato del training attuale",
  "Diagnostica errore",
];

function makeMessage(role, content) {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    role,
    content,
    createdAt: new Date().toISOString(),
  };
}

function cleanAssistantText(text) {
  return String(text || "")
    .split("<|end|>")[0]
    .replace(/^<\|assistant\|>\s*/i, "")
    .replace(/^assistant\s*:\s*/i, "")
    .trim();
}

function buildModelPrompt(messages, mode) {
  const lastUserMessage = [...messages].reverse().find((message) => message.role === "user");
  if (mode === "classic") {
    return lastUserMessage?.content.trim() || "";
  }

  const recent = messages
    .filter((message) => message.id !== "welcome")
    .slice(-10)
    .map((message) => {
      const roleToken = message.role === "assistant" ? "<|assistant|>" : "<|user|>";
      return `${roleToken}\n${message.content.trim()}\n<|end|>`;
    });

  return [...recent, "<|assistant|>\n"].join("\n");
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "-" : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function GlassPanel({ title, icon: Icon, action, children, className = "", contentClassName = "p-6", style = {} }) {
  return (
    <section className={`relative flex flex-col rounded-3xl border border-white/5 bg-white/[0.02] backdrop-blur-md overflow-hidden ${className}`} style={style}>
      <div className="flex shrink-0 min-h-14 items-center justify-between border-b border-white/5 px-6">
        <div className="flex items-center gap-3">
          {Icon ? <Icon size={18} className="text-white/50" /> : null}
          <h2 className="text-xs font-bold uppercase tracking-wider text-white/80">{title}</h2>
        </div>
        {action}
      </div>
      <div className={`relative flex-1 flex flex-col min-h-0 ${contentClassName}`}>{children}</div>
    </section>
  );
}

function DetailRow({ label, value, tone = "text-white/70" }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-white/5 py-3 last:border-b-0 group">
      <span className="text-[11px] font-semibold uppercase tracking-wider text-white/40 group-hover:text-white/60 transition-colors">{label}</span>
      <span className={`min-w-0 truncate text-right text-xs font-medium ${tone}`}>{value || "-"}</span>
    </div>
  );
}

function ModernStatBox({ label, value, subValue, icon: Icon, color = "white" }) {
  const colorMap = {
    white: "text-white",
    emerald: "text-emerald-400",
    amber: "text-amber-400",
    blue: "text-blue-400",
  };

  return (
    <div className="group relative overflow-hidden rounded-3xl border border-white/5 bg-white/[0.02] backdrop-blur-md p-6 transition-colors duration-300 hover:bg-white/[0.04]">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          {Icon && <Icon size={16} className="text-white/40 group-hover:text-white/60 transition-colors" />}
          <span className="text-[11px] font-semibold uppercase tracking-wider text-white/50">{label}</span>
        </div>
      </div>
      <div className="mt-4">
        <div className={`text-3xl sm:text-4xl font-bold tracking-tight ${colorMap[color] || colorMap.white}`}>
          {value || "-"}
        </div>
        {subValue && <div className="mt-2 text-xs font-medium text-white/40">{subValue}</div>}
      </div>
    </div>
  );
}

function ControlSlider({ label, value, min, max, step, onChange }) {
  return (
    <label className="block group">
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-white/40 group-hover:text-white/60 transition-colors">{label}</span>
        <span className="font-mono text-[10px] font-bold text-white/80 bg-white/5 px-2 py-1 rounded-md border border-white/5 shadow-inner">{value}</span>
      </div>
      <div className="relative h-1.5 rounded-full bg-black/50 shadow-inner overflow-hidden border border-white/5">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          className="absolute inset-0 w-full cursor-pointer appearance-none bg-transparent outline-none z-10 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(255,255,255,0.8)] [&::-webkit-slider-thumb]:transition-transform hover:[&::-webkit-slider-thumb]:scale-110"
        />
        <div 
          className="absolute top-0 left-0 h-full rounded-full bg-white/20 pointer-events-none transition-all" 
          style={{ width: `${((value - min) / (max - min)) * 100}%` }} 
        />
      </div>
    </label>
  );
}

function ModeControl({ value, onChange }) {
  return (
    <div className="grid grid-cols-2 gap-1 rounded-xl border border-white/5 bg-black/40 p-1 shadow-inner">
      {["classic", "chat"].map((mode) => (
        <button
          key={mode}
          type="button"
          onClick={() => onChange(mode)}
          className={`h-8 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all duration-300 ${
            value === mode 
              ? "bg-white/10 text-white shadow-[0_0_15px_rgba(255,255,255,0.1)] border border-white/20" 
              : "text-white/40 hover:text-white/70 hover:bg-white/5 border border-transparent"
          }`}
        >
          {mode}
        </button>
      ))}
    </div>
  );
}

function ChatMessage({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`relative max-w-[85%] rounded-3xl px-6 py-5 shadow-xl ${
        isUser 
          ? "bg-white text-black" 
          : "border border-white/5 bg-white/[0.02] backdrop-blur-md text-white/90"
      }`}>
        <div className={`mb-3 flex items-center gap-3 text-[10px] font-bold uppercase tracking-wider ${
          isUser ? "text-black/50" : "text-white/40"
        }`}>
          <div className="flex items-center gap-2">
            {isUser ? <History size={12} /> : <Bot size={14} className="text-emerald-400" />}
            {isUser ? "Operatore" : "J.A.R.V.I.S."}
          </div>
          <span className="font-mono font-medium opacity-50">{formatTime(message.createdAt)}</span>
        </div>
        <p className="whitespace-pre-wrap break-words text-[13px] leading-relaxed font-medium">
          {message.content}
        </p>
        
        {!isUser && (
          <div className="absolute -left-px top-1/2 h-8 w-0.5 -translate-y-1/2 bg-emerald-400/50 shadow-[0_0_10px_rgba(52,211,153,0.5)] rounded-full" />
        )}
      </div>
    </div>
  );
}

export default function TestModelPage() {
  const [status, setStatus] = useState(null);
  const [messages, setMessages] = useState(initialMessages);
  const [draft, setDraft] = useState("");
  const [promptMode, setPromptMode] = useState("classic");
  const [maxNewTokens, setMaxNewTokens] = useState(100);
  const [temperature, setTemperature] = useState(0.8);
  const [topP, setTopP] = useState(0.9);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.1);
  const [lastStats, setLastStats] = useState(null);
  const [error, setError] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const messagesEndRef = useRef(null);

  const refreshStatus = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const response = await fetch("/api/model/status", { cache: "no-store" });
      setStatus(await response.json());
    } catch {
      setStatus(null);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    const initial = setTimeout(refreshStatus, 0);
    const timer = setInterval(refreshStatus, 5000);
    return () => {
      clearTimeout(initial);
      clearInterval(timer);
    };
  }, [refreshStatus]);

  const checkpoint = status?.checkpoint;
  const workerRunning = Boolean(status?.worker?.running);
  const tokenizerReady = Boolean(status?.tokenizer?.available);
  const modelReady = Boolean(checkpoint?.modelFile && tokenizerReady);
  const canCloseModel = workerRunning && !isClosing;
  const chatTurns = Math.max(0, messages.filter((message) => message.id !== "welcome").length);
  const canSend = useMemo(() => draft.trim().length > 0 && !isGenerating && modelReady, [draft, isGenerating, modelReady]);

  function resetChat() {
    setMessages(initialMessages);
    setDraft("");
    setError("");
    setLastStats(null);
  }

  async function closeModel() {
    if (!canCloseModel) return;

    setIsClosing(true);
    setError("");

    try {
      const response = await fetch("/api/model/generate", {
        method: "DELETE",
        cache: "no-store",
      });
      const payload = await response.json();

      if (!response.ok || payload.ok === false) {
        throw new Error(payload.error || "Chiusura modello fallita");
      }

      setLastStats(null);
      setMessages((items) => [...items, makeMessage("assistant", "Protocollo di chiusura completato. Modello scaricato dalla memoria.")]);
      setStatus((current) => ({
        ...current,
        worker: {
          ...(current?.worker || {}),
          running: false,
          stopping: false,
          pending: 0,
          lastError: null,
        },
      }));
      setTimeout(refreshStatus, 500);
    } catch (err) {
      setError(err.message || "Chiusura modello fallita");
    } finally {
      setIsClosing(false);
    }
  }

  async function submitMessage(event) {
    if (event) event.preventDefault();
    if (!canSend) return;

    const userMessage = makeMessage("user", draft.trim());
    const nextMessages = [...messages, userMessage];
    const prompt = buildModelPrompt(nextMessages, promptMode);

    setMessages(nextMessages);
    setDraft("");
    setIsGenerating(true);
    setError("");

    try {
      const response = await fetch("/api/model/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          maxNewTokens,
          temperature,
          topP,
          repetitionPenalty,
        }),
      });

      const payload = await response.json();
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.error || "Generazione fallita");
      }

      const assistantText = cleanAssistantText(payload.completion || payload.text);
      setMessages((items) => [
        ...items,
        makeMessage("assistant", assistantText || payload.completion || payload.text || "..."),
      ]);
      setLastStats(payload);
      refreshStatus();
    } catch (err) {
      const message = err.message || "Errore generazione";
      setError(message);
      setMessages((items) => [...items, makeMessage("assistant", message)]);
    } finally {
      setIsGenerating(false);
    }
  }

  return (
    <main className="mx-auto max-w-[1600px] min-h-screen w-full overflow-x-hidden px-4 py-8 font-sans sm:px-8 lg:px-12 bg-[#0a0a0a] text-white selection:bg-white/20">
      {/* Header Section */}
      <header className="mb-10 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="flex h-10 w-10 items-center justify-center rounded-xl border border-white/5 bg-white/[0.02] text-white/50 transition-colors hover:bg-white/10 hover:text-white"
            >
              <ArrowLeft size={18} className="transition-transform hover:-translate-x-1" />
            </Link>
            <div className="h-10 w-px bg-white/10" />
            <div className="flex flex-wrap items-center gap-3">
              <Badge className={`rounded-lg px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider border-none ${
                modelReady 
                  ? "bg-emerald-500/10 text-emerald-400 shadow-[0_0_15px_rgba(52,211,153,0.15)]" 
                  : "bg-white/5 text-white/50 border border-white/10"
              }`}>
                {modelReady ? "System Online" : "System Standby"}
              </Badge>
              <span className="text-[10px] font-mono font-bold text-white/40 tracking-wider">
                NODE_ID: {checkpoint?.pipelineStep?.id || "NULL"}
              </span>
            </div>
          </div>
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-white md:text-4xl">
              Model <span className="text-white/40">Interface</span>
            </h1>
            <p className="mt-2 text-xs font-medium text-white/40">
              Laboratorio di test e validazione checkpoint
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={refreshStatus}
            className="flex h-10 w-10 items-center justify-center rounded-xl border border-white/5 bg-white/[0.02] text-white/60 transition-colors hover:bg-white/10 hover:text-white"
          >
            <RefreshCcw size={16} className={isRefreshing ? "animate-spin" : ""} />
          </button>
        </div>
      </header>

      {/* Quick Stats Grid */}
      <section className="mb-8 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <ModernStatBox 
          label="Checkpoint Active" 
          value={checkpoint?.name || "None"} 
          subValue={checkpoint?.source}
          icon={Zap}
          color="white"
        />
        <ModernStatBox 
          label="Model Weights" 
          value={checkpoint?.modelFile ? "Loaded" : "Missing"} 
          subValue={checkpoint?.modelFile}
          icon={Layers}
          color={checkpoint?.modelFile ? "emerald" : "amber"}
        />
        <ModernStatBox 
          label="Tokenizer" 
          value={tokenizerReady ? "Ready" : "Offline"} 
          subValue="HuggingFace / Local"
          icon={Cpu}
          color={tokenizerReady ? "emerald" : "amber"}
        />
        <ModernStatBox 
          label="Process Engine" 
          value={workerRunning ? "Active" : "Idle"} 
          subValue={status?.worker?.startedAt ? `Started ${formatTime(status.worker.startedAt)}` : "Waiting for trigger"}
          icon={Terminal}
          color={workerRunning ? "blue" : "white"}
        />
      </section>

      {checkpoint?.warnings?.length > 0 && (
        <div className="mb-8">
          <div className="rounded-3xl border border-amber-500/20 bg-amber-500/5 px-6 py-5 flex items-center gap-5">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-amber-500/10 text-amber-400">
              <AlertTriangle size={20} />
            </div>
            <div>
              <div className="text-[11px] font-bold uppercase tracking-wider text-amber-400 mb-1">Nota di sistema</div>
              <p className="text-sm font-medium text-amber-200/80">{checkpoint.warnings.join(" ")}</p>
            </div>
          </div>
        </div>
      )}

      {/* Main Workspace */}
      <div className="grid grid-cols-1 gap-8 xl:grid-cols-[1fr_420px]">
        {/* Chat Console */}
        <div className="relative min-h-[600px] xl:min-h-0">
          <div className="h-full xl:absolute xl:inset-0">
            <GlassPanel 
              title="Neural Console" 
              icon={MessageSquare} 
              className="h-full min-h-0"
              contentClassName=""
            >
          <div className="flex flex-1 flex-col min-h-0 overflow-hidden">
            <div className="flex-1 overflow-y-auto p-6 sm:p-8 custom-scrollbar space-y-6">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              {isGenerating && (
                <div className="flex items-center gap-4 px-6 py-4 rounded-3xl border border-white/5 bg-white/[0.02] w-fit">
                  <Loader2 size={16} className="animate-spin text-emerald-400" />
                  <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400/80">Generazione in corso...</span>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {error && (
              <div className="mx-6 mb-4 rounded-2xl border border-red-500/20 bg-red-500/5 p-4 text-xs font-bold text-red-400 flex items-center gap-3">
                <AlertTriangle size={14} />
                {error}
              </div>
            )}

            <div className="p-6 border-t border-white/5 bg-black/20">
              <div className="mb-4 flex flex-wrap gap-2">
                {promptPresets.map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => setDraft(preset)}
                    className="rounded-full border border-white/5 bg-white/[0.02] px-4 py-2 text-[10px] font-bold uppercase tracking-wider text-white/50 transition-colors hover:bg-white/10 hover:text-white"
                  >
                    {preset}
                  </button>
                ))}
              </div>
              
              <div className="relative group">
                <textarea
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      submitMessage();
                    }
                  }}
                  rows={draft.split('\n').length > 1 ? 3 : 1}
                  className="w-full resize-none rounded-2xl border border-white/10 bg-white/[0.02] backdrop-blur-md pl-6 pr-16 py-4 text-sm leading-relaxed text-white outline-none placeholder:text-white/30 transition-colors focus:bg-white/[0.04] focus:border-white/20 min-h-[56px] max-h-48 custom-scrollbar"
                  placeholder={modelReady ? "Inserisci istruzioni per J.A.R.V.I.S..." : "In attesa di inizializzazione sistema..."}
                  disabled={!modelReady || isGenerating}
                />
                <button
                  onClick={submitMessage}
                  disabled={!canSend}
                  className="absolute right-3 top-3 h-10 w-10 flex items-center justify-center rounded-xl bg-white text-black transition-all hover:scale-105 active:scale-95 disabled:opacity-20 disabled:hover:scale-100 shadow-[0_0_15px_rgba(255,255,255,0.2)]"
                >
                  {isGenerating ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                </button>
              </div>
            </div>
          </div>
        </GlassPanel>
          </div>
        </div>

        {/* Sidebar Controls */}
        <aside className="space-y-6">
          <GlassPanel title="Configuration" icon={Settings2}>
            <div className="space-y-6">
              <div>
                <div className="text-[10px] font-bold uppercase tracking-wider text-white/40 mb-3 px-1">Prompt Engine</div>
                <ModeControl value={promptMode} onChange={setPromptMode} />
              </div>
              
              <div className="space-y-2">
                <ControlSlider label="Max Tokens" value={maxNewTokens} min={1} max={512} step={1} onChange={setMaxNewTokens} />
                <ControlSlider label="Temperature" value={temperature} min={0.1} max={2} step={0.1} onChange={setTemperature} />
                <ControlSlider label="Top P" value={topP} min={0.1} max={1} step={0.05} onChange={setTopP} />
                <ControlSlider label="Repetition Pen." value={repetitionPenalty} min={1} max={1.6} step={0.05} onChange={setRepetitionPenalty} />
              </div>
            </div>
          </GlassPanel>

          <GlassPanel title="System Telemetry" icon={Database}>
            <div className="space-y-1">
              <DetailRow label="Pipeline Node" value={checkpoint?.pipelineStep?.id || "N/A"} />
              <DetailRow label="Evolution Step" value={checkpoint?.step ? `#${checkpoint.step}` : "N/A"} />
              <DetailRow label="Sequence Depth" value={String(chatTurns)} />
              <DetailRow label="Input Load" value={lastStats?.input_tokens ? `${lastStats.input_tokens} tok` : "N/A"} />
              <DetailRow label="Compute Output" value={lastStats?.generated_tokens ? `${lastStats.generated_tokens} tok` : "N/A"} />
              <DetailRow label="Hardware" value={lastStats?.device || "CPU/Auto"} />
              <DetailRow 
                label="Validation" 
                value={checkpoint?.complete ? "Verified" : "Unfinished"} 
                tone={checkpoint?.complete ? "text-emerald-400" : "text-amber-400"} 
              />
            </div>
          </GlassPanel>

          <GlassPanel 
            title="Core Session" 
            icon={Terminal}
            action={
              <button
                type="button"
                onClick={closeModel}
                disabled={!canCloseModel}
                className="flex h-8 items-center gap-2 rounded-lg border border-red-500/20 bg-red-500/5 px-3 text-[10px] font-bold uppercase tracking-wider text-red-400 transition-colors hover:bg-red-500/20 hover:text-red-300 disabled:opacity-20 group"
              >
                {isClosing ? <Loader2 size={12} className="animate-spin" /> : <Power size={12} className="group-hover:rotate-12 transition-transform" />}
                Unload Core
              </button>
            }
          >
            <div className="mb-4">
              <button
                type="button"
                onClick={resetChat}
                className="flex h-10 w-full items-center justify-center gap-2 rounded-xl border border-white/5 bg-white/[0.02] text-[10px] font-bold uppercase tracking-wider text-white/50 transition-colors hover:bg-white/10 hover:text-white"
              >
                <RotateCcw size={14} />
                Purge Chat Buffer
              </button>
            </div>
            <div className="h-40 overflow-y-auto rounded-2xl border border-white/5 bg-black/40 p-4 font-mono text-[10px] text-white/30 custom-scrollbar leading-relaxed shadow-inner">
              {(status?.worker?.logs || []).length ? (
                status.worker.logs.map((line, index) => (
                  <div key={`${line}-${index}`} className="mb-1">
                    <span className="text-white/10 mr-2">[{index.toString().padStart(3, '0')}]</span>
                    {line}
                  </div>
                ))
              ) : (
                <div className="flex items-center gap-2 italic">
                  <div className="h-1.5 w-1.5 rounded-full bg-white/20 animate-pulse" />
                  No active logs in buffer.
                </div>
              )}
            </div>
          </GlassPanel>
        </aside>
      </div>
    </main>
  );
}
