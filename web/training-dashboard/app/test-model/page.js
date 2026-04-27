"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Bot,
  BrainCircuit,
  Clock,
  Database,
  Loader2,
  Monitor,
  Power,
  RefreshCcw,
  RotateCcw,
  Send,
  Server,
  SlidersHorizontal,
  Terminal,
  User,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";

const initialMessages = [
  {
    id: "welcome",
    role: "assistant",
    content: "Ciao, sono Jarvis. Dimmi pure cosa vuoi testare.",
    createdAt: new Date().toISOString(),
  },
];

function formatStatus(value) {
  return value ? "online" : "standby";
}

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

function StatusBar({ status }) {
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

  const workerRunning = Boolean(status?.worker?.running);

  return (
    <div className="w-full bg-white/[0.02] border-b border-white/[0.05] py-2 px-6 sm:px-10 flex items-center justify-between text-[9px] font-black uppercase tracking-[0.2em] text-white/30 backdrop-blur-3xl">
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2.5">
          <div className={`size-1.5 rounded-full ${workerRunning ? "bg-emerald-500 animate-pulse shadow-[0_0_10px_#10b981]" : "bg-white/20"}`} />
          <span className="text-white/50">Inference Core: {workerRunning ? "Active" : "Standby"}</span>
        </div>
        <div className="hidden md:flex items-center gap-2">
          <Server size={10} className="text-white/10" />
          <span>Jarvis-Protocol-01</span>
        </div>
      </div>
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2 font-mono">
          <Monitor size={10} className="text-white/10" />
          <span className="text-white/40">Mode: Inference</span>
        </div>
        <div className="flex items-center gap-2 font-mono">
          <Clock size={10} className="text-white/10" />
          <span className="text-white/40">{time || "--:--:--"}</span>
        </div>
      </div>
    </div>
  );
}

function PageHeader({ status, onRefresh, isRefreshing }) {
  const checkpointReady = Boolean(status?.checkpoint?.modelFile);
  const tokenizerReady = Boolean(status?.tokenizer?.available);

  return (
    <div className="mb-12 admin-fade-up sticky top-0 z-20 bg-background/20 backdrop-blur-3xl py-6 border-b border-white/[0.03] -mx-6 px-6 sm:-mx-10 sm:px-10 group/header">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-8 w-full">
        <div className="flex items-center gap-6">
          <div className="size-12 rounded-2xl bg-white flex items-center justify-center text-black shadow-xl">
            <BrainCircuit size={24} />
          </div>
          <div>
            <p className="text-[9px] text-white/15 font-black uppercase tracking-[0.4em] mb-1">Neural Inference</p>
            <div className="flex items-center gap-4">
              <h1 className="text-4xl font-black tracking-tighter text-white text-glow leading-none">Chat Test</h1>
              <Badge className={`rounded-full px-4 py-1.5 text-[10px] font-black uppercase tracking-[0.3em] transition-all duration-700 ${checkpointReady && tokenizerReady ? 'bg-white text-black animate-pulse' : 'bg-white/5 text-white/30 border-white/10'}`}>
                {checkpointReady && tokenizerReady ? "READY" : "WAITING"}
              </Badge>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={onRefresh}
            className="size-12 rounded-xl bg-white/[0.02] border border-white/10 flex items-center justify-center transition-all hover:bg-white hover:text-black group"
          >
            <RefreshCcw size={18} className={`${isRefreshing ? "animate-spin" : ""} group-hover:rotate-180 transition-transform duration-700`} />
          </button>
          <Link
            href="/"
            className="px-8 h-12 rounded-xl bg-white/[0.04] border border-white/10 text-white text-[12px] font-black uppercase tracking-[0.24em] hover:bg-white hover:text-black transition-all flex items-center gap-3"
          >
            <ArrowLeft size={14} /> Dashboard
          </Link>
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, tone = "text-white" }) {
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
        </div>
        <p className={`text-base font-black tracking-tight tabular-nums truncate ${tone}`}>{value || "-"}</p>
      </div>
    </div>
  );
}

function CheckpointWarning({ checkpoint }) {
  const warnings = checkpoint?.warnings || [];
  if (!warnings.length) return null;

  const stage = checkpoint?.pipelineStep?.id || "stage sconosciuto";
  const source = checkpoint?.source === "official" ? "official" : "snapshot";

  return (
    <div className="mb-12 rounded-xl border border-amber-400/20 bg-amber-400/[0.06] px-6 py-5 text-amber-100/80 admin-fade-up">
      <div className="flex items-start gap-4">
        <div className="mt-0.5 shrink-0 rounded-lg bg-amber-300 text-black p-2">
          <AlertTriangle size={17} />
        </div>
        <div className="min-w-0">
          <div className="text-[10px] font-black uppercase tracking-[0.28em] text-amber-200/70">
            Checkpoint {source} · {stage}
          </div>
          <p className="mt-2 text-sm leading-6 text-amber-50/75">{warnings.join(" ")}</p>
        </div>
      </div>
    </div>
  );
}

function ControlSlider({ label, value, min, max, step, onChange }) {
  return (
    <label className="block">
      <div className="flex items-center justify-between gap-4 mb-3">
        <span className="text-[10px] font-black uppercase tracking-[0.25em] text-white/20">{label}</span>
        <span className="text-[11px] font-black tabular-nums text-white/60">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full accent-white h-1 bg-white/5 rounded-full appearance-none cursor-pointer"
      />
    </label>
  );
}

function ModeControl({ value, onChange }) {
  return (
    <div className="grid grid-cols-2 gap-1 rounded-2xl border border-white/[0.06] bg-black/45 p-1">
      {["classic", "chat"].map((mode) => (
        <button
          key={mode}
          type="button"
          onClick={() => onChange(mode)}
          className={`h-10 rounded-xl text-[10px] font-black uppercase tracking-[0.24em] transition-all ${value === mode ? "bg-white text-black" : "text-white/30 hover:text-white/70"}`}
        >
          {mode}
        </button>
      ))}
    </div>
  );
}

function SurfaceCard({ title, icon: Icon, children, className = "", description }) {
  return (
    <div className={`surface-elevated rounded-[2.5rem] admin-fade-up group/card ${className}`}>
      <div className="px-10 py-7 border-b border-white/[0.04] flex items-center justify-between relative z-10">
        <div className="flex items-center gap-4">
          {Icon ? (
            <div className="size-10 rounded-xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center text-white/45 group-hover/card:bg-white group-hover/card:text-black transition-all duration-500">
              <Icon size={17} />
            </div>
          ) : (
            <div className="size-3 rounded-full bg-white shadow-[0_0_15px_rgba(255,255,255,0.6)] animate-pulse" />
          )}
          <div>
            <h2 className="text-xs font-black uppercase tracking-[0.4em] text-white/80">{title}</h2>
            {description && <p className="text-[9px] font-bold text-white/10 uppercase tracking-widest mt-1">{description}</p>}
          </div>
        </div>
      </div>
      <div className={`p-10 relative z-10 ${className.includes('flex-col') ? 'flex-1 flex flex-col' : ''}`}>
        {children}
      </div>
    </div>
  );
}

function ChatMessage({ message }) {
  const isUser = message.role === "user";
  const Icon = isUser ? User : Bot;

  return (
    <div className={`flex items-start gap-6 ${isUser ? "justify-end" : "justify-start"} admin-fade-up`}>
      {!isUser && (
        <div className="size-12 shrink-0 rounded-2xl bg-white text-black flex items-center justify-center shadow-2xl">
          <Icon size={20} />
        </div>
      )}
      <div className={`max-w-[86%] md:max-w-[75%] rounded-[2rem] px-8 py-6 border shadow-2xl transition-all duration-500 ${isUser ? "bg-white text-black border-white hover:shadow-white/5" : "bg-white/[0.03] text-white/80 border-white/[0.06] hover:bg-white/[0.05]"}`}>
        <div className={`mb-3 flex items-center justify-between gap-8 text-[9px] font-black uppercase tracking-[0.3em] ${isUser ? "text-black/40" : "text-white/20"}`}>
          <span className="flex items-center gap-2">
            {isUser ? <User size={10} /> : <Bot size={10} />}
            {isUser ? "Authorized User" : "Jarvis System"}
          </span>
          <span className="font-mono opacity-50">{new Date(message.createdAt).toLocaleTimeString()}</span>
        </div>
        <p className="text-base leading-8 tracking-tight selection:bg-black selection:text-white">{message.content}</p>
      </div>
      {isUser && (
        <div className="size-12 shrink-0 rounded-2xl bg-white/[0.04] border border-white/[0.08] text-white/50 flex items-center justify-center shadow-xl">
          <Icon size={20} />
        </div>
      )}
    </div>
  );
}

function TypingRow() {
  return (
    <div className="flex items-start gap-4">
      <div className="size-10 shrink-0 rounded-xl bg-white text-black flex items-center justify-center shadow-xl">
        <Bot size={17} />
      </div>
      <div className="rounded-2xl border border-white/[0.06] bg-black/55 px-5 py-4 text-white/35 flex items-center gap-3">
        <Loader2 size={16} className="animate-spin" />
        <span className="text-[10px] font-black uppercase tracking-[0.3em]">generazione</span>
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
      const payload = await response.json();
      setStatus(payload);
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

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isGenerating]);

  const checkpointName = status?.checkpoint?.name || "nessun checkpoint";
  const checkpointFile = status?.checkpoint?.modelFile || "missing";
  const tokenizerState = status?.tokenizer?.available ? "jarvis.model" : "missing";
  const workerState = status?.worker?.running ? "active" : "idle";
  const modelReady = Boolean(status?.checkpoint?.modelFile && status?.tokenizer?.available);
  const canCloseModel = Boolean(status?.worker?.running) && !isClosing;
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
      setMessages((items) => [...items, makeMessage("assistant", "Modello chiuso. La prossima risposta lo ricarichera'.")]);
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
    event.preventDefault();
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
    <main className="min-h-screen w-full pb-24 font-sans overflow-x-hidden">
      <StatusBar status={status} />

      <div className="px-6 sm:px-10">
        <PageHeader status={status} onRefresh={refreshStatus} isRefreshing={isRefreshing} />

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-12">
          <StatCard icon={Database} label="Checkpoint" value={checkpointName} />
          <StatCard icon={Terminal} label="Model File" value={checkpointFile} />
          <StatCard icon={Bot} label="Tokenizer" value={tokenizerState} />
          <StatCard icon={Activity} label="Core State" value={workerState} tone={status?.worker?.running ? "text-emerald-300" : "text-white/50"} />
        </div>

        <CheckpointWarning checkpoint={status?.checkpoint} />

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-10 items-start">
          <div className="xl:col-span-8">
            <SurfaceCard title="Neural Sequence" description="Direct interface stream" className="min-h-[750px] flex flex-col relative overflow-hidden">
              <div className="absolute top-0 right-0 p-10 opacity-[0.02] pointer-events-none">
                <BrainCircuit size={400} />
              </div>

              <div className="flex-1 overflow-auto custom-scrollbar pr-4 -mr-4 space-y-10 min-h-[500px]">
                {messages.map((message) => (
                  <ChatMessage key={message.id} message={message} />
                ))}
                {isGenerating && <TypingRow />}
                <div ref={messagesEndRef} />
              </div>

              {error && (
                <div className="mt-8 rounded-2xl border border-red-500/20 bg-red-500/5 px-6 py-4 text-xs font-black uppercase tracking-widest text-red-400/80 animate-pulse">
                  System Error: {error}
                </div>
              )}

              <form onSubmit={submitMessage} className="mt-10 flex items-end gap-6 relative z-10">
                <div className="flex-1 relative group">
                  <textarea
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        submitMessage(event);
                      }
                    }}
                    className="min-h-16 max-h-44 w-full resize-none rounded-[1.5rem] border border-white/[0.06] bg-black/45 px-8 py-5 text-base leading-relaxed text-white/90 outline-none transition-all placeholder:text-white/10 focus:border-white/20 focus:bg-black/60 shadow-inner"
                    placeholder="Input command..."
                    rows={1}
                  />
                  <div className="absolute bottom-4 right-6 flex items-center gap-3 text-[9px] font-black uppercase tracking-widest text-white/10 group-focus-within:text-white/30 transition-colors">
                    <span>Shift+Enter for newline</span>
                  </div>
                </div>
                <button
                  type="submit"
                  disabled={!canSend}
                  className="size-16 shrink-0 rounded-2xl bg-white text-black hover:scale-105 active:scale-95 disabled:scale-100 disabled:cursor-not-allowed disabled:opacity-20 transition-all shadow-[0_20px_50px_rgba(255,255,255,0.15)] flex items-center justify-center group"
                >
                  {isGenerating ? <Loader2 size={24} className="animate-spin" /> : <Send size={24} className="group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />}
                </button>
              </form>
            </SurfaceCard>
          </div>

          <div className="xl:col-span-4 space-y-10">
            <SurfaceCard title="Sampling" description="Response parameters" icon={SlidersHorizontal}>
              <div className="space-y-10">
                <ModeControl value={promptMode} onChange={setPromptMode} />
                <div className="space-y-8">
                  <ControlSlider label="Max Tokens" value={maxNewTokens} min={1} max={512} step={1} onChange={setMaxNewTokens} />
                  <ControlSlider label="Temperature" value={temperature} min={0.1} max={2} step={0.1} onChange={setTemperature} />
                  <ControlSlider label="Top P" value={topP} min={0.1} max={1} step={0.05} onChange={setTopP} />
                  <ControlSlider label="Repeat Penalty" value={repetitionPenalty} min={1} max={1.6} step={0.05} onChange={setRepetitionPenalty} />
                </div>
              </div>
            </SurfaceCard>

            <SurfaceCard title="Session" description="Active metrics" icon={Terminal}>
              <div className="space-y-2">
                <div className="flex items-center justify-between py-6 border-b border-white/[0.04]">
                  <div className="flex items-center gap-4">
                    <div className="size-2 rounded-full bg-white/20" />
                    <span className="text-[10px] font-black uppercase tracking-[0.3em] text-white/20">Turns</span>
                  </div>
                  <span className="text-xl font-black tabular-nums text-white">{chatTurns}</span>
                </div>
                <div className="flex items-center justify-between py-6 border-b border-white/[0.04]">
                  <div className="flex items-center gap-4">
                    <div className="size-2 rounded-full bg-white/20" />
                    <span className="text-[10px] font-black uppercase tracking-[0.3em] text-white/20">Input</span>
                  </div>
                  <span className="text-xl font-black tabular-nums text-white/60">{lastStats?.input_tokens || "-"}</span>
                </div>
                <div className="flex items-center justify-between py-6 border-b border-white/[0.04]">
                  <div className="flex items-center gap-4">
                    <div className="size-2 rounded-full bg-emerald-500/40" />
                    <span className="text-[10px] font-black uppercase tracking-[0.3em] text-white/20">Generated</span>
                  </div>
                  <span className="text-xl font-black tabular-nums text-emerald-400">{lastStats?.generated_tokens || "-"}</span>
                </div>
                
                <div className="grid grid-cols-2 gap-4 mt-10">
                  <button
                    type="button"
                    onClick={resetChat}
                    className="h-14 rounded-2xl bg-white/[0.03] border border-white/10 text-[10px] font-black uppercase tracking-[0.3em] text-white/40 hover:bg-white hover:text-black transition-all flex items-center justify-center gap-3 group"
                  >
                    <RotateCcw size={16} className="group-hover:rotate-180 transition-transform duration-500" />
                    Reset
                  </button>
                  <button
                    type="button"
                    onClick={closeModel}
                    disabled={!canCloseModel}
                    className="h-14 rounded-2xl bg-red-500/10 border border-red-500/20 text-[10px] font-black uppercase tracking-[0.3em] text-red-400 hover:bg-red-500 hover:text-white disabled:opacity-5 transition-all flex items-center justify-center gap-3"
                  >
                    {isClosing ? <Loader2 size={16} className="animate-spin" /> : <Power size={16} />}
                    Close Model
                  </button>
                </div>
              </div>
            </SurfaceCard>
          </div>
        </div>
      </div>
    </main>
  );
}
