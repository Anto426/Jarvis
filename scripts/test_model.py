import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

sys.path.insert(0, str(ROOT))

import sentencepiece as spm
import torch
from safetensors.torch import load_file

from models.init_model import initialize_model
from training.paths import configure_cache_env, get_path


configure_cache_env()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = get_path("model_output_dir", create=True)
STOP_TOKENS = ("<|end|>", "<|user|>", "<|system|>", "<|context|>")
CHECKPOINT_POLICY = os.environ.get("JARVIS_TEST_CHECKPOINT_POLICY", "auto").strip().lower()
SEPARATOR_LINE_RE = re.compile(r"(?m)^\s*(?:[\*\-=~_#\.]\s*){8,}\s*$")
LONG_SYMBOL_RUN_RE = re.compile(r"(?:[\*\-=~_#\.]\s*){10,}")


def log(message, stream=sys.stderr):
    print(message, file=stream, flush=True)


def read_json_file(path):
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def checkpoint_model_file(path):
    path = Path(path)
    for filename in ("model.safetensors", "pytorch_model.bin"):
        if (path / filename).exists():
            return filename
    return None


def checkpoint_is_complete(path):
    path = Path(path)
    return bool(
        checkpoint_model_file(path)
        and (path / "optimizer.bin").exists()
        and (path / "scheduler.bin").exists()
    )


def checkpoint_step_number(path):
    try:
        return int(Path(path).name.split("_")[1])
    except (IndexError, ValueError):
        return None


def get_latest_active_checkpoint(path):
    checkpoints = []
    for item in Path(path).iterdir():
        if not item.is_dir() or not item.name.startswith("step_"):
            continue

        step = checkpoint_step_number(item)
        if step is None or not checkpoint_model_file(item):
            continue

        checkpoints.append((step, item, checkpoint_is_complete(item)))

    if not checkpoints:
        raise ValueError("Nessun checkpoint trovato")

    checkpoints.sort(key=lambda item: item[0])
    complete = [item for item in checkpoints if item[2]]
    return str((complete or checkpoints)[-1][1])


def get_latest_official_checkpoint():
    latest_path = Path(CHECKPOINT_DIR) / "official" / "latest.json"
    latest = read_json_file(latest_path)
    if not isinstance(latest, dict) or latest.get("kind") != "train":
        return None

    artifact_path = latest.get("artifact_path")
    if not artifact_path:
        return None

    artifact = Path(artifact_path)
    if artifact.exists() and checkpoint_model_file(artifact):
        return str(artifact)

    return None


def select_checkpoint():
    if CHECKPOINT_POLICY in {"auto", "official"}:
        official = get_latest_official_checkpoint()
        if official:
            return official, "official"
        if CHECKPOINT_POLICY == "official":
            raise ValueError("Nessun checkpoint ufficiale di training trovato")

    if CHECKPOINT_POLICY not in {"auto", "active"}:
        log(f"Policy checkpoint non riconosciuta: {CHECKPOINT_POLICY}; uso auto.")

    return get_latest_active_checkpoint(CHECKPOINT_DIR), "active"


def checkpoint_meta(path):
    meta = read_json_file(Path(path) / "jarvis_checkpoint_meta.json")
    return meta if isinstance(meta, dict) else {}


def checkpoint_status(path, source):
    path = Path(path)
    meta = checkpoint_meta(path)
    pipeline_step = meta.get("pipeline_step") if isinstance(meta.get("pipeline_step"), dict) else {}
    stage_id = pipeline_step.get("id")
    warnings = []

    if source != "official":
        warnings.append("Snapshot attivo: non e un checkpoint ufficiale pubblicato dalla pipeline.")
    if not checkpoint_is_complete(path):
        warnings.append("Checkpoint incompleto: per inferenza uso il modello, ma non e sicuro per riprendere il training.")
    if stage_id == "step_1_italian_corpus":
        warnings.append("Stage 1 LM: completa testo, non risponde ancora come assistente Jarvis.")
    if source != "official" and not stage_id:
        warnings.append("Metadata stage assente: potrebbe essere uno snapshot vecchio o parziale.")

    return {
        "source": source,
        "model_file": checkpoint_model_file(path),
        "complete": checkpoint_is_complete(path),
        "step": checkpoint_step_number(path),
        "pipeline_step": pipeline_step,
        "warnings": warnings,
    }


def clean_completion(text):
    value = str(text or "").strip()
    for token in STOP_TOKENS:
        if token in value:
            value = value.split(token, 1)[0].strip()

    if value.startswith("<|assistant|>"):
        value = value[len("<|assistant|>"):].strip()

    value = SEPARATOR_LINE_RE.sub("", value)
    value = LONG_SYMBOL_RUN_RE.sub(" ", value)
    return value


class JarvisModelRuntime:
    def __init__(self, stream=sys.stderr):
        self.stream = stream
        log("Cerco checkpoint modello...", self.stream)
        self.checkpoint_path, self.checkpoint_source = select_checkpoint()
        self.checkpoint_status = checkpoint_status(self.checkpoint_path, self.checkpoint_source)
        log(f"Carico: {self.checkpoint_path}", self.stream)
        for warning in self.checkpoint_status["warnings"]:
            log(f"Avviso checkpoint: {warning}", self.stream)

        self.model = initialize_model()
        self._load_checkpoint()

        self.model.to(DEVICE)
        self.model.eval()

        tokenizer_path = os.path.join(str(get_path("tokenizer_dir", create=True)), "jarvis.model")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer non trovato: {tokenizer_path}")

        self.tokenizer_path = tokenizer_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        log(f"Modello pronto su {DEVICE}", self.stream)

    def _load_checkpoint(self):
        safe_model_path = os.path.join(self.checkpoint_path, "model.safetensors")
        torch_model_path = os.path.join(self.checkpoint_path, "pytorch_model.bin")

        if os.path.exists(safe_model_path):
            state_dict = load_file(safe_model_path)
        elif os.path.exists(torch_model_path):
            state_dict = torch.load(torch_model_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"Nessun file modello trovato in {self.checkpoint_path}: "
                "atteso model.safetensors o pytorch_model.bin"
            )

        self.model.load_state_dict(state_dict, strict=False)

    def status(self):
        return {
            "device": DEVICE,
            "checkpoint": os.path.relpath(self.checkpoint_path, ROOT),
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_status": self.checkpoint_status,
            "tokenizer": os.path.relpath(self.tokenizer_path, ROOT),
        }

    def generate(
        self,
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
    ):
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt vuoto")

        max_new_tokens = max(1, min(int(max_new_tokens), 512))
        temperature = max(0.01, min(float(temperature), 2.0))
        top_p = max(0.05, min(float(top_p), 1.0))
        repetition_penalty = max(1.0, min(float(repetition_penalty), 2.0))

        input_ids = self.sp.encode(prompt, out_type=int)
        if not input_ids:
            raise ValueError("Il prompt non ha prodotto token validi")

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
        max_context = int(getattr(self.model.config, "max_position_embeddings", 2048) or 2048)
        if input_tensor.shape[1] + max_new_tokens > max_context:
            keep_tokens = max(1, max_context - max_new_tokens)
            input_tensor = input_tensor[:, -keep_tokens:]

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_tensor,
                attention_mask=torch.ones_like(input_tensor),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.sp.eos_id(),
            )

        tokens = output[0].tolist()
        prompt_token_count = input_tensor.shape[1]
        completion_tokens = tokens[prompt_token_count:]

        return {
            **self.status(),
            "prompt": prompt,
            "text": self.sp.decode(tokens),
            "completion": clean_completion(self.sp.decode(completion_tokens)),
            "input_tokens": prompt_token_count,
            "generated_tokens": len(completion_tokens),
        }


def load_runtime(stream=sys.stderr):
    # initialize_model prints by design; keep stdio worker stdout JSON-only.
    with contextlib.redirect_stdout(stream):
        return JarvisModelRuntime(stream=stream)


def run_stdio_server():
    try:
        runtime = load_runtime(sys.stderr)
    except Exception as exc:
        log(f"Errore caricamento modello: {exc}", sys.stderr)
        sys.exit(1)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
            command = payload.get("command", "generate")

            if command == "status":
                result = runtime.status()
            elif command == "generate":
                result = runtime.generate(
                    payload.get("prompt", ""),
                    max_new_tokens=payload.get("max_new_tokens", 100),
                    temperature=payload.get("temperature", 0.8),
                    top_p=payload.get("top_p", 0.9),
                    repetition_penalty=payload.get("repetition_penalty", 1.1),
                )
            else:
                raise ValueError(f"Comando non supportato: {command}")

            print(json.dumps({"ok": True, **result}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)


def run_prompt(args):
    runtime = load_runtime(sys.stderr)
    result = runtime.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print(json.dumps({"ok": True, **result}, ensure_ascii=False, indent=2))


def run_interactive_cli():
    runtime = load_runtime(sys.stdout)

    print()
    print("Scrivi qualcosa (exit per uscire)")
    print()

    while True:
        prompt = input("> ")

        if prompt.lower() == "exit":
            break

        response = runtime.generate(prompt)

        print()
        print("Jarvis >")
        print(response["text"])
        print("-" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Test locale del modello Jarvis")
    parser.add_argument("--stdio-server", action="store_true", help="Avvia worker JSONL per la dashboard Next.js")
    parser.add_argument("--prompt", help="Genera una singola risposta e stampa JSON")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stdio_server:
        run_stdio_server()
    elif args.prompt:
        run_prompt(args)
    else:
        run_interactive_cli()


if __name__ == "__main__":
    main()
