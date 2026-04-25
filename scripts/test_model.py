import os
import torch
import sentencepiece as spm
from safetensors.torch import load_file

from models.init_model import initialize_model
from training.paths import configure_cache_env, get_path


configure_cache_env()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = get_path("model_output_dir", create=True)


def get_latest_checkpoint(path):
    checkpoints = [
        d for d in os.listdir(path)
        if d.startswith("step_")
    ]

    if not checkpoints:
        raise ValueError("❌ Nessun checkpoint trovato")

    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(x.split("_")[1])
    )

    return os.path.join(path, checkpoints[-1])


# --------------------------
# 🔥 CARICA CHECKPOINT
# --------------------------

print("🔍 Cerco ultimo checkpoint...")
checkpoint_path = get_latest_checkpoint(CHECKPOINT_DIR)
print(f"📦 Carico: {checkpoint_path}")

model = initialize_model()

state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"))
model.load_state_dict(state_dict, strict=False)

model.to(DEVICE)
model.eval()


tokenizer_path = os.path.join(str(get_path("tokenizer_dir", create=True)), "jarvis.model")

if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer non trovato: {tokenizer_path}")

sp = spm.SentencePieceProcessor()
sp.load(tokenizer_path)


print(f"\n✅ Modello pronto su {DEVICE}")
print("💬 Scrivi qualcosa (exit per uscire)\n")


# --------------------------
# 🔥 GENERAZIONE
# --------------------------

def generate(prompt, max_new_tokens=100):

    input_ids = sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=sp.eos_id()
        )

    tokens = output[0].tolist()
    return sp.decode(tokens)


# --------------------------
# 🔥 LOOP INTERATTIVO
# --------------------------

while True:
    prompt = input("👤 > ")

    if prompt.lower() == "exit":
        break

    response = generate(prompt)

    print("\n🤖 >")
    print(response)
    print("-" * 60)
