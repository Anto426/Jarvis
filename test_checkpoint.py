import os
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer
from transformers.trainer_utils import get_last_checkpoint

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(BASE_DIR, "jarvis_model")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 120
TEMPERATURE = 0.8
TOP_K = 50

# =====================================================
# LOAD TOKENIZER (CUSTOM CORRETTO)
# =====================================================

def load_tokenizer():

    base_tokenizer = ByteLevelBPETokenizer(
        vocab=os.path.join(TOKENIZER_DIR, "vocab.json"),
        merges=os.path.join(TOKENIZER_DIR, "merges.txt")
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base_tokenizer._tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>"
    )

    return tokenizer

# =====================================================
# FIND CHECKPOINT
# =====================================================

def find_latest_checkpoint():
    last_checkpoint = get_last_checkpoint(MODEL_ROOT)
    if last_checkpoint:
        print(f"Checkpoint selezionato automaticamente: {last_checkpoint}")
        return last_checkpoint
    print("Nessun checkpoint trovato, uso modello base.")
    return MODEL_ROOT

# =====================================================
# LOAD MODEL
# =====================================================

def load_model(path):

    print(f"\nCaricamento modello da: {path}")
    print(f"Device: {DEVICE}")

    model = GPT2LMHeadModel.from_pretrained(path).to(DEVICE)

    model.config.use_cache = False
    model.eval()

    return model

# =====================================================
# MANUAL GENERATION (STABILE)
# =====================================================

def generate(model, tokenizer, prompt):

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)

    if input_ids.shape[1] == 0:
        return "[Tokenizer non coerente]"

    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):

            outputs = model(input_ids=generated)
            logits = outputs.logits[:, -1, :]

            logits = logits / TEMPERATURE

            top_k = min(TOP_K, logits.size(-1))
            values, indices = torch.topk(logits, top_k)
            probs = torch.softmax(values, dim=-1)

            next_token = indices.gather(
                -1,
                torch.multinomial(probs, num_samples=1)
            )

            generated = torch.cat((generated, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    new_tokens = generated[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return text.strip()

# =====================================================
# MAIN
# =====================================================

def main():

    tokenizer = load_tokenizer()
    model_path = find_latest_checkpoint()
    model = load_model(model_path)

    print("\n=== JARVIS CHECKPOINT TEST (FIXED TOKENIZER) ===")
    print("Scrivi 'exit' per uscire.\n")

    while True:
        try:
            user_input = input("Prompt > ").strip()

            if user_input.lower() == "exit":
                break

            if not user_input:
                continue

            output = generate(model, tokenizer, user_input)

            print("\nRisposta:\n")
            print(output)
            print("\n" + "="*60 + "\n")

        except KeyboardInterrupt:
            print("\nUscita.")
            break

        except Exception as e:
            print(f"\nErrore: {e}\n")

if __name__ == "__main__":
    main()