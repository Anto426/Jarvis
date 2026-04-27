import torch
from torch.utils.data import DataLoader
from training.dataset import causal_lm_collate

def evaluate(model, dataset, batch_size=1):

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=causal_lm_collate)

    total_loss = 0
    total_steps = 0

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels", batch["input_ids"]),
            )
            total_loss += outputs.loss.item()
            total_steps += 1

    avg_loss = total_loss / total_steps
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity.item()
