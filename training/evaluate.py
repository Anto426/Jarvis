import torch
from torch.utils.data import DataLoader

def evaluate(model, dataset, batch_size=1):

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)

    total_loss = 0
    total_steps = 0

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                labels=batch["input_ids"]
            )
            total_loss += outputs.loss.item()
            total_steps += 1

    avg_loss = total_loss / total_steps
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity.item()