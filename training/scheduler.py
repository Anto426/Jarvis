from transformers import get_scheduler


def build_scheduler(optimizer, num_training_steps, num_warmup_steps):
    """
    Costruisce uno scheduler cosine con warmup.
    """
    if isinstance(num_warmup_steps, float):
        num_warmup_steps = int(num_training_steps * num_warmup_steps)

    return get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
