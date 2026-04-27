"""Quick debug: inspect actual columns of sapienzanlp/piqa_italian."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from training.paths import configure_cache_env
configure_cache_env()

from datasets import load_dataset

ds = load_dataset("sapienzanlp/piqa_italian", split="train", streaming=True)
for i, sample in enumerate(ds):
    print("=== KEYS ===")
    print(sorted(sample.keys()))
    print()
    for k, v in sample.items():
        print(f"  {k!r}: {v!r}")
    if i >= 1:
        break
