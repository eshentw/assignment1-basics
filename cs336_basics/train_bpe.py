import os
from datasets import load_dataset
from .tokenizer import BPETokenizer


def download_openwebtxt_data(cache_dir: str = "./data"):
    dataset = load_dataset("Skylion007/openwebtext", cache_dir=cache_dir)
    train = dataset["train"]
    return train, None, None

def download_tinystories_data(cache_dir: str = "./data"):
    dataset = load_dataset("roneneldan/TinyStories", cache_dir=cache_dir)
    train = dataset["train"]
    valid = dataset["validation"]
    print(f"Train size: {len(train)}, Valid size: {len(valid)}")
    return train, valid, None

def download_data(cache_dir: str = "./data/openwebtxt"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if cache_dir.endswith("openwebtxt"):
        return download_openwebtxt_data(cache_dir)
    elif cache_dir.endswith("tinystories"):
        return download_tinystories_data(cache_dir)
    else:
        raise ValueError("Unsupported dataset. Please use 'openwebtxt' or 'tinystories' in the cache_dir path.")

def train_bpe(file_path: str, vocab_size: int, special_tokens: list[str], num_workers: int = 4):
    tokenizer = BPETokenizer()
    tokenizer.train(
        file_path=file_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_workers=num_workers,
    )
    return tokenizer
    