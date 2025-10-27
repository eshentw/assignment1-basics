import numpy as np
import torch


def load_data(file_path: str) -> np.ndarray:
    """
    Load token IDs from a binary file using memory mapping.

    Args:
        file_path (str): Path to the text file.
    Returns:
        np.ndarray: 1D array of integer token IDs.
    """
    data = np.memmap(file_path, dtype=np.int32, mode="r")
    return np.array(data, dtype=np.int32)


def data_loading(x: np.ndarray, batch_size: int, seq_len: int, device: torch.device):
    """
    Randomly sample language modeling batches from a 1D token sequence.

    Args:
        x: 1D numpy array-like of integer token IDs, length N.
        batch_size (int): Number of sequences per batch.
        seq_len (int): Context length of each sampled sequence.
        device (torch.device): Device on which to place the returned tensors.

    Yields:
        Tuple[torch.Tensor, torch.Tensor]: Input and target tensors, each of shape
        (batch_size, seq_len) and dtype torch.long, residing on `device`. The target
        tensor is the input tensor shifted by one position.
    """
    tokens = np.asarray(x)
    if tokens.ndim != 1:
        raise ValueError("`x` must be a 1D array of token IDs.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer.")
    if seq_len <= 0:
        raise ValueError("`seq_len` must be a positive integer.")

    num_tokens = tokens.shape[0]
    possible_starts = num_tokens - seq_len
    if possible_starts <= 0:
        raise ValueError("Sequence length must be smaller than the number of tokens.")

    offsets = np.arange(seq_len) # indices within each sequence
    while True:
        start_indices = np.random.choice(possible_starts, size=batch_size, replace=False)
        input_indices = start_indices[:, None] + offsets
        target_indices = input_indices + 1

        inputs = torch.tensor(tokens[input_indices], dtype=torch.long, device=device)
        targets = torch.tensor(tokens[target_indices], dtype=torch.long, device=device)
        yield inputs, targets
