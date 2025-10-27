import torch
import torch.nn as nn
import math
from typing import Iterable, Optional, Callable
import numpy as np


def data_loading(x: np.ndarray, batch_size: int, seq_len:int, device: torch.device):
    """
    Loads data into batches and transfers them to the specified device.

    Args:
        x: 1D array-like (np.ndarray or np.memmap), length N
        batch_size (int): The size of each batch.
        seq_len (int): The length of each sequence.
        device (torch.device): The device to transfer the batches to.
    Returns:
        Tuple[torch.Tensor]: A generator yielding batches of data on the specified device.
                             1. input sequences of shape (batch_size, seq_length)
                             2. target sequences of shape (batch_size, seq_length)
    """
    N = x.shape[0]
    num_batches = N // (batch_size * seq_len)
    x = x[:num_batches * batch_size * seq_len]
    x = x.reshape(batch_size, -1)

    for i in range(0, x.shape[1] - seq_len, seq_len):
        x_batch = x[:, i:i + seq_len]
        y_batch = x[:, i + 1:i + seq_len + 1]
        yield torch.tensor(x_batch, dtype=torch.long, device=device), torch.tensor(y_batch, dtype=torch.long, device=device)