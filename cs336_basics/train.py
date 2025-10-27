import torch
import torch.nn as nn
import math
from typing import Iterable, Optional, Callable


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, path: str):
    """
    Save the model and optimizer state dictionaries along with the current epoch.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        iteration (int): The current training iteration.
        path (str): The file path to save the checkpoint.
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, path)
    

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int:
    """
    Load the model and optimizer state dictionaries from a checkpoint file.

    Args:
        model (nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        path (str): The file path of the checkpoint to load.
    Returns:
        int: The iteration number at which the checkpoint was saved.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iteration = checkpoint['iteration']
    return iteration