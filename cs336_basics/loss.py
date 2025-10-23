import torch
import torch.nn as nn

from cs336_basics.model import softmax

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Computes the cross-entropy loss between logits and targets.

        Args:
            logits (torch.Tensor): The predicted logits of i+1 based on 1:i of shape (batch_size, num_classes).
            targets (torch.Tensor): The true labels of shape (batch_size,).
        Returns:
            torch.Tensor: The computed cross-entropy loss.
        """
        log_probs = softmax(logits, dim=-1).log()
        neg_log_probs_target = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = neg_log_probs_target.mean()
        return loss
        
        