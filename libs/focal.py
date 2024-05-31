import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    nn.Module class for the Pytorch implementation of the focal loss function.
    --------------------------------------------------------------------------
    Reference: https://arxiv.org/abs/1708.02002v2
    """
    def __init__(self, alpha = 1, gamma = 2, reduction='mean'):
        """
        Initialise the FocalLoss class with default attributes.
        Args:
            alpha (float): Weighting factor for class imbalance. Default is 1.
            gamma (float): Focusing parameter. Default is 2.
            reduction (str): Specifies the reduction to apply to the output. Default is 'mean'.
                             Options are 'none' and 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        valid_reductions = ['none', 'mean']
        if self.reduction not in valid_reductions:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. "
                             f"Valid options are: {valid_reductions}")


    def forward(self, inputs, targets):
        """
        Performs the calculation of the focal loss
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # Compute the probability for the true class
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'none':
            return focal_loss
