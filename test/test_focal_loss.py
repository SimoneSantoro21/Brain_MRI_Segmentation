import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.focal import FocalLoss

def test_focal_loss_initialization():
    """
    Testing the proper initialization of the FocalLoss class.

    GIVEN: Alpha, gamma and reduction values
    WHEN: A FocalLoss object is initialised  
    THEN: The attributes are the expected ones
    """
    loss_fn = FocalLoss(alpha=0.5, gamma=1, reduction='mean')

    assert loss_fn.alpha == 0.5
    assert loss_fn.gamma == 1
    assert loss_fn.reduction == 'mean'

    with pytest.raises(ValueError):
        FocalLoss(reduction='invalid')


def test_focal_loss_forward_mean():
    """
    Testing the forward pass of FocalLoss with mean reduction.

    GIVEN: Inputs and targets tensors
    WHEN: The forward method is called with reduction='mean'
    THEN: The output should be a scalar tensor
    """
    loss_fn = FocalLoss(alpha=1, gamma=2, reduction='mean')
    inputs = torch.tensor([0.9, 0.2]).unsqueeze(0)
    targets = torch.tensor([1.0, 0.0]).unsqueeze(0)
    loss = loss_fn(inputs, targets)

    assert torch.is_tensor(loss)
    assert torch.numel(loss) == 1


def test_focal_loss_forward_sum():
    """
    Testing the forward pass of FocalLoss with sum reduction.

    GIVEN: Inputs and targets tensors
    WHEN: The forward method is called with reduction='sum'
    THEN: The output should be a scalar tensor
    """
    loss_fn = FocalLoss(alpha=1, gamma=2, reduction='sum')
    inputs = torch.tensor([0.9, 0.2]).unsqueeze(0)
    targets = torch.tensor([1.0, 0.0]).unsqueeze(0)
    loss = loss_fn(inputs, targets)

    assert torch.is_tensor(loss)
    assert torch.numel(loss) == 1


def test_focal_loss_forward_none():
    """
    Testing the forward pass of FocalLoss with no reduction.

    GIVEN: Inputs and targets tensors
    WHEN: The forward method is called with reduction='none'
    THEN: The output should be a tensor with the same shape as the inputs
    """
    loss_fn = FocalLoss(alpha=1, gamma=2, reduction='none')
    inputs = torch.tensor([0.9, 0.2]).unsqueeze(0)
    targets = torch.tensor([1.0, 0.0]).unsqueeze(0)
    loss = loss_fn(inputs, targets)

    assert torch.is_tensor(loss)
    assert loss.size() == inputs.size()


def test_focal_loss_value():
    """
    Testing the focal loss value for given inputs and targets.

    GIVEN: Inputs and targets tensors
    WHEN: The forward method is called with reduction='mean'
    THEN: The output should be close to the expected loss value
    """
    loss_fn = FocalLoss(alpha=1, gamma=2, reduction='mean')
    inputs = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    expected_loss = 0.005
    loss = loss_fn(inputs, targets)

    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-4)


def test_focal_loss_identical_tensors():
    """
    Testing the forward pass of FocalLoss with mean reduction on identical tensors.

    GIVEN: Identical inputs and targets tensors
    WHEN: The forward method is called with reduction='mean'
    THEN: The output should be 0
    """
    loss_fn = FocalLoss(alpha=1, gamma=2, reduction='mean')
    inputs = torch.tensor([[1, 1.], [1., 1.]], requires_grad=True)
    targets = torch.tensor([[1., 1.], [1., 1.]])
    loss = loss_fn(inputs, targets)
    expected_loss = 0.  

    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-3)


def test_focal_loss_different_tensors():
    """
    Testing the forward pass of FocalLoss with mean reduction on totally different tensors.

    GIVEN: Completely different inputs and targets tensors
    WHEN: The forward method is called with reduction='mean'
    THEN: The output loss should be close to the expected loss value
    """
    loss_fn = FocalLoss(alpha=1, gamma=2, reduction='mean')
    inputs = torch.tensor([[1, 1.], [1., 1.]], requires_grad=True)
    targets = torch.tensor([[0., 0.], [0., 0.]])
    loss = loss_fn(inputs, targets)
    expected_loss = 100.  

    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-3)
