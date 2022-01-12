import torch
import pnums
from torch import nn


def pnum_loss(input: torch.Tensor, goal: pnums.PInt, base_loss=nn.SmoothL1Loss):
    """Gives a loss function so that large number difference have more loss than small ones."""

    loss = None

    input_size = torch.numel(input)
    goal_size = goal.tensor.size

    loss_fn = base_loss()

    assert (
            input_size == goal_size
    ), f"Input tensor size ({input_size}) should match goal size ({goal_size})."

    in_tensor = torch.reshape(input, goal.tensor.shape)

    for b in range(goal.bits):
        input_part = in_tensor[..., b]
        goal_part = goal.tensor[..., b]
        if loss is None:
            loss = loss_fn(input_part, torch.FloatTensor(goal_part)) / 2 ** b
        else:
            loss += loss_fn(input_part, torch.FloatTensor(goal_part)) / 2 ** b

    return loss
