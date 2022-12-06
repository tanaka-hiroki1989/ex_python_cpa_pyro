import torch
from torch.distributions import constraints

class _zero_to_one(constraints.Constraint):
    def check(self, value):
        return torch.all(value >= 0) & torch.all(value <= 1)

zero_to_one =_zero_to_one()