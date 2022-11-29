# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.distributions.binomial import Binomial

from pyro.distributions.torch_distribution import TorchDistribution


class PoissonBinomial(TorchDistribution):
    """

    :param probs: Location parameter.
    """

    arg_constraints = {"probs": constraints.nonnegative}
    support = constraints.real
    has_rsample = True

    """
    def __init__(self, loc, scale, *, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        super().__init__(self.loc.shape, validate_args=validate_args)
    """
    def __init__(self, probs,validate_args=None):
        self.N=len(probs)
        self.probs = probs
        super().__init__(self.probs.shape,validate_args=validate_args)
    """
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(PoissonBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(PoissonBinomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
    """

    """
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return math.log(2 / math.pi) - self.scale.log() - torch.logaddexp(z, -z)
    """
    """ Categorical
    def log_prob(self, value):
        if getattr(value, '_pyro_categorical_support', None) == id(self):
            # Assume value is a reshaped torch.arange(event_shape[0]).
            # In this case we can call .reshape() rather than torch.gather().
            if not torch._C._get_tracing_state():
                if self._validate_args:
                    self._validate_sample(value)
                assert value.size(0) == self.logits.size(-1)
            logits = self.logits
            if logits.dim() <= value.dim():
                logits = logits.reshape((1,) * (1 + value.dim() - logits.dim()) + logits.shape)
            if not torch._C._get_tracing_state():
                assert logits.size(-1 - value.dim()) == 1
            return logits.transpose(-1 - value.dim(), -1).squeeze(-1)
        return super().log_prob(value)
    """ 
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return math.log(2 / math.pi) - self.scale.log() - torch.logaddexp(z, -z)

    """
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = self.loc.new_empty(shape).uniform_()
        return self.icdf(u)
    """
    def rsample(self,sample_shape=torch.Size()):
        binomial = Binomial(1, self.probs)
        x = binomial.sample()
        return torch.sum(x,dtype=torch.int)
    """
    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return z.exp().atan().mul(2 / math.pi)


    def icdf(self, value):
        return value.mul(math.pi / 2).tan().log().mul(self.scale).add(self.loc)
    """