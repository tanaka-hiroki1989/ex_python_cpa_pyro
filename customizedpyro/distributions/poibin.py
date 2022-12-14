import math
#import statistics
import torch
#from torch.distributions.utils import broadcast_all
from torch.distributions.binomial import Binomial
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.constraints import real_vector

class PoissonBinomial(TorchDistribution):
    """
    :param probs: Location parameter.
    """
    arg_constraints = {"p": real_vector}
    has_rsample = True

    def __init__(self, p,validate_args=None):
        self.N=len(p)
        self.p = p
        #support = constraints.integer_interval(0,self.N-1)
        super().__init__(self.p.shape,validate_args=validate_args)
        self.prob = self.pmf(self.p)
    
    def pmf(self, p):
        q = [p_j/(1-p_j) for p_j in p]
        T = torch.tensor([math.fsum([math.pow(q_j,i) for q_j in q]) for i in range(self.N)])
        #T[0]はつかわない.
        prob=[0]*self.N
        for k in range(self.N):
            if k==0:
                prob[k] = math.prod([1.0 - p[i] for i in range(self.N)])
            else:
                list = [math.pow(-1.0,i-1) * prob[k-i] * T[i] for i in range(1,k+1)]
                #print(k,list)
                prob[k] = torch.tensor(math.fsum(list)*(1.0/k))
        return prob

    def log_prob(self, value):
        return torch.tensor(math.log(self.prob[value]))

    def rsample(self,sample_shape=torch.Size()):
        binomial = Binomial(1, self.p)
        x = binomial.sample()
        return torch.sum(x,dtype=torch.int)

    def enumerate_support(self, expand=True):
        result = super().enumerate_support(expand=expand)
        if not expand:
            result._pyro_categorical_support = id(self)
        return result