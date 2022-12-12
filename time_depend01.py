import torch
import pyro
from  pyro.distributions import Gamma,Normal
from pyro.distributions.constraints import positive,real_vector
from customizedpyro.distributions.poibin import PoissonBinomial
# from customizedtorch.constrains import zero_to_one

class vb2:
    def __init__(self,init_a1,init_b1,init_a2,init_b2):
        self.init_a1=init_a1
        self.init_b1=init_b1
        self.init_a2=init_a2 
        self.init_b2=init_b2
    def model(self,data):   
        N = data.size(0)
        a1 = 1.0
        b1 = 1.0
        a2 = 1.0
        b2 = 1.0
        lambda1 = pyro.sample("lambda1", Gamma(a1,b1))
        lambda2 = pyro.sample("lambda2", Gamma(a2,b2))
        p = torch.tensor([0.0]*(N-1),dtype=torch.double)
        tau = pyro.sample("tau", PoissonBinomial(torch.sigmoid(p)))
        z_seq_before = torch.tensor([lambda1 * i for i in range(tau+1)])
        z_seq_after = torch.tensor([lambda1 * (tau-1) +lambda2 * (i-tau) for i in range(tau+1,N)])
        z_seq = torch.concat([z_seq_before,z_seq_after])

        with pyro.plate("data", N):
            pyro.sample("obs", Normal(z_seq, torch.tensor([1.0]*N)), obs=data)

    def guide(self,data):
        N=100
        a1 = pyro.param('a1', lambda: torch.tensor(self.init_a1,dtype=torch.double),constraint=positive)
        b1 = pyro.param('b1', lambda: torch.tensor(self.init_b1,dtype=torch.double),constraint=positive)
        a2 = pyro.param('a2', lambda: torch.tensor(self.init_a2,dtype=torch.double),constraint=positive)
        b2 = pyro.param('b2', lambda: torch.tensor(self.init_b2,dtype=torch.double),constraint=positive)
        p = pyro.param('p',lambda: torch.tensor([0.5]*(N-1),dtype=torch.double),constraint=real_vector)
        lambda1 = pyro.sample("lambda1", Gamma(a1,b1))
        lambda2 = pyro.sample("lambda2", Gamma(a2,b2))
        tau = pyro.sample("tau", PoissonBinomial(torch.sigmoid(p),torch.double))
        return {"lambda1": lambda1, "lambda2": lambda2, "tau": tau}
