import torch
import pyro
from  pyro.distributions import Gamma,Normal
from pyro.distributions.constraints import positive,real_vector
from customizedpyro.distributions.poibin import PoissonBinomial
# from customizedtorch.constrains import zero_to_one

class model2:
    def __init__(self,data,N,hp_a1,hp_b1,hp_a2,hp_b2,init_a1,init_b1,init_a2,init_b2):        
        self.data=data
        self.N=N
        self.hp_a1=hp_a1
        self.hp_b1=hp_b1
        self.hp_a2=hp_a2 
        self.hp_b2=hp_b2
        self.init_a1=init_a1
        self.init_b1=init_b1
        self.init_a2=init_a2 
        self.init_b2=init_b2

    def generator(self,data):   
        lambda1 = pyro.sample("lambda1", Normal(self.hp_a1,self.hp_b1))
        lambda2 = pyro.sample("lambda2", Normal(self.hp_a2,self.hp_b2))
        p = torch.tensor([0.0]*(self.N-1),dtype=torch.double)
        tau = pyro.sample("tau", PoissonBinomial(torch.sigmoid(p)))
        z_seq_before = torch.tensor([lambda1 * i for i in range(tau+1)])
        z_seq_after = torch.tensor([lambda1 * (tau-1) +lambda2 * (i-tau) for i in range(tau+1,self.N)])
        z_seq = torch.concat([z_seq_before,z_seq_after])
        with pyro.plate("data", self.N):
            pyro.sample("obs", Normal(z_seq, torch.tensor([1.0]*self.N)), obs=self.data)

    def inference(self,data):
        loc1 = pyro.param('loc1', lambda: torch.tensor(self.init_a1,dtype=torch.double))
        scale1 = pyro.param('scale1', lambda: torch.tensor(self.init_b1,dtype=torch.double),constraint=positive)
        loc2 = pyro.param('loc2', lambda: torch.tensor(self.init_a2,dtype=torch.double))
        scale2 = pyro.param('scale2', lambda: torch.tensor(self.init_b2,dtype=torch.double),constraint=positive)
        p = pyro.param('p',lambda: torch.tensor([0.5]*(self.N-1),dtype=torch.double),constraint=real_vector)
        lambda1 = pyro.sample("lambda1", Normal(loc1,scale1))
        lambda2 = pyro.sample("lambda2", Normal(loc2,scale2))
        tau = pyro.sample("tau", PoissonBinomial(torch.sigmoid(p),torch.double))
        return {"lambda1": lambda1, "lambda2": lambda2, "tau": tau}
