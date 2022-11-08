import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

def model(data):
    N = data.size(0)
    a1 = 1.0
    b1 = 1.0
    a2 = 1.0
    b2 = 1.0
    lambda1 = pyro.sample("lambda1", dist.Gamma(a1,b1))
    lambda2 = pyro.sample("lambda2", dist.Gamma(a2,b2))
    pi = torch.tensor([1.0]*(N-1))
    tau = pyro.sample("tau", dist.Categorical(torch.softmax(pi,0,dtype=torch.double)))
    z_seq_before = torch.tensor([lambda1 * i for i in range(tau+1)])
    z_seq_after = torch.tensor([lambda1 * (tau-1) +lambda2 * (i-tau) for i in range(tau+1,N)])
    z_seq = torch.concat([z_seq_before,z_seq_after])

    with pyro.plate("data", N):
        pyro.sample("obs", dist.Normal(z_seq, torch.tensor([1.0]*N)), obs=data)

def guide(data):
    N = data.size(0)
    a1 = pyro.param('a1', lambda: torch.tensor(1.0,dtype=torch.double),
        constraint=constraints.positive)
    b1 = pyro.param('b1', lambda: torch.tensor(0.1,dtype=torch.double),
        constraint=constraints.positive)
    a2 = pyro.param('a2', lambda: torch.tensor(1.0,dtype=torch.double),
        constraint=constraints.positive)
    b2 = pyro.param('b2', lambda: torch.tensor(0.1,dtype=torch.double),
        constraint=constraints.positive)
    pi = pyro.param('pi',lambda: torch.tensor([1.0]*99))
    lambda1 = pyro.sample("lambda1", dist.Gamma(a1,b1))
    lambda2 = pyro.sample("lambda2", dist.Gamma(a2,b2))
    tau = pyro.sample("tau", dist.Categorical(torch.softmax(pi,0,torch.double)))
    return {"lambda1": lambda1, "lambda2": lambda2, "tau": tau}
