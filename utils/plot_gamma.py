import matplotlib.pyplot as plt
from scipy.stats import gamma
import numpy as np
def plot_gamma(step,a1,b1,a2,b2):
    fig, ax=plt.subplots(1,1)
    x = np.linspace(max(gamma.ppf(0.0,a1,b1),gamma.ppf(0.0,a2,b2)),
                    max(gamma.ppf(0.999,a1,b1),gamma.ppf(0.999,a2,b2)),100)
    ax.set_xlim(0.0,20.0)
    ax.set_ylim(0.0,1.0)
    ax.plot(x, gamma.pdf(x,a1, b1))
    ax.plot(x, gamma.pdf(x,a2, b2))
    fig.savefig("results/results02/"+str(step)+".png")