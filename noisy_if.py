"""Interspike interval statistics of a noisy integrate-and-fire neuron.

    du/dt = -alpha*u + Ibar + noise,   reset u -> 0 at u = 1

With subthreshold drive (Ibar below threshold) the neuron cannot fire
without noise at all, so every spike is noise-driven and the ISI
distribution is a first-passage density: sharply peaked with a long right
tail, nothing like the delta a deterministic model gives. Raising sigma
raises the firing rate even though the mean input has not changed, which
is the point of the model.

Two ways to collect ISIs, both here:
  - one long simulation, taking differences of spike times
  - many short first-passage runs, each stopped at threshold

The second gives a fixed sample size, so histograms are easier to compare
across parameters.

Uses lif_sim from lif_sim.py for the trace, rather than repeating the
integration loop as Noisy_IF.ipynb did twice more.

Fixed from the original: the first-passage loop wrote isi[j-1] instead of
isi[j], so entry 0 was never assigned and stayed exactly zero while entry
nsim-1 was written twice. One spurious zero interval in every sample
pulled the mean down and inflated the variance. The notebook also
demonstrated noise with rnd.rand() (uniform) where rnd.randn() (normal)
was meant.

Regression check: with Ibar = 0.95, sigma = 0.1 the two methods should
agree on the mean ISI to within sampling error. No ISI should be zero; if
one is, the off-by-one is back.
"""

import numpy as np
import matplotlib.pyplot as plt

def noisy_if(mu=0.95, sigma_sq=0.1, uth=1.0, dt=0.01, t_end=300, seed=0):
    rng = np.random.default_rng(seed)
    t_vals = np.linspace(0, t_end, int(t_end/dt)) 
    u_vals = np.zeros(len(t_vals)
    for j in range(len(t_vals)):
        u = u_vals[j]
        if u < uth:
            u += dt * (Ibar) + np.sqrt(2 * dt*sigma_sq) * rng.standard_normal()
            t += dt
        else:
            u = 0
        u_vals[j+1] = u
    return t_vals, u_vals


t_test, u_test = noisy_if(mu = 1, sigma_sq = .25, uth = 1, dt = .01, t_end = 300, seed = 10)
plt.hist(u_test)
plt.show()

plt.plot(t_test, u_test)
plt.show()

