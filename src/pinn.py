"""
pinn.py — Sinusoidal (SIREN) value-function network and helpers for the HJB
reachability solver.

The network represents V_theta(zeta, t). It takes the RAW physical inputs
(zeta in R^9, t in R) so that torch autograd of the output with respect to those
raw inputs returns the physical dV/dt and grad_zeta V directly.

State representation (important):
  * The relative heading psi_rel (index 2 of zeta) is a PERIODIC coordinate:
    psi_rel and psi_rel + 2*pi are the same physical state. Feeding it as a raw
    scalar would let the network learn a false discontinuity across +/- pi. We
    therefore encode it on the unit circle as (cos psi_rel, sin psi_rel) INSIDE
    the network, so the input the network sees is periodic by construction. The
    dynamics state stays 9-dimensional; only the network featurisation changes.
    Autograd still returns dV/dpsi_rel correctly, via the chain rule through
    cos/sin.
  * All other coordinates are scaled linearly to [-1, 1] over the sampling
    domain [lo, hi]. (cos, sin) are already in [-1, 1] and are passed unscaled.

Raw input order (length 10):
  [X, Y, psi_rel, u_e, v_e, u_i, v_i, r_e, r_i, t]
Feature order into the first layer (length 11):
  [X, Y, cos psi_rel, sin psi_rel, u_e, v_e, u_i, v_i, r_e, r_i, t]  (all scaled
  except cos/sin)
"""

import numpy as np
import torch
import torch.nn as nn

ANGLE_IDX = 2          # index of psi_rel in the raw input vector


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SIREN(nn.Module):
    """Sinusoidal network with internal input scaling and periodic-heading encoding."""

    def __init__(self, lo, hi, hidden=256, layers=3, w0_first=30.0, w0=30.0,
                 dtype=torch.float32, angle_idx=ANGLE_IDX):
        super().__init__()
        self.register_buffer("lo", torch.as_tensor(lo, dtype=dtype))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=dtype))
        self.angle_idx = angle_idx
        din = len(lo) + 1          # one angular dim becomes two (cos, sin)

        net = [nn.Linear(din, hidden), Sine(w0_first)]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), Sine(w0)]
        net += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*net).to(dtype)
        self._siren_init(w0_first, w0)

    def _siren_init(self, w0_first, w0):
        lins = [m for m in self.net if isinstance(m, nn.Linear)]
        with torch.no_grad():
            first = lins[0]
            din = first.weight.shape[1]
            first.weight.uniform_(-1.0 / din, 1.0 / din)
            for lin in lins[1:]:
                n = lin.weight.shape[1]
                b = np.sqrt(6.0 / n) / w0
                lin.weight.uniform_(-b, b)

    def _featurise(self, x):
        """Raw input (B, D) -> features (B, D+1): scale non-angular dims to
        [-1, 1]; replace psi_rel by (cos, sin)."""
        xs = 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0     # scaled (angle col unused)
        cols = []
        for i in range(x.shape[1]):
            if i == self.angle_idx:
                cols.append(torch.cos(x[:, i:i + 1]))
                cols.append(torch.sin(x[:, i:i + 1]))
            else:
                cols.append(xs[:, i:i + 1])
        return torch.cat(cols, dim=1)

    def forward(self, x):
        return self.net(self._featurise(x))


class ExactBCValue(nn.Module):
    """Exact-boundary-condition value: V(zeta,t) = ell(zeta) + t * softplus(N(zeta,t)).

    This enforces two properties of the value function by construction, not by a
    soft penalty:
      * terminal condition  V(zeta, 0) = ell(zeta)  exactly (the t factor vanishes);
      * tube property        V(zeta, t) <= ell(zeta) for all t in [-T, 0]
        (t <= 0 and softplus >= 0 give a non-positive correction).
    softplus is unbounded above, so the value can drop arbitrarily far below ell as
    time-to-go grows, which the reachable tube requires. The base network N is the
    SIREN. terminal_fn maps zeta (B,9) -> ell (B,)."""

    def __init__(self, siren, terminal_fn):
        super().__init__()
        self.siren = siren
        self.terminal_fn = terminal_fn

    def forward(self, inp):
        zeta = inp[:, :9]
        t = inp[:, 9]
        ell = self.terminal_fn(zeta)
        N = self.siren(inp).squeeze(1)
        V = ell + t * torch.nn.functional.softplus(N)
        return V.unsqueeze(1)


def value_and_grads(net, zeta, t):
    """
    Return V (B,), dV/dt (B,), grad_zeta V (B,9) with create_graph=True so the
    residual can be back-propagated to the network parameters. The gradient with
    respect to psi_rel is obtained through the cos/sin encoding by the chain rule.
    """
    inp = torch.cat([zeta, t.reshape(-1, 1)], dim=1).requires_grad_(True)
    V = net(inp).squeeze(1)
    grads = torch.autograd.grad(V.sum(), inp, create_graph=True)[0]
    gradV_zeta = grads[:, :9]
    dVdt = grads[:, 9]
    return V, dVdt, gradV_zeta
