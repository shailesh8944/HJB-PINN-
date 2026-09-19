"""Direct sinusoidal value network used by the DeepReach paper.

The network predicts V_theta(z,t) directly.  Its terminal condition is learned
from the paper's terminal-value loss; it is not imposed by an ExactBC wrapper.
"""
import math
import torch
import torch.nn as nn

from config import IPSI


class Sine(nn.Module):
    def __init__(self, w0=1.0): super().__init__(); self.w0 = w0
    def forward(self, x): return torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(self, lo, hi, hidden=256, layers=3, w0=30.0, dtype=torch.float32):
        super().__init__()
        self.register_buffer("lo", torch.as_tensor(lo, dtype=dtype))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=dtype))
        din = len(lo) + 1                       # psi -> (cos,sin) adds one
        net = [nn.Linear(din, hidden), Sine(w0)]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), Sine(w0)]
        net += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*net).to(dtype)
        with torch.no_grad():
            lins = [m for m in self.net if isinstance(m, nn.Linear)]
            lins[0].weight.uniform_(-1.0/din, 1.0/din)
            for lin in lins[1:]:
                b = math.sqrt(6.0/lin.weight.shape[1]) / w0
                lin.weight.uniform_(-b, b)

    def _feat(self, x):
        xs = 2.0*(x - self.lo)/(self.hi - self.lo) - 1.0
        cols = []
        for i in range(x.shape[1]):
            if i == IPSI:
                cols += [torch.cos(x[:, i:i+1]), torch.sin(x[:, i:i+1])]
            else:
                cols.append(xs[:, i:i+1])
        return torch.cat(cols, 1)

    def forward(self, x): return self.net(self._feat(x))


def value_and_grads(net, z, t):
    inp = torch.cat([z, t.reshape(-1, 1)], 1).requires_grad_(True)
    V = net(inp).squeeze(1)
    g = torch.autograd.grad(V.sum(), inp, create_graph=True)[0]
    return V, g[:, 9], g[:, :9]
