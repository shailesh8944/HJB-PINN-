"""Static diagnostics saved during training: the head-on danger/safe map
(with the V=0 barrier contour) plus the training-loss curve."""
import numpy as np
import torch

from config import IX, IY, IPSI, IUE, IUI, R_COLLIDE, T_HORIZON


def save_map(net, path, device, approach=0.8, ng=160, hist=None,
             time_remaining=T_HORIZON):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = -15.0, 15.0
    xs = np.linspace(lo, hi, ng); ys = np.linspace(lo, hi, ng)
    XX, YY = np.meshgrid(xs, ys)
    Z = np.zeros((ng*ng, 9), np.float32)
    Z[:, IX] = XX.ravel(); Z[:, IY] = YY.ravel()
    Z[:, IPSI] = np.pi                     # head-on
    Z[:, IUE] = approach; Z[:, IUI] = approach
    if not 0.0 <= time_remaining <= T_HORIZON:
        raise ValueError("time_remaining must lie in [0, T_HORIZON]")
    tt = np.full((ng*ng, 1), -time_remaining, np.float32)
    inp = torch.tensor(np.hstack([Z, tt]), device=device)
    with torch.no_grad():
        V = net(inp).float().cpu().numpy().reshape(ng, ng)
    fig, ax = plt.subplots(1, 2 if hist else 1, figsize=(13, 5.3) if hist else (6.5, 5.3))
    a0 = ax[0] if hist else ax
    a0.contourf(YY, XX, (V <= 0).astype(float), levels=[-.5,.5,1.5], colors=["#c7e9c0","#fb6a4a"])
    a0.contour(YY, XX, V, levels=[0.0], colors="k", linewidths=2)
    th = np.linspace(0, 2*np.pi, 100)
    a0.plot(R_COLLIDE*np.sin(th), R_COLLIDE*np.cos(th), "b--", lw=1.2, label=f"collision R={R_COLLIDE:.0f} m")
    a0.plot(0, 0, "ks", ms=7, label="pursuer"); a0.axis("equal")
    a0.set_xlabel("Y rel [m]"); a0.set_ylabel("X rel [m]")
    a0.set_title(f"Head-on, time remaining={time_remaining:.1f}s: danger (red) / safe (green)")
    a0.legend(loc="upper right")
    if hist:
        its = [h[0] for h in hist]; ls = [max(h[1],1e-12) for h in hist]
        ax[1].semilogy(its, ls); ax[1].set_xlabel("iter"); ax[1].set_ylabel("HJI residual (log)")
        ax[1].grid(True, which="both", alpha=0.3); ax[1].set_title("Training loss")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
