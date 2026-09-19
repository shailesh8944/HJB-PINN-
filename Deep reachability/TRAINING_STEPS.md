# DeepReach-paper training procedure

The ASV dynamics and game are specific to this project. The learning procedure
follows *DeepReach: A Deep Learning Approach to High-Dimensional Reachability*
(Bansal and Tomlin, 2021), Section V and its 9D case study:

1. Use a direct three-layer sinusoidal (SIREN) network for \(V_\theta(z,t)\).
   State and time inputs are scaled to the network range.
2. Sample states uniformly over the configured nine-dimensional state box.
3. Pretrain only the terminal loss

   \[
   \lVert V_\theta(z,0)-\ell(z)\rVert_1.
   \]

   The paper's 9D case uses 60,000 iterations.
4. Train using the HJI-VI loss from Eq. (14):

   \[
   \lVert V_\theta(z_T,0)-\ell(z_T)\rVert_1+
   \lambda\left\lVert\min\{V_t+H,\ell-V\}\right\rVert_1.
   \]

   The trainer calibrates the fixed \(\lambda\) from the untrained network,
   before terminal pretraining, so the two Eq. (14) components begin at
   comparable scale as described in the paper. The 9D case uses 100,000
   curriculum iterations.
5. Expand the sampled interval linearly from terminal time to the full 20 s
   horizon over the curriculum. Every state sample remains uniform; there is no
   targeted encounter or collision-boundary sampling.

The paper uses width 512, three layers, Adam at \(10^{-4}\), and batches of
65,000 samples. The launcher preserves that effective update size with sixteen
uniform 4,096-sample GPU microbatches. Gradient accumulation gives the same
mean gradient as one 65,536-sample batch without exceeding the RTX 3050 6 GB
memory limit.

Run the paper-method configuration:

```text
powershell -ExecutionPolicy Bypass -File .\run_reachability.ps1
```
