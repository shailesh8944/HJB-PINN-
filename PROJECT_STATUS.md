# Sookshma HJI Reachability — Project Status

Twin-thruster autonomous surface vessel (ASV) *Sookshma*, pursuit-evasion
collision reachability solved as a two-player Hamilton-Jacobi-Isaacs (HJI)
differential game with a physics-informed neural network (PINN).

---

## 1. Objective

For two identical Sookshma vessels approaching head-on, compute the **danger
region** (relative states from which a collision within a 1 metre radius is
unavoidable) and the **safe region** (states from which the evader can guarantee
avoidance), by solving the backward reachable tube of the collision set.

- Pursuer = adversary, tries to force the separation to ≤ 1 m (minimiser).
- Evader = ego vessel, tries to keep separation > 1 m (maximiser).
- Danger = { V ≤ 0 }, Safe = { V > 0 }, barrier = { V = 0 }.

---

## 2. The model we locked (verified in chat)

### Rigid body and mass
- Mass m = 20 kg; yaw inertia I_z = 2.44 kg·m² (SolidWorks CAD at CoG).
- Added mass supplied as **non-dimensional** hydrodynamic derivatives (from a
  reference vessel's `hyd.yml`), redimensionalized by the prime (bis) system at
  ρ = 1000 kg/m³, L = 1.0 m: X_u̇ = −1.72 kg, Y_v̇ = −9.214 kg,
  Y_ṙ = −0.7626 kg·m, N_v̇ = −0.7614 kg·m, N_ṙ = −0.4232 kg·m².
  Added mass is a pure inertia — it carries **no** velocity dependence (that lives
  in the Coriolis and damping terms).
- Effective coupled mass matrix:
  **M = [[21.72,0,0],[0,29.214,0.7626],[0,0.7614,2.8632]]**.

### Damping
- Identified 34-term polynomial D(ν)ν = [X_D, Y_D, N_D], set **θ_anneal**
  (‖θ‖ = 240.88), dimensional, identical for both vessels, subtracted
  (damping_sign = −1). Rigid-body + added-mass Coriolis C(ν)ν kept.

### Thrusters (control)
- Two forward-only thrusters, arm d = 0.21 m, F_max = 1.82 kgf = 17.85 N
  (from the T200 16 V bollard curve). Allocation τ_u = F_p + F_s,
  τ_r = d(F_s − F_p).
- Actuator set modelled as the largest inscribed **ellipse** of the thrust
  diamond, centre (F_max, 0), semi-axes α_u = F_max/√2, α_r = d·F_max/√2. Held
  **constant** (speed-independent) for this first training.
- Deferred: the speed-dependent propeller model T = ρD⁴K_T·n² with the linear
  K_T(J) open-water curve. Measured K_T0 ≈ 0.43 from the T200 data; still need J₀
  (advance ratio where K_T = 0) to enable it later.

### Game (nine-state HJI)
- State ζ = (X, Y, ψ_rel, u_e, v_e, u_i, v_i, r_e, r_i) ∈ ℝ⁹.
- Control-affine in both players: ζ̇ = a₀(ζ) + G_i u_i + G_e u_e.
- Collision cost ℓ(ζ) = √(X² + Y²) − R, R = 1 m. Horizon T = 20 s.
- Isaacs Hamiltonian in closed form via the ellipse support functions;
  saddle-point controls closed-form. Value V solves the HJI variational
  inequality min{ ∂V/∂t + H, ℓ − V } = 0, V(ζ,0) = ℓ(ζ).

### Neural network
- SIREN (sinusoidal) network with heading encoded as (cos ψ, sin ψ) so the value
  is periodic by construction.
- Direct SIREN value Vθ(ζ,t), trained as in the cited DeepReach paper: terminal
  L1 pretraining of Vθ(ζ,0)=ℓ(ζ), then uniform-state HJI-VI curriculum training
  with the terminal and PDE losses together.

---

## 3. What has been verified so far

- **3-DOF dynamics (implementation):** equilibrium, Coriolis is workless
  (νᵀC(ν)ν = 0), port-starboard symmetry (symmetric thrust keeps sway/yaw exactly
  zero), fourth-order Runge-Kutta convergence vs a reference integrator, and
  energy dissipation under free decay — all pass.
- **Isaacs game math:** closed-form Hamiltonian equals brute-force min-max
  (~1e-5), min-max = max-min so the game value exists (gap ~1e-15), and the
  saddle-point controls are realisable within the thruster band — all pass.
- **Network:** autograd gradients match finite differences (~1e-11); heading
  periodicity holds by construction.

**Open model-quality flags (not bugs, but they affect the barrier):** θ_anneal has
a small anti-damping region at high combined sway/yaw, weak linear yaw damping,
and θ_boot vs θ_anneal diverge strongly — i.e. the identification is not tightly
constrained. These trace to the coefficients, not the solver.

---

## 4. Code and repository

Everything lives in the git repo (`D:\HJB PINN` on Windows →
`/mnt/newvolume/HJB_PINN` on the Ubuntu training box).

- `train_overnight.py` — **self-contained** HJI trainer (all physics baked in with
  the locked numbers; needs only torch/numpy/matplotlib). Use this for the run.
- `src/train_hji.py` — config-driven HJI trainer (checkpoint/resume).
- `src/hji_torch.py`, `src/hjb_torch.py`, `src/pinn.py` — game physics + network.
- `config/vessel_config.yaml` — the locked parameters (I_z = 2.44, dimensional
  coupled added mass, Coriolis enabled, θ_anneal damping).
- `config/game_config.yaml` — collision radius 1 m, horizon 20 s.
- 3-DOF simulation + checks: `src/dynamics.py`, `src/verify.py`, `src/simulate.py`,
  `src/thruster.py`, `src/controller.py`, `src/track_waypoints.py`.

---

## 5. Next steps

1. **Push (Windows) → pull (Ubuntu) → train on the RTX A4000.**
   `nohup python3 train_overnight.py --device cuda --iters 150000 --hidden 512
   --layers 3 --batch 65536 --lr 2e-5 --warmup 10000 --save-every 5000
   --out out_hji > train.log 2>&1 &`
   It checkpoints and refreshes `out_hji/danger_safe.png` every 5000 iterations.
2. **Watch convergence** overnight (loss falling and stabilising, barrier settling).
   If still rough by morning: more iterations (`--iters 300000 --resume`) or a
   wider network (`--hidden 768`).
3. **Post-training analysis:** render the barrier at several approach speeds and
   relative headings (not just head-on) to map how the safe region changes.
4. **Verify correctness** — see section 6.
5. **Refine inputs later:** measure I_z with a bifilar swing test; add the
   speed-dependent thruster model (needs J₀); consider a cleaner damping set given
   the θ_anneal flags.

---

## 6. How we verify the results are correct

Correctness has layers; each is a distinct check.

### (a) Implementation faithfulness — done, automated
The code computes the intended equations: the 3-DOF checks in `verify.py` and the
Isaacs checks (closed-form vs brute-force min-max, existence of the game value,
control realisability) all pass, and the network's autograd gradients match finite
differences. `train_overnight.py` re-runs the Isaacs checks at the start of every
run, so a bad parameter shows up immediately, not after hours.

### (b) Training convergence — check after the run
- The HJI residual loss should fall and stabilise, not oscillate or diverge.
- The danger/safe barrier should stop moving between successive checkpoints
  (compare consecutive `danger_safe.png`). If it is still shifting, it has not
  converged — train longer.
- The terminal condition V(ζ,0) = ℓ(ζ) is exact by the network construction, so
  any residual error is purely the interior HJI equation.

### (c) Reachability sanity checks — physical, cheap
- **Shrink-to-disc:** as the horizon → 0, the danger set must collapse to just the
  1 m collision disc. Evaluate V at small |t| and confirm.
- **Monotonic growth:** the danger tube must only grow (never shrink) as the
  horizon increases — it is a tube.
- **Symmetry:** for the symmetric head-on setup, the barrier must be symmetric
  under Y → −Y (port/starboard). Any asymmetry beyond the known weak-yaw-damping
  effect signals an error.
- **Direction:** a faster head-on closing speed should enlarge the danger region;
  an off-axis or diverging geometry should be safer. Confirm the trend.

### (d) Closed-loop rollout — the gold standard
Take many initial states spanning the barrier. From each, integrate the full game
dynamics with **both** players using the saddle-point controls read from the
trained value gradient (pursuer minimising, evader maximising), and record the
actual minimum separation over the horizon. The set of states whose realised
minimum separation ≤ 1 m must coincide with the predicted danger set { V ≤ 0 }.
Agreement here is the strongest evidence the value function is correct; a
Monte-Carlo sweep gives a quantitative false-safe / false-danger rate.

### (e) Independent reference — optional but convincing
Solve a reduced-dimension version of the game on a direct grid-based
Hamilton-Jacobi level-set solver (which converges reliably in low dimensions) and
compare its barrier to the network's on the matching slice. Close agreement
cross-validates the PINN against a method with no neural-network approximation.

### (f) Model validity against reality — the deepest, still to do
None of the above proves the *coefficients* match the real Sookshma; the barrier
is only as trustworthy as the dynamics. Validate the identified model against
logged sea-trial trajectories: a one-step check (feed measured u, v, r and thrust
into the acceleration function, compare to measured accelerations, report
root-mean-square error per axis) and a multi-step check (integrate forward from a
measured start with the real thrust log and overlay on the recorded track). Until
this is done, the danger/safe map is correct *for the assumed model*, with the
θ_anneal caveats noted in section 3.
