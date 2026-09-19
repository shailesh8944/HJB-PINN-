# Sookshma ASV — 3-DOF Dynamics Simulator

Standard **Fossen 3-DOF manoeuvring model** for the twin-thruster ASV *Sookshma*,
driven by the identified damping polynomial (θ coefficients) you supplied. The
structure matches equations (1)–(6) of the project HJB paper
*"HJB Reachability Formulation for a Twin-Thruster Surface Vessel Pursuing an
APF-Driven Evader."*

## The model

Body-frame velocity `ν = (u, v, r)` (surge, sway, yaw-rate). With generalised
force `τ = (τ_u, 0, τ_r)` (the two parallel thrusters give surge force and yaw
moment; sway force is zero):

```
M ν̇ + C(ν)ν + D(ν)ν = τ
ν̇ = M⁻¹ [ τ − C(ν)ν − D(ν)ν ]
```

Kinematics (body → NED):

```
ẋ   =  u cos ψ − v sin ψ
ẏ   =  u sin ψ + v cos ψ
ψ̇  =  r
```

**Mass matrix** is coupled in sway-yaw:
`M = [[M11,0,0],[0,M22,M23],[0,M32,M33]]`, with current values
`[[21.72,0,0],[0,29.214,0.7626],[0,0.7614,2.8632]]`.

**Coriolis** (rigid-body + added mass form from the paper):

```
C(ν)ν = [ −(M22·v+M23·r)r,
           M11·u·r,
           (M22·v+M23·r)u−M11·u·v ]
```

**Damping** `D(ν)ν = [X_D, Y_D, N_D]` is the identified polynomial, replacing the
paper's simple diagonal `diag(d₁₁+d₁₁⁽²⁾|u|, …)`. Basis functions and both
coefficient sets (`θ_anneal`, `θ_boot`) live in `config/vessel_config.yaml`.

### Thrusters

Two symmetric thrusters, arm `d = 0.21 m` from the CoG. Allocation follows the
paper's `B = [[1, 1], [−d, d]]`:

```
τ_u = F_port + F_stbd
τ_r = d · (F_stbd − F_port)
```

Body frame: x forward, y starboard, z down; **+r = bow to starboard**. Forces
are limited to the operating band **0 – 1.82 kgf** (1 kgf = 9.80665 N →
0 – 17.85 N). PWM↔force is looked up from the Blue Robotics **T200 @ 16 V**
curve (`config/thruster_T200_16V.csv`): 0 kgf ≈ PWM 1500 µs, 1.82 kgf ≈ PWM
1700 µs (RPM ≈ 2129).

## Layout

```
config/
  vessel_config.yaml       <-- the "dynamic input file": all params + coefficients
  thruster_T200_16V.csv    <-- T200 16 V performance curve (from your upload)
src/
  dynamics.py              <-- Fossen 3-DOF kinematics + kinetics
  thruster.py              <-- PWM/force lookup, clamping, thrust allocation
  simulate.py              <-- RK4 integrator, open-loop demo run, CSV + plot
  verify.py                <-- automated correctness checks (see below)
  controller.py            <-- LOS guidance + heading-PD / speed-PI + allocation
  track_waypoints.py       <-- closed-loop waypoint mission, track + thruster plots
outputs/
  trajectory.csv / .png    <-- results of the last run
```

## Run

```bash
pip install numpy pyyaml matplotlib
python src/simulate.py                    # theta_anneal, Coriolis on (defaults)
python src/simulate.py --set theta_boot   # switch coefficient set
python src/simulate.py --no-coriolis      # drop C(ν) term
```

Everything is data-driven from `config/vessel_config.yaml` — edit values there,
no numbers are hard-coded in the Python. Change the mass, inertia, thruster
limits, the active coefficient set, or the demo command schedule from that file.

## Waypoint tracking

```bash
python src/track_waypoints.py                 # default square mission
python src/track_waypoints.py --set theta_boot
```

A line-of-sight guidance law steers toward each waypoint; a proportional-
derivative (PD) heading controller sets the yaw moment and a proportional-
integral (PI) speed controller sets the surge force. The desired forces are
allocated back to the two thrusters and clamped to the 0-1.82 kgf band, so
saturation is modelled honestly. Waypoints, cruise speed, acceptance radius and
all gains live in `config/vessel_config.yaml -> mission`.

Outputs: `outputs/waypoint_track.png` (track, cross-track error, heading, speed)
and `outputs/waypoint_thrusters.png` (per-thruster force, PWM command, and
commanded-vs-applied surge force and yaw moment).

**What the default run shows.** The vessel closes a 15 m square, all four
waypoints reached, root-mean-square cross-track error ~0.23 m. The key physical
finding is in the thruster plot: at every corner the controller *asks* for about
8.7 N.m of yaw moment, but the thrusters can only deliver about 3.6 N.m
(= arm x max force x g = 0.20 x 1.82 x 9.80665). Yaw authority saturates on each
turn, which is why the cross-track error spikes at corners before recovering.
That is a design limit of the 0.20 m thruster arm and 1.82 kgf limit, not a
controller-tuning problem: wider spacing or more thrust would be needed for
tighter turns.

## How do we know the dynamics are correct?

"Correct" has three separate meanings. Only the first two can be settled by code;
the third needs your measured data.

**Level 1 — Implementation faithfulness (does the code compute the equations?).**
Run `python src/verify.py`. It runs formal checks, each independent of the
coefficient values:

- *Equilibrium*: zero velocity + zero force gives exactly zero acceleration.
- *Coriolis is workless*: `νᵀ C(ν) ν = 0` for random states (the Coriolis matrix
  must be skew-symmetric so it neither adds nor removes energy). Confirmed to ~1e-14.
- *Port-starboard symmetry*: symmetric thrust from rest keeps sway and yaw
  exactly zero for all time — a structural test of both the coefficient basis and
  the thrust allocation.
- *Integrator convergence*: the Runge-Kutta 4th-order (RK4) integrator is checked
  against step-halving (error should fall ~16x per halving) and against SciPy's
  reference integrator at tolerance 1e-10.

**Level 2 — Physical plausibility (do the numbers behave like a real vessel?).**
The verifier also flags:

- *Dissipativity*: the damping must remove energy everywhere, i.e. the dissipated
  power `νᵀ D(ν) ν ≥ 0` across the operating envelope. Any point where it goes
  negative is "anti-damping" — the model injecting energy — and is reported.
- *Principal damping signs*: linear surge, sway and yaw damping (`Dx_u`, `Dy_v`,
  `Dn_r`) must be positive.
- *Free decay*: with no thrust, kinetic energy must not increase.
- *Coefficient-set robustness*: it runs the same 10-second turn with `θ_boot` and
  `θ_anneal` and reports how far apart they end up. A large gap means the
  identification is not well constrained by the data.

**Level 3 — Model validity (do the coefficients match the actual Sookshma?).**
No amount of code proves this. You need to validate against experimental
trajectories that were NOT used to fit the coefficients:

1. Hold out a portion of your logged runs (train/test split).
2. *One-step check*: feed measured `(u, v, r)` and thrust into `nu_dot()` and
   compare the predicted accelerations to the measured `(u̇, v̇, ṙ)` — report
   root-mean-square error (RMSE) and normalised RMSE per axis.
3. *Multi-step check* (the hard one): from a measured initial state, integrate
   the model forward with the measured thrust log and overlay the predicted track
   on the real GPS/heading log. Multi-step error grows fast if the model is off.
4. Compare standard manoeuvres to reality: turning-circle radius at fixed
   differential thrust, and a zig-zag (Kempf) manoeuvre.
5. Sanity-check dominant coefficients against a first-principles or
   computational-fluid-dynamics (CFD) estimate.

## Assumptions & things to set (flagged)

1. `Izz = 2.44 kg·m²` is the current SolidWorks CAD value at the CoG.
2. Added mass is dimensionalized from the reference derivatives and includes the
   sway-yaw cross terms shown above.
3. The identified polynomial represents `D(ν)ν` only, so the separate
   skew-symmetric Coriolis term `C(ν)ν` is enabled.
4. **Damping sign** is `−1` (`D(ν)ν` subtracted, per Fossen). If your fit
   produced the signed RHS force directly, set `model.damping_sign: +1`.
5. The T200 curve is the **16 V** branch, as requested. Forces above 1.82 kgf
   exist in the data but are clamped out by the operating band.
