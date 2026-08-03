# Powered Descent — landing a rocket with convex optimization

Interactive demo + reproducible solver from my **MSc thesis (Space Systems Engineering)**: powered-descent guidance (a Falcon-9 / Mars-lander style rocket landing) solved as a **second-order cone program** via **lossless convexification (LCvx)**.

### ▶ Live demo
**https://iseralx.github.io/powered_descendent/**

Watch the real optimal trajectory fly: bang-bang throttle, gimbal cone, live telemetry, a `t_f` slider that morphs the trajectory, and the `J*(t_f)` fuel curve.

---

## The idea in three lines

- **The problem.** Land the rocket burning the least fuel, touching the pad exactly — while the engine can't throttle below `ρ_min`. That floor makes the feasible thrust set an **annulus**, which is **non-convex**: no global-optimum guarantee, no bound on solve time.
- **The fix.** *Lossless convexification* (Açıkmeşe & Ploen, 2007): add a slack `σ`, require `‖T‖ ≤ σ` (a convex cone) and `ρ_min ≤ σ ≤ ρ_max`. Provably tight at the optimum (`σ* = ‖T*‖`). A log-mass change of variables removes the last nonlinearity → a plain **SOCP**.
- **The result.** The optimal burn is **bang-bang** (max → min → max). Reproduces the Mars-lander benchmark of Malyuta et al. (2022): **t_f\* = 75 s, 373.76 kg** of fuel, exact landing, LCvx gap ≈ 10⁻² N.

## Reproduce

```bash
pip install numpy scipy cvxpy    # Clarabel ships with cvxpy
python pdg_solve.py              # solves the SOCP, exports pdg_data.json
```

`pdg_solve.py` builds the continuous dynamics (with Mars rotation / Coriolis), discretizes with a zero-order hold via a matrix-exponential trick, and solves the LCvx SOCP with **CVXPY + Clarabel** over a grid of flight times. The demo (`index.html`) replays the exported trajectories — nothing is faked.

## Files

| File | What |
|---|---|
| `index.html` | The interactive demo (self-contained, real solver data embedded) |
| `pdg_solve.py` | The SOCP solver + data export |
| `pdg_data.json` | 19 optimal trajectories (t_f 73–91 s) + the J*(t_f) curve |

## References

- D. Malyuta et al., *Convex Optimization for Trajectory Generation*, 2022.
- B. Açıkmeşe & S. Ploen, *Convex Programming Approach to Powered Descent Guidance for Mars Landing*, JGCD, 2007.

---

Alejandro Soler Gonzálvez · [portfolio](https://iseralx.github.io/ISeralx/)
