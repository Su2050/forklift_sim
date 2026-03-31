# Reference Trajectory External Review Package

Date: 2026-03-31

## What This Package Is About

We are debugging the reference trajectory used in a forklift pallet-insert RL task.

Current Stage1 setting:

- near-field reset only
- root x in `[-3.60, -3.45]`
- root y in `[-0.15, 0.15]`
- root yaw in `[-6 deg, 6 deg]`
- physical minimum turn radius fixed at `R_min = 2.34 m`
- reference trajectory terminal pose is still fairly hard/exact

The core problem is:

- `rs_exact` / `rs_forward_preferred` are more kinematically grounded, but in the current near-field geometry they produce trajectories that are mathematically valid yet poor as RL teaching trajectories
- current default `root_path_first` is smoother for training, but it is not truly forklift-kinematic-feasible

So right now we do not yet have a trajectory family that is both:

1. kinematically plausible for the forklift
2. suitable as the main teaching/reference trajectory for the current Stage1 near-field RL training

## Confirmed Findings

### 1. RS is not "implementation-broken", but it is not suitable as the main Stage1 teacher

- `rs_exact` can produce legal Reeds-Shepp paths, but in the current near-field setting they often fold back / knot / over-correct locally
- `rs_forward_preferred` currently fails to select usable RS candidates under reasonable thresholds
- offline scan shows that to start accepting RS in this near-field setting, `max_extra_length_m` would need to be relaxed to about `13-14 m`, which defeats the intended meaning of "forward-preferred"

Interpretation:

- this is not just a small threshold-tuning problem
- it looks like a structural geometry mismatch between:
  - near-field start states
  - hard terminal pose
  - large physical `R_min`

### 2. `root_path_first` is training-friendly but not physically valid enough

- current default trajectory is a root-space cubic plus final straight insertion
- it is smoother and easier to use as a reward/reference guide in Stage1
- but it does not enforce wheelbase, steering bound, minimum turn radius, or full vehicle kinematics

A concrete offending case is included in this package:

- `artifacts/root_path_first/c05_xm3p600_ym0p150_yawp6p000_20260331_164729.png`

For that case:

- reported `root_kappa_max = 5.438 1/m`
- with `R_min = 2.34 m`, physical curvature limit should be about `1 / 2.34 = 0.427 1/m`
- equivalent turn radius from that case is about `0.184 m`

So the concern is valid: `root_path_first` is not just visually suspicious, it really violates the intended kinematic bound.

## Main Question For External Review

Given the current Stage1 near-field setup, what is the best next direction?

Options we are considering:

1. stop treating the current trajectory as a true vehicle-feasible planner, and instead use only a local alignment / final-insert guide for reward shaping
2. design a new curvature-bounded trajectory family, such as:
   - `arc + straight`
   - `arc + arc + straight`
   - another explicitly curvature-limited construction
3. conclude that current Stage1 goal geometry itself is the issue, and change:
   - terminal goal definition
   - pre-alignment goal
   - curriculum distance

The most useful external feedback would be:

- whether the current diagnosis is sound
- whether a curvature-bounded non-RS family is the right next step
- whether the current near-field task is simply too short / too hard-pose-constrained for exact pose-to-pose planning to be a good reward teacher

## Package Contents

- `code/`
  - current task config
  - current trajectory-building code
  - current visualization script
- `docs/`
  - RS feasibility takeaway
  - forward-preferred extra-length probe
  - scope audit of current reference trajectory
  - RS source audit / tooling notes
- `artifacts/`
  - current `root_path_first` overlay, manifest, and representative bad case
  - RS exact overlay and manifest
  - forward-preferred RS manifests / overlays for the threshold probe
