# Mirrored Steering Audit: `exp83_force_steering_guidance_v3_seed42_point_`

## Overall

- mirrored pairs analyzed: 3
- pairs where env target flips sign: 1/3
- pairs where policy raw steer flips sign: 0/3
- pairs where raw steer matches target sign in both cases: 0/3

## Pairs

- `|y|=0.000, |yaw|=4.0`: target flip=True, raw flip=False, raw matches both=False
  case A `(y=+0.000, yaw=-4.0)`: target=- (-0.848), raw=+ (+0.339), normal_success=False, zero_success=False, wrong_sign_frac=1.000
  case B `(y=+0.000, yaw=+4.0)`: target=+ (+0.077), raw=+ (+0.261), normal_success=True, zero_success=False, wrong_sign_frac=0.358

- `|y|=0.100, |yaw|=0.0`: target flip=False, raw flip=False, raw matches both=False
  case A `(y=-0.100, yaw=+0.0)`: target=- (-0.576), raw=+ (+0.336), normal_success=False, zero_success=False, wrong_sign_frac=1.000
  case B `(y=+0.100, yaw=+0.0)`: target=- (-0.487), raw=+ (+0.314), normal_success=False, zero_success=True, wrong_sign_frac=0.946

- `|y|=0.100, |yaw|=4.0`: target flip=False, raw flip=False, raw matches both=False
  case A `(y=+0.100, yaw=-4.0)`: target=- (-0.582), raw=+ (+0.338), normal_success=False, zero_success=True, wrong_sign_frac=1.000
  case B `(y=-0.100, yaw=+4.0)`: target=- (-0.393), raw=+ (+0.298), normal_success=False, zero_success=True, wrong_sign_frac=0.937
