# Mirrored Steering Audit: `exp83_force_steering_guidance_v3_seed42_point_`

## Overall

- mirrored pairs analyzed: 3
- pairs where first-step env target flips sign: 3/3
- pairs where first-step raw steer flips sign: 0/3
- pairs where early-window env target flips sign: 3/3
- pairs where early-window raw steer flips sign: 0/3
- pairs where env target flips sign: 1/3
- pairs where policy raw steer flips sign: 0/3
- pairs where raw steer matches target sign in both cases: 0/3

## Pairs

- `|y|=0.000, |yaw|=4.0`: first target flip=True, first raw flip=False, early target flip=True, early raw flip=False, full target flip=True, full raw flip=False
  case A `(y=+0.000, yaw=-4.0)`: first target=- (-0.274), first raw=+ (+0.264), early target=- (-0.407), early raw=+ (+0.317), full target=- (-0.848), full raw=+ (+0.339), normal_success=False, zero_success=False, wrong_sign_frac=1.000
  case B `(y=+0.000, yaw=+4.0)`: first target=+ (+0.272), first raw=+ (+0.191), early target=+ (+0.388), early raw=+ (+0.198), full target=+ (+0.077), full raw=+ (+0.261), normal_success=True, zero_success=False, wrong_sign_frac=0.358

- `|y|=0.100, |yaw|=0.0`: first target flip=True, first raw flip=False, early target flip=True, early raw flip=False, full target flip=False, full raw flip=False
  case A `(y=-0.100, yaw=+0.0)`: first target=- (-0.167), first raw=+ (+0.255), early target=- (-0.293), early raw=+ (+0.309), full target=- (-0.576), full raw=+ (+0.336), normal_success=False, zero_success=False, wrong_sign_frac=1.000
  case B `(y=+0.100, yaw=+0.0)`: first target=+ (+0.164), first raw=+ (+0.202), early target=+ (+0.276), early raw=+ (+0.211), full target=- (-0.487), full raw=+ (+0.314), normal_success=False, zero_success=True, wrong_sign_frac=0.946

- `|y|=0.100, |yaw|=4.0`: first target flip=True, first raw flip=False, early target flip=True, early raw flip=False, full target flip=False, full raw flip=False
  case A `(y=+0.100, yaw=-4.0)`: first target=- (-0.104), first raw=+ (+0.238), early target=- (-0.098), early raw=+ (+0.270), full target=- (-0.582), full raw=+ (+0.338), normal_success=False, zero_success=True, wrong_sign_frac=1.000
  case B `(y=-0.100, yaw=+4.0)`: first target=+ (+0.101), first raw=+ (+0.221), early target=+ (+0.073), early raw=+ (+0.264), full target=- (-0.393), full raw=+ (+0.298), normal_success=False, zero_success=True, wrong_sign_frac=0.937
