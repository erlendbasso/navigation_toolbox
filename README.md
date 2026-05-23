# navigation-toolbox

Small `no_std` Rust utilities for navigation math:

- WGS84 geodetic latitude/longitude/height to ECEF conversion
- ECEF to geodetic latitude/longitude/height conversion
- Analytic LLH-to-ECEF and ECEF/NED rotation Jacobians
- ECEF/NED direction cosine matrices, quaternion helper, and Earth-rate helpers
- NED transport and inertial rates
- ECEF and NED gravity
- ECEF gravity gradient for linearized filters
- SO(3)/SE(3) primitives for estimator and strapdown math

## Conventions

- Angles are radians.
- Distances and heights are meters.
- Velocities are meters per second.
- Accelerations are meters per second squared.
- NED vectors are ordered `[north, east, down]`.
- ECEF vectors are ordered `[x, y, z]`.
- SE(3) twist helpers use `[translation, rotation]` ordering.

The crate is currently `#![no_std]` and uses `nalgebra` with its `libm` feature.

## Math and Linearization Helpers

The crate includes small deterministic helpers commonly duplicated across
navigation filters:

- `skew`, `rotx`, `roty`, `rotz`, and `wrap_to_pi`
- `so3_exp`, `so3_left_jacobian`, `se3_exp`, and `ad_se3`
- `quat_update_from_rotation_vector` and `quat_to_mrp4x`
- `lat_lon_height_to_ecef_jacobian`
- `rot_ned_ecef_jacobian_lat` and `rot_ned_ecef_jacobian_lon`
- `earth_rate_ecef`, `earth_rate_ned`, and `omega_in_n`

These helpers are intentionally stateless. Full EKF engines, timestamp buffers,
sensor models, alignment routines, and coning/sculling accumulators should live
in higher-level crates that can own their runtime assumptions.

## Gravity Model

`gravity_ecef` computes apparent gravity in the rotating ECEF frame. The model is:

- central Newtonian gravity
- WGS84 `J2` correction
- centrifugal acceleration from Earth rotation

`gravity_ned` rotates that same ECEF gravity vector into local NED coordinates.
`grav_accel_ned` is a compatibility wrapper around `gravity_ned(lat, 0.0, height)`.
The model is axisymmetric, so longitude does not affect the result.

`gravity_gradient_ecef` returns the Jacobian of `gravity_ecef` with respect to
ECEF position:

```text
G[i, j] = d gravity_ecef_i / d r_ecef_j
```

The units are `1/s^2`. This is the position Jacobian typically needed when an
ECEF-position EKF or INS error-state model propagates velocity with
`gravity_ecef(r)`.

If an estimator needs pure gravitational attraction without the ECEF centrifugal
term, use a matching gravity model and matching Jacobian. The provided Jacobian
intentionally differentiates the exact acceleration returned by `gravity_ecef`.

## Singular Inputs

The API keeps simple return types and uses NaNs for invalid or singular inputs:

- `gravity_ecef` and `gravity_gradient_ecef` return NaNs at the ECEF origin.
- `omega_en_n` returns NaNs at exact NED poles and for invalid negative heights
  where local curvature radius plus height is non-positive.
- `comp_lat_lon_height` returns `(NaN, 0.0, NaN)` at the ECEF origin.
- At exact geographic poles, `comp_lat_lon_height` returns longitude `0.0` by
  convention.

## Example

```rust
use core::f64::consts::PI;
use navigation_toolbox::{
    comp_lat_lon_height, earth_rate_ned, gravity_ecef, gravity_gradient_ecef,
    gravity_ned, lat_lon_height_to_ecef, lat_lon_height_to_ecef_jacobian,
    rot_ned_ecef, rot_ned_ecef_jacobian_lat, so3_exp,
};
use nalgebra::Vector3;

let lat = 63.0 * PI / 180.0;
let lon = 10.0 * PI / 180.0;
let height = 50.0;

let r_ecef = lat_lon_height_to_ecef(lat, lon, height);
let (lat_back, lon_back, height_back) = comp_lat_lon_height(&r_ecef);

let g_ecef = gravity_ecef(&r_ecef);
let g_ned = gravity_ned(lat, lon, height);
let gravity_jacobian = gravity_gradient_ecef(&r_ecef);
let c_n_e = rot_ned_ecef(lat, lon);
let llh_jacobian = lat_lon_height_to_ecef_jacobian(lat, lon, height);
let c_n_e_lat_jacobian = rot_ned_ecef_jacobian_lat(lat, lon);
let omega_ie_n = earth_rate_ned(lat, lon);
let small_rotation = so3_exp(&Vector3::new(0.001, 0.0, 0.0));
```

## Verification

The test suite covers:

- LLH/ECEF round trips across equator, mid-latitude, near-pole, high-altitude,
  and longitude-wrap cases
- exact polar-axis ECEF-to-geodetic handling
- zero-vector singular behavior
- rotation inverse and orthonormality
- LLH-to-ECEF Jacobian against central finite differences
- ECEF/NED rotation Jacobians against central finite differences
- quaternion/rotation-matrix consistency
- SO(3), SE(3), quaternion-update, MRP, and Earth-rate helpers
- gravity sanity checks
- `gravity_gradient_ecef` against central finite differences at multiple
  locations
