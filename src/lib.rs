#![no_std]
#![allow(non_snake_case)]

//! Navigation helper functions for WGS84 geodesy, ECEF/NED frame transforms, and
//! a simple Earth gravity model.
//!
//! # Conventions
//!
//! - Angles are radians.
//! - Distances and heights are meters.
//! - Velocities are meters per second.
//! - NED vectors are ordered `[north, east, down]`.
//! - ECEF vectors use the conventional Earth-centered, Earth-fixed axes.
//!
//! # Gravity Model
//!
//! [`gravity_ecef`] computes apparent gravity in the rotating ECEF frame: central
//! Newtonian gravity, a WGS84 `J2` correction, and the centrifugal term from
//! Earth rotation. [`gravity_gradient_ecef`] is the position Jacobian of that
//! same function, so it includes the derivative of the centrifugal term too.
//!
//! The functions preserve the original simple API and return NaNs for invalid or
//! singular inputs such as the ECEF origin or exact NED poles.

use core::f64::consts::FRAC_PI_2;
use libm::{atan, atan2, cbrt, copysign, cos, fabs, sin, sqrt};

extern crate nalgebra as na;
use na::{Matrix3, Rotation3, UnitQuaternion, Vector3};

pub struct Ellipsoid {
    a: f64,
    f: f64,
}

const WGS84_A: f64 = 6_378_137.0;
const WGS84_F: f64 = 1.0 / 298.257_223_563;
const WGS84_B: f64 = WGS84_A * (1.0 - WGS84_F);
const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);
const WGS84_J2: f64 = 1.082627e-3;
const EARTH_MU: f64 = 3.986004418e14;
const EARTH_ROTATION_RATE: f64 = 7.292115e-5;
const NED_POLE_COS_EPS: f64 = 1.0e-12;

const WGS84: Ellipsoid = Ellipsoid {
    a: WGS84_A,
    f: WGS84_F,
};

fn nan_vector3() -> Vector3<f64> {
    Vector3::new(f64::NAN, f64::NAN, f64::NAN)
}

fn nan_matrix3() -> Matrix3<f64> {
    Matrix3::new(
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
    )
}

/// Converts geodetic latitude, longitude, and ellipsoidal height to ECEF.
///
/// `lat` and `lon` are in radians. `height` is height above the WGS84
/// ellipsoid in meters.
///
/// The returned vector is `[x, y, z]` in meters.
pub fn lat_lon_height_to_ecef(lat: f64, lon: f64, height: f64) -> Vector3<f64> {
    let sin_lat = sin(lat);
    let cos_lat = cos(lat);
    let sin_lon = sin(lon);
    let cos_lon = cos(lon);

    let radius_n = WGS84.a / sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);
    Vector3::new(
        (radius_n + height) * cos_lat * cos_lon,
        (radius_n + height) * cos_lat * sin_lon,
        (radius_n * (1.0 - WGS84_E2) + height) * sin_lat,
    )
}

/// Computes the transport rate of the NED frame relative to ECEF, resolved in NED.
///
/// `vel_eb_n` is ordered `[v_north, v_east, v_down]` in meters per second.
/// The returned vector is ordered `[omega_north, omega_east, omega_down]` in
/// radians per second.
///
/// NED is singular at the exact poles. This function returns a NaN vector for
/// polar latitudes or invalid heights where the local curvature radius plus
/// height is non-positive.
pub fn omega_en_n(lat: f64, height: f64, vel_eb_n: Vector3<f64>) -> Vector3<f64> {
    let sin_lat = sin(lat);
    let cos_lat = cos(lat);
    let denom = 1.0 - WGS84_E2 * sin_lat * sin_lat;
    let sqrt_denom = sqrt(denom);

    let radius_e = WGS84.a / sqrt_denom;
    let radius_n = WGS84.a * (1.0 - WGS84_E2) / (denom * sqrt_denom);
    let radius_e_height = radius_e + height;
    let radius_n_height = radius_n + height;

    if fabs(cos_lat) <= NED_POLE_COS_EPS || radius_e_height <= 0.0 || radius_n_height <= 0.0 {
        return nan_vector3();
    }

    Vector3::new(
        vel_eb_n[1] / radius_e_height,
        -vel_eb_n[0] / radius_n_height,
        -vel_eb_n[1] * sin_lat / (cos_lat * radius_e_height),
    )
}

/// Computes apparent gravity in ECEF at an ECEF position.
///
/// The input and output are ECEF vectors. The output has units of meters per
/// second squared.
///
/// This model includes central gravity, a WGS84 `J2` correction, and the
/// centrifugal term from Earth rotation. It returns a NaN vector at the ECEF
/// origin, where the model is singular.
pub fn gravity_ecef(r_eb_e: &Vector3<f64>) -> Vector3<f64> {
    let r_norm_squared = r_eb_e.norm_squared();
    if r_norm_squared == 0.0 {
        return nan_vector3();
    }

    let r_norm = sqrt(r_norm_squared);
    let r_norm_cubed = r_norm_squared * r_norm;
    let z_over_r = r_eb_e[2] / r_norm;
    let a_over_r = WGS84.a / r_norm;

    let z_scale = 5.0 * z_over_r * z_over_r;
    let gamma = -EARTH_MU / r_norm_cubed
        * (r_eb_e
            + 1.5
                * WGS84_J2
                * a_over_r
                * a_over_r
                * Vector3::new(
                    (1.0 - z_scale) * r_eb_e[0],
                    (1.0 - z_scale) * r_eb_e[1],
                    (3.0 - z_scale) * r_eb_e[2],
                ));

    gamma + EARTH_ROTATION_RATE * EARTH_ROTATION_RATE * Vector3::new(r_eb_e[0], r_eb_e[1], 0.0)
}

/// Computes the ECEF gravity gradient matrix.
///
/// Entry `(i, j)` is `d gravity_ecef_i / d r_ecef_j`, with units of `1/s^2`.
/// The derivative is for the exact model used by [`gravity_ecef`], including
/// the `J2` correction and centrifugal term. This is the position Jacobian often
/// used in ECEF-position EKF or INS error-state propagation.
///
/// The function returns a NaN matrix at the ECEF origin, where the gravity model
/// is singular.
pub fn gravity_gradient_ecef(r_eb_e: &Vector3<f64>) -> Matrix3<f64> {
    let x = r_eb_e[0];
    let y = r_eb_e[1];
    let z = r_eb_e[2];
    let r = [x, y, z];
    let r_norm_squared = r_eb_e.norm_squared();

    if r_norm_squared == 0.0 {
        return nan_matrix3();
    }

    let r_norm = sqrt(r_norm_squared);
    let inv_r3 = 1.0 / (r_norm_squared * r_norm);
    let inv_r5 = inv_r3 / r_norm_squared;
    let inv_r7 = inv_r5 / r_norm_squared;
    let inv_r9 = inv_r7 / r_norm_squared;
    let z_squared = z * z;
    let z_cubed = z_squared * z;
    let j2_scale = 1.5 * WGS84_J2 * WGS84.a * WGS84.a;
    let omega_squared = EARTH_ROTATION_RATE * EARTH_ROTATION_RATE;

    let mut gradient = Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            let delta = if i == j { 1.0 } else { 0.0 };
            gradient[(i, j)] = -EARTH_MU * (delta * inv_r3 - 3.0 * r[i] * r[j] * inv_r5);
        }
    }

    for i in 0..2 {
        for j in 0..3 {
            let delta = if i == j { 1.0 } else { 0.0 };
            let dz = if j == 2 { 1.0 } else { 0.0 };
            gradient[(i, j)] +=
                -EARTH_MU * j2_scale * (delta * inv_r5 - 5.0 * r[i] * r[j] * inv_r7)
                    + 5.0
                        * EARTH_MU
                        * j2_scale
                        * (delta * z_squared * inv_r7 + 2.0 * r[i] * z * dz * inv_r7
                            - 7.0 * r[i] * z_squared * r[j] * inv_r9);
        }
    }

    for j in 0..3 {
        let dz = if j == 2 { 1.0 } else { 0.0 };
        gradient[(2, j)] += -3.0 * EARTH_MU * j2_scale * (dz * inv_r5 - 5.0 * z * r[j] * inv_r7)
            + 5.0
                * EARTH_MU
                * j2_scale
                * (3.0 * z_squared * dz * inv_r7 - 7.0 * z_cubed * r[j] * inv_r9);
    }

    gradient[(0, 0)] += omega_squared;
    gradient[(1, 1)] += omega_squared;
    gradient
}

/// Computes apparent gravity at a geodetic position and resolves it in NED.
///
/// `lat` and `lon` are in radians. `height` is height above the WGS84 ellipsoid
/// in meters. The returned vector is ordered `[north, east, down]` in meters per
/// second squared.
pub fn gravity_ned(lat: f64, lon: f64, height: f64) -> Vector3<f64> {
    let r_eb_e = lat_lon_height_to_ecef(lat, lon, height);
    let g_e = gravity_ecef(&r_eb_e);

    rot_ned_ecef(lat, lon) * g_e
}

/// Returns a unit quaternion for the same ECEF-to-NED rotation as [`rot_ned_ecef`].
pub fn quat_ned_ecef(lat: f64, lon: f64) -> UnitQuaternion<f64> {
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rot_ned_ecef(lat, lon)))
}

/// Returns the direction cosine matrix that maps ECEF vectors into NED vectors.
///
/// For an ECEF-resolved vector `v_e`, `rot_ned_ecef(lat, lon) * v_e` gives the
/// same vector resolved in local NED coordinates.
pub fn rot_ned_ecef(lat: f64, lon: f64) -> Matrix3<f64> {
    let sin_lat = sin(lat);
    let cos_lat = cos(lat);
    let sin_lon = sin(lon);
    let cos_lon = cos(lon);

    Matrix3::new(
        -sin_lat * cos_lon,
        -sin_lat * sin_lon,
        cos_lat,
        -sin_lon,
        cos_lon,
        0.0,
        -cos_lat * cos_lon,
        -cos_lat * sin_lon,
        -sin_lat,
    )
}

/// Returns the direction cosine matrix that maps NED vectors into ECEF vectors.
pub fn rot_ecef_ned(lat: f64, lon: f64) -> Matrix3<f64> {
    rot_ned_ecef(lat, lon).transpose()
}

/// Computes apparent gravity resolved in NED at longitude zero.
///
/// This is a compatibility wrapper around [`gravity_ned`]. The current gravity
/// model is axisymmetric, so longitude does not affect the result. Use
/// [`gravity_ned`] directly when the local longitude is already available.
pub fn grav_accel_ned(lat: f64, height: f64) -> Vector3<f64> {
    gravity_ned(lat, 0.0, height)
}

/// Converts an ECEF position to geodetic latitude, longitude, and height.
///
/// Returns `(lat, lon, height)`, with angles in radians and height in meters.
/// At the exact north or south pole, longitude is defined as `0.0` by
/// convention. At the ECEF origin, latitude and height are returned as NaN.
pub fn comp_lat_lon_height(pos_ecef: &Vector3<f64>) -> (f64, f64, f64) {
    let beta = sqrt(pos_ecef[0] * pos_ecef[0] + pos_ecef[1] * pos_ecef[1]);
    let lon = atan2(pos_ecef[1], pos_ecef[0]);

    if beta == 0.0 {
        return if pos_ecef[2] > 0.0 {
            (FRAC_PI_2, 0.0, pos_ecef[2] - WGS84_B)
        } else if pos_ecef[2] < 0.0 {
            (-FRAC_PI_2, 0.0, -pos_ecef[2] - WGS84_B)
        } else {
            (f64::NAN, 0.0, f64::NAN)
        };
    }

    let k1 = (1.0 - WGS84.f) * fabs(pos_ecef[2]);
    let k2 = WGS84_E2 * WGS84.a;
    let E = (k1 - k2) / beta;
    let F = (k1 + k2) / beta;

    let P = 4.0 / 3.0 * (E * F + 1.0);

    let Q = 2.0 * (E * E - F * F);

    let D = P * P * P + Q * Q;
    let sqrt_D = sqrt(D);
    let V = cbrt(sqrt_D - Q) - cbrt(sqrt_D + Q);
    let G = 0.5 * (sqrt(E * E + V) + E);
    let T = sqrt(G * G + (F - V * G) / (2.0 * G - E)) - G;

    let z_sign = copysign(1.0, pos_ecef[2]);
    let lat = z_sign * atan((1.0 - T * T) / (2.0 * T * (1.0 - WGS84.f)));

    let height = (beta - WGS84.a * T) * cos(lat) + (pos_ecef[2] - z_sign * WGS84_B) * sin(lat);

    (lat, lon, height)
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::{FRAC_PI_2, PI};

    fn deg(value: f64) -> f64 {
        value * PI / 180.0
    }

    fn angle_diff(lhs: f64, rhs: f64) -> f64 {
        let mut diff = lhs - rhs;
        while diff > PI {
            diff -= 2.0 * PI;
        }
        while diff < -PI {
            diff += 2.0 * PI;
        }
        diff
    }

    fn assert_matrix_relative_eq(lhs: &Matrix3<f64>, rhs: &Matrix3<f64>, epsilon: f64) {
        for row in 0..3 {
            for col in 0..3 {
                assert_relative_eq!(lhs[(row, col)], rhs[(row, col)], epsilon = epsilon);
            }
        }
    }

    fn finite_difference_gravity_gradient(pos_ecef: &Vector3<f64>, step: f64) -> Matrix3<f64> {
        let mut gradient = Matrix3::zeros();

        for col in 0..3 {
            let mut pos_plus = *pos_ecef;
            let mut pos_minus = *pos_ecef;
            pos_plus[col] += step;
            pos_minus[col] -= step;

            let diff = (gravity_ecef(&pos_plus) - gravity_ecef(&pos_minus)) / (2.0 * step);
            for row in 0..3 {
                gradient[(row, col)] = diff[row];
            }
        }

        gradient
    }

    #[test]
    fn check_gravity() {
        let lat = deg(63.0);
        let lon = deg(10.0);
        let height = 50.0;

        let grav_1 = gravity_ned(lat, lon, height);
        let grav_2 = grav_accel_ned(lat, height);

        assert_relative_eq!(grav_1, Vector3::new(0.0, 0.0, 9.8213), epsilon = 0.0001);
        assert_relative_eq!(grav_2, Vector3::new(0.0, 0.0, 9.8213), epsilon = 0.0001);
    }

    #[test]
    fn test_lat_lon_h_to_ecef() {
        let lat = deg(63.0);
        let lon = deg(10.0);
        let height = 50.0;

        let r_eb_e = lat_lon_height_to_ecef(lat, lon, height);

        assert_relative_eq!(
            r_eb_e,
            Vector3::new(2.859253e6, 0.504163e6, 5.660022e6),
            epsilon = 1.0
        );
    }

    #[test]
    fn gravity_ecef_test() {
        let p_eb = Vector3::new(2.859253e6, 0.504163e6, 5.660022e6);

        let grav = gravity_ecef(&p_eb);

        assert_relative_eq!(
            grav,
            Vector3::new(-4.3910, -0.7742, -8.7509),
            epsilon = 0.0001
        );
    }

    #[test]
    fn gravity_ecef_returns_nan_for_zero_vector() {
        let grav = gravity_ecef(&Vector3::zeros());
        let gradient = gravity_gradient_ecef(&Vector3::zeros());

        assert!(grav[0].is_nan());
        assert!(grav[1].is_nan());
        assert!(grav[2].is_nan());
        for row in 0..3 {
            for col in 0..3 {
                assert!(gradient[(row, col)].is_nan());
            }
        }
    }

    #[test]
    fn gravity_gradient_matches_central_difference() {
        let p_eb = Vector3::new(2.859253e6, 0.504163e6, 5.660022e6);
        let analytic = gravity_gradient_ecef(&p_eb);
        let finite_difference = finite_difference_gravity_gradient(&p_eb, 10.0);

        assert_matrix_relative_eq(&analytic, &finite_difference, 1.0e-10);
        assert_matrix_relative_eq(&analytic, &analytic.transpose(), 1.0e-12);
    }

    #[test]
    fn omega_en_n_returns_nan_for_singular_inputs() {
        let at_pole = omega_en_n(FRAC_PI_2, 0.0, Vector3::new(0.0, 100.0, 0.0));
        let below_center = omega_en_n(0.0, -WGS84_A, Vector3::new(100.0, 100.0, 0.0));

        assert!(at_pole[0].is_nan());
        assert!(at_pole[1].is_nan());
        assert!(at_pole[2].is_nan());
        assert!(below_center[0].is_nan());
        assert!(below_center[1].is_nan());
        assert!(below_center[2].is_nan());
    }

    #[test]
    fn comp_lat_lon_height_test() {
        let p_eb = Vector3::new(2.812917e6, 0.5136468e6, 5.6821953e6);

        let (lat, lon, height) = comp_lat_lon_height(&p_eb);
        let pos_ecef = lat_lon_height_to_ecef(lat, lon, height);

        assert_relative_eq!(lat, deg(63.4414868604), epsilon = 1.0e-10);
        assert_relative_eq!(lon, deg(10.3483627043), epsilon = 1.0e-10);
        assert_relative_eq!(height, 50.6126, epsilon = 1.0e-4);
        assert_relative_eq!(pos_ecef, p_eb, epsilon = 1.0e-3);
    }

    #[test]
    fn comp_lat_lon_height_handles_polar_axis() {
        let north = Vector3::new(0.0, 0.0, WGS84_B + 25.0);
        let south = Vector3::new(0.0, 0.0, -WGS84_B - 30.0);

        let (north_lat, north_lon, north_height) = comp_lat_lon_height(&north);
        let (south_lat, south_lon, south_height) = comp_lat_lon_height(&south);

        assert_relative_eq!(north_lat, FRAC_PI_2, epsilon = 0.0);
        assert_relative_eq!(north_lon, 0.0, epsilon = 0.0);
        assert_relative_eq!(north_height, 25.0, epsilon = 1.0e-9);
        assert_relative_eq!(south_lat, -FRAC_PI_2, epsilon = 0.0);
        assert_relative_eq!(south_lon, 0.0, epsilon = 0.0);
        assert_relative_eq!(south_height, 30.0, epsilon = 1.0e-9);
    }

    #[test]
    fn comp_lat_lon_height_returns_nan_for_zero_vector() {
        let (lat, lon, height) = comp_lat_lon_height(&Vector3::zeros());

        assert!(lat.is_nan());
        assert_relative_eq!(lon, 0.0, epsilon = 0.0);
        assert!(height.is_nan());
    }

    #[test]
    fn lat_lon_height_round_trips_through_ecef() {
        let cases = [
            (0.0, 0.0, 0.0),
            (0.0, deg(90.0), 100.0),
            (deg(63.0), deg(10.0), 50.0),
            (deg(-45.0), deg(170.0), 1_000.0),
            (deg(89.999_999), deg(70.0), 10.0),
            (deg(-89.999_999), deg(-115.0), 0.0),
            (deg(20.0), deg(-70.0), 1.0e6),
            (deg(10.0), PI - 1.0e-12, -100.0),
            (deg(-10.0), -PI + 1.0e-12, 0.0),
        ];

        for (lat, lon, height) in cases {
            let pos_ecef = lat_lon_height_to_ecef(lat, lon, height);
            let (lat_rt, lon_rt, height_rt) = comp_lat_lon_height(&pos_ecef);
            let pos_ecef_rt = lat_lon_height_to_ecef(lat_rt, lon_rt, height_rt);

            assert_relative_eq!(lat_rt, lat, epsilon = 1.0e-8);
            if cos(lat).abs() > 1.0e-7 {
                assert_relative_eq!(angle_diff(lon_rt, lon), 0.0, epsilon = 1.0e-11);
            }
            assert_relative_eq!(height_rt, height, epsilon = 1.0e-3);
            let pos_error = (pos_ecef_rt - pos_ecef).norm();
            let pos_epsilon = if cos(lat).abs() > 1.0e-7 {
                1.0e-3
            } else {
                2.0e-2
            };
            assert!(pos_error <= pos_epsilon);
        }
    }

    #[test]
    fn rotations_are_inverse_and_orthonormal() {
        for (lat, lon) in [(0.0, 0.0), (deg(63.0), deg(10.0)), (deg(-30.0), deg(140.0))] {
            let ned_ecef = rot_ned_ecef(lat, lon);
            let ecef_ned = rot_ecef_ned(lat, lon);
            let identity = Matrix3::identity();

            assert_matrix_relative_eq(&(ned_ecef * ecef_ned), &identity, 1.0e-12);
            assert_matrix_relative_eq(&(ned_ecef * ned_ecef.transpose()), &identity, 1.0e-12);
            assert_matrix_relative_eq(&(ecef_ned * ecef_ned.transpose()), &identity, 1.0e-12);
        }
    }

    #[test]
    fn quaternion_matches_ned_ecef_rotation_matrix() {
        for (lat, lon) in [(0.0, 0.0), (deg(63.0), deg(10.0)), (deg(-30.0), deg(140.0))] {
            let quat_rotation = quat_ned_ecef(lat, lon).to_rotation_matrix();
            let matrix_rotation = rot_ned_ecef(lat, lon);

            assert_matrix_relative_eq(quat_rotation.matrix(), &matrix_rotation, 1.0e-12);
        }
    }

    #[test]
    fn gravity_sanity_at_equator_and_high_latitude() {
        let equator_model = grav_accel_ned(0.0, 0.0);
        let equator_ecef = gravity_ned(0.0, deg(20.0), 0.0);
        let high_latitude_model = grav_accel_ned(deg(89.0), 0.0);
        let high_latitude_ecef = gravity_ned(deg(89.0), deg(-40.0), 0.0);

        assert_relative_eq!(
            equator_model,
            Vector3::new(0.0, 0.0, 9.7803),
            epsilon = 1.0e-4
        );
        assert_relative_eq!(equator_ecef[0], 0.0, epsilon = 1.0e-9);
        assert_relative_eq!(equator_ecef[1], 0.0, epsilon = 1.0e-9);
        assert_relative_eq!(equator_ecef[2], equator_model[2], epsilon = 1.0e-12);
        assert!(high_latitude_model[2] > equator_model[2]);
        assert!(high_latitude_ecef[2] > equator_ecef[2]);
    }

    #[test]
    fn grav_accel_ned_uses_general_altitude_model() {
        let high_altitude = 10_000_000.0;
        let surface = grav_accel_ned(deg(45.0), 0.0);
        let high = grav_accel_ned(deg(45.0), high_altitude);
        let high_from_ecef = gravity_ned(deg(45.0), deg(120.0), high_altitude);

        assert_relative_eq!(high, high_from_ecef, epsilon = 1.0e-12);
        assert!(high[2] > 0.0);
        assert!(high[2] < surface[2]);
    }
}
