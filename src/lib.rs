#![no_std]
#![allow(non_snake_case)]

//! Navigation helper functions for WGS84 geodesy, ECEF/NED frame transforms,
//! Lie-group math primitives, and a simple Earth gravity model.
//!
//! # Conventions
//!
//! - Angles are radians.
//! - Distances and heights are meters.
//! - Velocities are meters per second.
//! - NED vectors are ordered `[north, east, down]`.
//! - ECEF vectors use the conventional Earth-centered, Earth-fixed axes.
//! - SE(3) twist helpers use `[translation, rotation]` ordering.
//!
//! # Linearization Helpers
//!
//! The crate includes analytic Jacobians for LLH-to-ECEF, ECEF-to-NED rotation,
//! and ECEF gravity. These are intended for estimators and strapdown navigation
//! code that need deterministic `no_std` primitives without pulling in full EKF
//! state machinery.
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
use na::{Matrix3, Quaternion, Rotation3, SMatrix, UnitQuaternion, Vector3};

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

/// Returns the skew-symmetric cross-product matrix for `v`.
///
/// For any vectors `v` and `w`, `skew(&v) * w == v.cross(&w)`.
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

/// Returns the active right-handed rotation matrix about the x-axis.
pub fn rotx(angle: f64) -> Matrix3<f64> {
    let s = sin(angle);
    let c = cos(angle);

    Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
}

/// Returns the active right-handed rotation matrix about the y-axis.
pub fn roty(angle: f64) -> Matrix3<f64> {
    let s = sin(angle);
    let c = cos(angle);

    Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
}

/// Returns the active right-handed rotation matrix about the z-axis.
pub fn rotz(angle: f64) -> Matrix3<f64> {
    let s = sin(angle);
    let c = cos(angle);

    Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
}

/// Wraps an angle in radians to the interval `[-pi, pi)`.
pub fn wrap_to_pi(angle: f64) -> f64 {
    if !angle.is_finite() {
        return f64::NAN;
    }

    let pi = core::f64::consts::PI;
    let two_pi = 2.0 * pi;
    let wrapped = (angle + pi) % two_pi;
    let wrapped = (wrapped + two_pi) % two_pi;

    wrapped - pi
}

/// Returns the SO(3) exponential map for a rotation vector.
///
/// The input vector direction is the rotation axis and its norm is the rotation
/// angle in radians.
pub fn so3_exp(rotation_vector: &Vector3<f64>) -> Matrix3<f64> {
    let theta_squared = rotation_vector.norm_squared();
    let theta = sqrt(theta_squared);
    let rotation_vector_skew = skew(rotation_vector);
    let rotation_vector_skew_squared = rotation_vector_skew * rotation_vector_skew;

    let (a, b) = if theta < 1.0e-8 {
        let theta_fourth = theta_squared * theta_squared;
        (
            1.0 - theta_squared / 6.0 + theta_fourth / 120.0,
            0.5 - theta_squared / 24.0 + theta_fourth / 720.0,
        )
    } else {
        (sin(theta) / theta, (1.0 - cos(theta)) / theta_squared)
    };

    Matrix3::identity() + a * rotation_vector_skew + b * rotation_vector_skew_squared
}

/// Returns the SO(3) left Jacobian for a rotation vector.
///
/// This matrix maps translational components in the SE(3) exponential map and
/// is also useful when linearizing rotation-vector perturbations.
pub fn so3_left_jacobian(rotation_vector: &Vector3<f64>) -> Matrix3<f64> {
    let theta_squared = rotation_vector.norm_squared();
    let theta = sqrt(theta_squared);
    let rotation_vector_skew = skew(rotation_vector);
    let rotation_vector_skew_squared = rotation_vector_skew * rotation_vector_skew;

    let (a, b) = if theta < 1.0e-6 {
        let theta_fourth = theta_squared * theta_squared;
        (
            0.5 - theta_squared / 24.0 + theta_fourth / 720.0,
            1.0 / 6.0 - theta_squared / 120.0 + theta_fourth / 5_040.0,
        )
    } else {
        (
            (1.0 - cos(theta)) / theta_squared,
            (theta - sin(theta)) / (theta_squared * theta),
        )
    };

    Matrix3::identity() + a * rotation_vector_skew + b * rotation_vector_skew_squared
}

/// Returns the SE(3) exponential map as `(rotation, translation)`.
///
/// The input convention is a twist ordered as `[translation, rotation_vector]`.
/// The returned translation is `so3_left_jacobian(rotation_vector) * translation`.
pub fn se3_exp(
    translation: &Vector3<f64>,
    rotation_vector: &Vector3<f64>,
) -> (Matrix3<f64>, Vector3<f64>) {
    (
        so3_exp(rotation_vector),
        so3_left_jacobian(rotation_vector) * translation,
    )
}

/// Returns the 6-by-6 Lie algebra adjoint matrix for an SE(3) twist.
///
/// The twist convention is `[translation, rotation]`, with both components
/// resolved in the same frame.
pub fn ad_se3(translation: &Vector3<f64>, rotation: &Vector3<f64>) -> SMatrix<f64, 6, 6> {
    let mut adjoint = SMatrix::<f64, 6, 6>::zeros();

    adjoint
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&skew(rotation));
    adjoint
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&skew(translation));
    adjoint
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&skew(rotation));

    adjoint
}

/// Applies an incremental rotation vector on the right side of a quaternion.
///
/// The returned quaternion is `quat * delta_q`, where `delta_q` is the unit
/// quaternion represented by `rotation_vector`.
pub fn quat_update_from_rotation_vector(
    quat: &UnitQuaternion<f64>,
    rotation_vector: &Vector3<f64>,
) -> UnitQuaternion<f64> {
    let angle_squared = rotation_vector.norm_squared();
    let angle = sqrt(angle_squared);

    let delta_quat = if angle > 1.0e-8 {
        let half_angle = 0.5 * angle;
        let vector_scale = sin(half_angle) / angle;
        UnitQuaternion::from_quaternion(Quaternion::new(
            cos(half_angle),
            vector_scale * rotation_vector[0],
            vector_scale * rotation_vector[1],
            vector_scale * rotation_vector[2],
        ))
    } else {
        let angle_fourth = angle_squared * angle_squared;
        let vector_scale = 0.5 - angle_squared / 48.0 + angle_fourth / 3_840.0;
        UnitQuaternion::from_quaternion(Quaternion::new(
            1.0 - angle_squared / 8.0 + angle_fourth / 384.0,
            vector_scale * rotation_vector[0],
            vector_scale * rotation_vector[1],
            vector_scale * rotation_vector[2],
        ))
    };

    quat * delta_quat
}

/// Converts a quaternion to a 4x modified Rodrigues parameter vector.
///
/// The quaternion sign is chosen internally so the scalar component is
/// non-negative, avoiding the MRP singularity for equivalent negative
/// quaternion representations.
pub fn quat_to_mrp4x(quat: &UnitQuaternion<f64>) -> Vector3<f64> {
    let quat = quat.quaternion();
    let scalar = quat.scalar();
    let vector = quat.vector();

    let mrp = if scalar >= 0.0 {
        vector / (1.0 + scalar)
    } else {
        -vector / (1.0 - scalar)
    };

    4.0 * mrp
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

/// Returns the Jacobian of [`lat_lon_height_to_ecef`].
///
/// Columns are partial derivatives with respect to `[lat, lon, height]`.
pub fn lat_lon_height_to_ecef_jacobian(lat: f64, lon: f64, height: f64) -> Matrix3<f64> {
    let sin_lat = sin(lat);
    let cos_lat = cos(lat);
    let sin_lon = sin(lon);
    let cos_lon = cos(lon);

    let denom = 1.0 - WGS84_E2 * sin_lat * sin_lat;
    let sqrt_denom = sqrt(denom);
    let radius_n = WGS84.a / sqrt_denom;
    let radius_n_dlat = WGS84.a * WGS84_E2 * sin_lat * cos_lat / (denom * sqrt_denom);

    let common_lat = radius_n_dlat * cos_lat - (radius_n + height) * sin_lat;
    let lat_col = Vector3::new(
        common_lat * cos_lon,
        common_lat * sin_lon,
        radius_n_dlat * (1.0 - WGS84_E2) * sin_lat
            + (radius_n * (1.0 - WGS84_E2) + height) * cos_lat,
    );
    let lon_col = Vector3::new(
        -(radius_n + height) * cos_lat * sin_lon,
        (radius_n + height) * cos_lat * cos_lon,
        0.0,
    );
    let height_col = Vector3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);

    Matrix3::from_columns(&[lat_col, lon_col, height_col])
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

/// Returns Earth rotation rate resolved in ECEF.
pub fn earth_rate_ecef() -> Vector3<f64> {
    Vector3::new(0.0, 0.0, EARTH_ROTATION_RATE)
}

/// Returns Earth rotation rate resolved in local NED.
///
/// Longitude is accepted for API symmetry with other local-frame helpers, but
/// the resolved Earth-rate vector is independent of longitude.
pub fn earth_rate_ned(lat: f64, _lon: f64) -> Vector3<f64> {
    Vector3::new(
        EARTH_ROTATION_RATE * cos(lat),
        0.0,
        -EARTH_ROTATION_RATE * sin(lat),
    )
}

/// Computes inertial rate of the NED frame resolved in NED.
///
/// This is `omega_in_n = omega_ie_n + omega_en_n`, where `omega_ie_n` is Earth
/// rotation resolved in NED and `omega_en_n` is the transport rate.
pub fn omega_in_n(lat: f64, height: f64, vel_eb_n: Vector3<f64>) -> Vector3<f64> {
    earth_rate_ned(lat, 0.0) + omega_en_n(lat, height, vel_eb_n)
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

/// Returns `d rot_ned_ecef(lat, lon) / d lat`.
pub fn rot_ned_ecef_jacobian_lat(lat: f64, lon: f64) -> Matrix3<f64> {
    let sin_lat = sin(lat);
    let cos_lat = cos(lat);
    let sin_lon = sin(lon);
    let cos_lon = cos(lon);

    Matrix3::new(
        -cos_lat * cos_lon,
        -cos_lat * sin_lon,
        -sin_lat,
        0.0,
        0.0,
        0.0,
        sin_lat * cos_lon,
        sin_lat * sin_lon,
        -cos_lat,
    )
}

/// Returns `d rot_ned_ecef(lat, lon) / d lon`.
pub fn rot_ned_ecef_jacobian_lon(lat: f64, lon: f64) -> Matrix3<f64> {
    let sin_lat = sin(lat);
    let cos_lat = cos(lat);
    let sin_lon = sin(lon);
    let cos_lon = cos(lon);

    Matrix3::new(
        sin_lat * sin_lon,
        -sin_lat * cos_lon,
        0.0,
        -cos_lon,
        -sin_lon,
        0.0,
        cos_lat * sin_lon,
        -cos_lat * cos_lon,
        0.0,
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

    fn assert_matrix6_relative_eq(
        lhs: &SMatrix<f64, 6, 6>,
        rhs: &SMatrix<f64, 6, 6>,
        epsilon: f64,
    ) {
        for row in 0..6 {
            for col in 0..6 {
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

    fn finite_difference_llh_to_ecef_jacobian(lat: f64, lon: f64, height: f64) -> Matrix3<f64> {
        let mut jacobian = Matrix3::zeros();
        let steps = [1.0e-6, 1.0e-6, 0.25];

        for col in 0..3 {
            let mut plus = [lat, lon, height];
            let mut minus = [lat, lon, height];
            plus[col] += steps[col];
            minus[col] -= steps[col];

            let pos_plus = lat_lon_height_to_ecef(plus[0], plus[1], plus[2]);
            let pos_minus = lat_lon_height_to_ecef(minus[0], minus[1], minus[2]);
            let diff = (pos_plus - pos_minus) / (2.0 * steps[col]);

            for row in 0..3 {
                jacobian[(row, col)] = diff[row];
            }
        }

        jacobian
    }

    fn finite_difference_rot_lat(lat: f64, lon: f64, step: f64) -> Matrix3<f64> {
        (rot_ned_ecef(lat + step, lon) - rot_ned_ecef(lat - step, lon)) / (2.0 * step)
    }

    fn finite_difference_rot_lon(lat: f64, lon: f64, step: f64) -> Matrix3<f64> {
        (rot_ned_ecef(lat, lon + step) - rot_ned_ecef(lat, lon - step)) / (2.0 * step)
    }

    #[test]
    fn skew_matches_cross_product() {
        let v = Vector3::new(1.0, -2.0, 0.5);
        let w = Vector3::new(-0.25, 0.75, 3.0);

        assert_relative_eq!(skew(&v) * w, v.cross(&w), epsilon = 1.0e-12);
        assert_matrix_relative_eq(&skew(&v), &(-skew(&v).transpose()), 1.0e-12);
    }

    #[test]
    fn axis_rotation_helpers_have_right_handed_signs() {
        let x = Vector3::new(1.0, 0.0, 0.0);
        let y = Vector3::new(0.0, 1.0, 0.0);
        let z = Vector3::new(0.0, 0.0, 1.0);

        assert_relative_eq!(rotx(FRAC_PI_2) * y, z, epsilon = 1.0e-12);
        assert_relative_eq!(rotx(FRAC_PI_2) * z, -y, epsilon = 1.0e-12);
        assert_relative_eq!(roty(FRAC_PI_2) * z, x, epsilon = 1.0e-12);
        assert_relative_eq!(roty(FRAC_PI_2) * x, -z, epsilon = 1.0e-12);
        assert_relative_eq!(rotz(FRAC_PI_2) * x, y, epsilon = 1.0e-12);
        assert_relative_eq!(rotz(FRAC_PI_2) * y, -x, epsilon = 1.0e-12);
    }

    #[test]
    fn wrap_to_pi_uses_half_open_interval() {
        assert_relative_eq!(wrap_to_pi(0.0), 0.0, epsilon = 0.0);
        assert_relative_eq!(wrap_to_pi(PI), -PI, epsilon = 0.0);
        assert_relative_eq!(wrap_to_pi(-PI), -PI, epsilon = 0.0);
        assert_relative_eq!(wrap_to_pi(3.0 * PI), -PI, epsilon = 0.0);
        assert_relative_eq!(wrap_to_pi(-3.0 * PI), -PI, epsilon = 0.0);
        assert_relative_eq!(wrap_to_pi(5.0 * PI / 2.0), PI / 2.0, epsilon = 1.0e-12);
        assert!(wrap_to_pi(f64::INFINITY).is_nan());
    }

    #[test]
    fn so3_exp_matches_axis_rotations_and_small_angle_series() {
        let angle = 0.4;
        assert_matrix_relative_eq(
            &so3_exp(&Vector3::new(angle, 0.0, 0.0)),
            &rotx(angle),
            1.0e-12,
        );
        assert_matrix_relative_eq(
            &so3_exp(&Vector3::new(0.0, angle, 0.0)),
            &roty(angle),
            1.0e-12,
        );
        assert_matrix_relative_eq(
            &so3_exp(&Vector3::new(0.0, 0.0, angle)),
            &rotz(angle),
            1.0e-12,
        );

        let small = Vector3::new(1.0e-9, -2.0e-9, 3.0e-9);
        let small_skew = skew(&small);
        let expected = Matrix3::identity() + small_skew + 0.5 * small_skew * small_skew;
        assert_matrix_relative_eq(&so3_exp(&small), &expected, 1.0e-24);
    }

    #[test]
    fn so3_left_jacobian_matches_small_angle_series() {
        let small = Vector3::new(1.0e-7, -2.0e-7, 3.0e-7);
        let small_skew = skew(&small);
        let expected = Matrix3::identity() + 0.5 * small_skew + small_skew * small_skew / 6.0;

        assert_matrix_relative_eq(&so3_left_jacobian(&small), &expected, 1.0e-20);
    }

    #[test]
    fn se3_exp_matches_known_rotation_and_translation() {
        let translation = Vector3::new(1.0, 0.5, 0.3);
        let rotation = Vector3::new(0.3, 0.3, 0.3);
        let (rot, trans) = se3_exp(&translation, &rotation);

        let expected_rot = Matrix3::new(
            0.9120, -0.2427, 0.3307, 0.3307, 0.9120, -0.2427, -0.2427, 0.3307, 0.9120,
        );
        let expected_trans = Vector3::new(0.9529, 0.6071, 0.2400);

        assert_matrix_relative_eq(&rot, &expected_rot, 1.0e-4);
        assert_relative_eq!(trans, expected_trans, epsilon = 1.0e-4);

        let zero_rotation = Vector3::zeros();
        let (identity, unchanged_translation) = se3_exp(&translation, &zero_rotation);
        assert_matrix_relative_eq(&identity, &Matrix3::identity(), 1.0e-12);
        assert_relative_eq!(unchanged_translation, translation, epsilon = 1.0e-12);
    }

    #[test]
    fn ad_se3_has_expected_twist_action() {
        use crate::na::SVector;

        let translation = Vector3::new(0.4, -0.2, 0.8);
        let rotation = Vector3::new(0.1, 0.2, -0.3);
        let other_translation = Vector3::new(-0.5, 0.7, 1.2);
        let other_rotation = Vector3::new(0.3, -0.4, 0.5);
        let other_twist = SVector::<f64, 6>::new(
            other_translation[0],
            other_translation[1],
            other_translation[2],
            other_rotation[0],
            other_rotation[1],
            other_rotation[2],
        );

        let mut expected_ad = SMatrix::<f64, 6, 6>::zeros();
        expected_ad
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&skew(&rotation));
        expected_ad
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&skew(&translation));
        expected_ad
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&skew(&rotation));
        assert_matrix6_relative_eq(&ad_se3(&translation, &rotation), &expected_ad, 1.0e-12);

        let actual = expected_ad * other_twist;
        let expected_translation =
            rotation.cross(&other_translation) + translation.cross(&other_rotation);
        let expected_rotation = rotation.cross(&other_rotation);
        let expected = SVector::<f64, 6>::new(
            expected_translation[0],
            expected_translation[1],
            expected_translation[2],
            expected_rotation[0],
            expected_rotation[1],
            expected_rotation[2],
        );

        assert_relative_eq!(actual, expected, epsilon = 1.0e-12);
    }

    #[test]
    fn quat_update_from_rotation_vector_matches_so3_exp() {
        let rotation_vector = Vector3::new(0.1, -0.2, 0.3);
        let updated =
            quat_update_from_rotation_vector(&UnitQuaternion::identity(), &rotation_vector);

        assert_matrix_relative_eq(
            updated.to_rotation_matrix().matrix(),
            &so3_exp(&rotation_vector),
            1.0e-12,
        );

        let base =
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotz(0.2)));
        let delta = Vector3::new(0.0, 0.0, 0.1);
        let updated = quat_update_from_rotation_vector(&base, &delta);

        assert_matrix_relative_eq(updated.to_rotation_matrix().matrix(), &rotz(0.3), 1.0e-12);
    }

    #[test]
    fn quat_to_mrp4x_uses_equivalent_positive_scalar_quaternion() {
        let identity_negative =
            UnitQuaternion::from_quaternion(Quaternion::new(-1.0, 0.0, 0.0, 0.0));
        assert_relative_eq!(
            quat_to_mrp4x(&identity_negative),
            Vector3::zeros(),
            epsilon = 1.0e-12
        );

        let half_angle = PI / 4.0;
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(
            cos(half_angle),
            0.0,
            0.0,
            sin(half_angle),
        ));
        let expected = Vector3::new(0.0, 0.0, 4.0 * sin(half_angle) / (1.0 + cos(half_angle)));

        assert_relative_eq!(quat_to_mrp4x(&quat), expected, epsilon = 1.0e-12);
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
    fn lat_lon_height_to_ecef_jacobian_matches_central_difference() {
        let cases = [
            (0.0, 0.0, 0.0),
            (deg(63.0), deg(10.0), 50.0),
            (deg(-45.0), deg(170.0), 1_000.0),
            (deg(80.0), deg(-30.0), 10.0),
            (deg(20.0), deg(-70.0), 1.0e6),
        ];

        for (lat, lon, height) in cases {
            let analytic = lat_lon_height_to_ecef_jacobian(lat, lon, height);
            let finite_difference = finite_difference_llh_to_ecef_jacobian(lat, lon, height);

            assert_matrix_relative_eq(&analytic, &finite_difference, 5.0e-3);
        }
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
        let cases = [
            Vector3::new(2.859253e6, 0.504163e6, 5.660022e6),
            lat_lon_height_to_ecef(0.0, 0.0, 0.0),
            lat_lon_height_to_ecef(deg(-45.0), deg(170.0), 1_000.0),
            lat_lon_height_to_ecef(deg(80.0), deg(-30.0), 10.0),
            lat_lon_height_to_ecef(deg(20.0), deg(-70.0), 1.0e6),
        ];

        for p_eb in cases {
            let analytic = gravity_gradient_ecef(&p_eb);
            let finite_difference = finite_difference_gravity_gradient(&p_eb, 10.0);

            assert_matrix_relative_eq(&analytic, &finite_difference, 1.0e-10);
            assert_matrix_relative_eq(&analytic, &analytic.transpose(), 1.0e-12);
        }
    }

    #[test]
    fn earth_rate_helpers_resolve_rate_in_expected_frames() {
        let lat = deg(63.0);
        let lon = deg(10.0);
        let expected_ned = Vector3::new(
            EARTH_ROTATION_RATE * cos(lat),
            0.0,
            -EARTH_ROTATION_RATE * sin(lat),
        );

        assert_relative_eq!(
            earth_rate_ecef(),
            Vector3::new(0.0, 0.0, EARTH_ROTATION_RATE),
            epsilon = 0.0
        );
        assert_relative_eq!(earth_rate_ned(lat, lon), expected_ned, epsilon = 1.0e-20);
        assert_relative_eq!(
            earth_rate_ned(lat, lon),
            rot_ned_ecef(lat, lon) * earth_rate_ecef(),
            epsilon = 1.0e-20
        );
    }

    #[test]
    fn omega_in_n_is_earth_rate_plus_transport_rate() {
        let lat = deg(63.0);
        let height = 50.0;
        let vel_eb_n = Vector3::new(12.0, -3.0, 0.5);
        let expected = earth_rate_ned(lat, 0.0) + omega_en_n(lat, height, vel_eb_n);

        assert_relative_eq!(
            omega_in_n(lat, height, vel_eb_n),
            expected,
            epsilon = 1.0e-20
        );
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
    fn rot_ned_ecef_jacobians_match_central_difference() {
        for (lat, lon) in [(0.0, 0.0), (deg(63.0), deg(10.0)), (deg(-30.0), deg(140.0))] {
            let lat_analytic = rot_ned_ecef_jacobian_lat(lat, lon);
            let lon_analytic = rot_ned_ecef_jacobian_lon(lat, lon);
            let lat_finite_difference = finite_difference_rot_lat(lat, lon, 1.0e-7);
            let lon_finite_difference = finite_difference_rot_lon(lat, lon, 1.0e-7);

            assert_matrix_relative_eq(&lat_analytic, &lat_finite_difference, 3.0e-9);
            assert_matrix_relative_eq(&lon_analytic, &lon_finite_difference, 3.0e-9);
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
