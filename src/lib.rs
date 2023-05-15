#![no_std]
// #![cfg_attr(feature = "no_std", no_std)]

// #[cfg(feature = "no_std")]
// extern crate core as std;
use libm::{sqrt, sin, cos, pow, tan};

extern crate nalgebra as na;
use na::{SMatrix, Vector3, UnitQuaternion, Quaternion, Matrix3, Rotation3};

// use approx::assert_relative_eq;

pub struct Ellipsoid {
    a: f64,
    f: f64,
}

const WGS84: Ellipsoid = Ellipsoid {
    a: 6378137.0,
    f: 1.0 / 298.257222101,
};


pub fn lat_lon_height_to_ecef(lat: f64, lon: f64, height: f64) -> Vector3<f64> {
    // a:      WGS84 semi major axis
    // e:      WGS84 eccentricity
    // radius_n:    Normal radius / prime vertical radius
    
    let a = 6378137.0;
    let e = 0.08181919;


    let radius_n = a / sqrt( 1.0 - pow(e, 2.0) * pow(sin(lat), 2.0) );
    Vector3::new(
        (radius_n + height) * cos(lat) * cos(lon),
        (radius_n + height) * cos(lat) * sin(lon),
        (radius_n * ( 1.0 - pow(e, 2.0) ) + height) * sin(lat)
    )
}

pub fn omega_en_n(lat: f64, height: f64, vel_eb_n: Vector3<f64>) -> Vector3<f64> {
    let a = WGS84.a;
    let e = 0.08181919;

    let radius_e = a / sqrt( 1.0 - pow(e, 2.0) * pow(sin(lat), 2.0) );
    let radius_n = a * (1.0 - pow(e, 2.0)) / pow( 1.0 - pow(e, 2.0) * pow(sin(lat), 2.0), 1.5);

    Vector3::new(
      vel_eb_n[1] / (radius_e + height),
      - vel_eb_n[0] / (radius_n + height),
      -vel_eb_n[1] * tan(lat) / (radius_e + height)
    )
}

pub fn gravity_ecef(r_eb_e: &Vector3<f64>) -> Vector3<f64> {
    let a           = 6378137.0;        // WGS84 equatorial radius in meters
    let mu          = 3.986004418e14; // WGS84 Earth gravitational constant (m^3 s^-2)
    let J_2         = 1.082627e-3;    // WGS84 Earth's second gravitational constant
    let omega_ie    = 7.292115e-5;    // Earth's rotation rate (rad/s)

    let r_norm = r_eb_e.norm();

    let z_scale = 5.0 * pow(r_eb_e[2] / r_norm, 2.0);
    let gamma = - mu / pow(r_norm, 3.0) * (r_eb_e + 1.5 * J_2 
        * pow(a / r_norm, 2.0) 
        *  Vector3::new((1.0 - z_scale ) * r_eb_e[0], 
                        (1.0 - z_scale) * r_eb_e[1], 
                        (3.0 - z_scale) * r_eb_e[2]));

    gamma + pow(omega_ie, 2.0) * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 0.0)) * r_eb_e
}

pub fn gravity_ned(lat: f64, lon: f64, height: f64) -> Vector3<f64> {
    let r_eb_e = lat_lon_height_to_ecef(lat, lon, height);
    let g_e = gravity_ecef(&r_eb_e);
    
    let rot_ned_ecef = Matrix3::new(
        -sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat),
        -sin(lon), cos(lon), 0.0,
        -cos(lat) * cos(lon), -cos(lat) * sin(lon), -sin(lat)
    );

    rot_ned_ecef * g_e
}

pub fn quat_ned_ecef(lat: f64, lon: f64) -> UnitQuaternion<f64> {
    let rot_ned_ecef = Matrix3::new(
        -sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat),
        -sin(lon), cos(lon), 0.0,
        -cos(lat) * cos(lon), -cos(lat) * sin(lon), -sin(lat)
    );

    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rot_ned_ecef))
}

pub fn rot_ned_ecef(lat: f64, lon: f64) -> Matrix3<f64> {
    Matrix3::new(
        -sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat),
        -sin(lon), cos(lon), 0.0,
        -cos(lat) * cos(lon), -cos(lat) * sin(lon), -sin(lat)
    )
}

pub fn rot_ecef_ned(lat: f64, lon: f64) -> Matrix3<f64> {
    Matrix3::new(
        -sin(lat) * cos(lon), -sin(lon), -cos(lat) * cos(lon),
        -sin(lat) * sin(lon), cos(lon), -cos(lat) * sin(lon),
        cos(lat), 0.0, -sin(lat)
    )
}

pub fn lat_height_to_grav_accel_ned(lat: f64, height: f64) -> Vector3<f64> {
    // let radius_wgs84 = 6378137.0;             //WGS84 Equatorial radius in meters
    let e = 0.0818191908425;       // WGS84 eccentricity
    // let f = 1.0 / 298.257223563;   // WGS84 flattening
    let radius_wgs84 = WGS84.a;
    let f = WGS84.f;

    let b = radius_wgs84 * (1.0 - f);          // WGS84 Polar radius in meters
    // b = 6356752.31425; 

    let mu = 3.986004418e14;    // WGS84 Earth gravitational constant (m^3 s^-2)
    let omega_ie = 7.292115E-5; // Earth rotation rate (rad/s)

    // Calculate surface gravity using the Somigliana model, (2.134)
    let sin_lat_squared = sin(lat) * sin(lat);
    let g_0 = 9.7803253359 * (1.0 + 0.001931853 * sin_lat_squared) / sqrt(1.0 - e * e * sin_lat_squared);

    Vector3::new(
        -8.08e-9 * height * sin(2.0 * lat), // Calculate north gravity using (2.140)
        0.0, // East gravity is zero
        g_0 * (1.0 - (2.0 / radius_wgs84) * (1.0 + f * (1.0 - 2.0 * sin_lat_squared) +
        (pow(omega_ie, 2.0) * pow(radius_wgs84, 2.0) * b / mu)) * height + (3.0 * pow(height, 2.0) / pow(radius_wgs84, 2.0)))   // Calculate down gravity using (2.139)
    )
    }


#[cfg(test)]
mod tests {
    use core::f64::consts::PI;
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn check_gravity() {
        let lat = 63.0 * PI / 180.0;
        let lon = 10.0 * PI / 180.0;
        let height = 50.0;

        let grav_1 = gravity_ned(lat, lon, height);
        let grav_2 = lat_height_to_grav_accel_ned(lat, height);

        assert_relative_eq!(grav_1, Vector3::new(0.0, 0.0, 9.8213), epsilon = 0.0001);
        assert_relative_eq!(grav_2, Vector3::new(0.0, 0.0, 9.8213), epsilon = 0.0001);
        
    }

    #[test]
    fn test_lat_lon_h_to_ecef() {
        let lat = 63.0 * PI / 180.0;
        let lon = 10.0 * PI / 180.0;
        let height = 50.0;

        let r_eb_e = lat_lon_height_to_ecef(lat, lon, height);

        assert_relative_eq!(r_eb_e, Vector3::new(2.859253e6, 0.504163e6, 5.660022e6), epsilon = 1.0);
    }

    #[test]
    fn gravity_ecef_test() {
        let p_eb = Vector3::new(2.859253e6, 0.504163e6, 5.660022e6);

        let grav = gravity_ecef(&p_eb);      
        // println!("gravity: {}", grav);  

        assert_relative_eq!(grav, Vector3::new(-4.3910, -0.7742, -8.7509), epsilon = 0.0001);

    }
}
