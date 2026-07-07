"""WGS84 geodetic -> ECEF -> local ENU conversions, implemented directly in
numpy (deliberately no pymap3d dependency).

All functions accept scalars or numpy arrays and are fully vectorized.
Angles in degrees at the geodetic interface, radians internally.
"""

import numpy as np

# WGS84 ellipsoid
WGS84_A = 6_378_137.0                    # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563            # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)     # first eccentricity squared


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    """WGS84 geodetic coordinates -> ECEF (x, y, z) in metres."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    alt = np.asarray(alt_m, dtype=float)

    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    # prime-vertical radius of curvature at each latitude
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)

    x = (n + alt) * cos_lat * np.cos(lon)
    y = (n + alt) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return x, y, z


def ecef_to_enu(x, y, z, lat0_deg, lon0_deg, alt0_m):
    """ECEF (x, y, z) -> local East/North/Up metres relative to the origin
    (lat0, lon0, alt0)."""
    x0, y0, z0 = geodetic_to_ecef(lat0_deg, lon0_deg, alt0_m)
    dx = np.asarray(x, dtype=float) - x0
    dy = np.asarray(y, dtype=float) - y0
    dz = np.asarray(z, dtype=float) - z0

    lat0 = np.radians(float(lat0_deg))
    lon0 = np.radians(float(lon0_deg))
    sin_lat, cos_lat = np.sin(lat0), np.cos(lat0)
    sin_lon, cos_lon = np.sin(lon0), np.cos(lon0)

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return east, north, up


def geodetic_to_enu(lat_deg, lon_deg, alt_m, lat0_deg, lon0_deg, alt0_m):
    """WGS84 geodetic coordinates -> local East/North/Up metres relative to
    the origin (lat0, lon0, alt0). Convenience composition of the above."""
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_to_enu(x, y, z, lat0_deg, lon0_deg, alt0_m)


def wrap_angle_2pi(rad):
    """Wrap angle(s) in radians to [0, 2*pi)."""
    return np.mod(np.asarray(rad, dtype=float), 2.0 * np.pi)
