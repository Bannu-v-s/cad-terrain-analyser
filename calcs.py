"""calcs.py - the eight equations, applied to a gridded selection."""
import numpy as np
from scipy.interpolate import griddata


def trapezoidal_area(h, d):
    h = np.asarray(h, float)
    return float(d * ((h[0] + h[-1]) / 2 + h[1:-1].sum()))


def simpson_area(h, d):
    h = np.asarray(h, float)
    n = len(h)
    if n < 3 or n % 2 == 0:
        return None
    return float((d / 3) * (h[0] + h[-1] + 4 * h[1:-1:2].sum() + 2 * h[2:-1:2].sum()))


def average_end_area(areas, l):
    a = np.asarray(areas, float)
    return float(l * ((a[0] + a[-1]) / 2 + a[1:-1].sum()))


def prismoidal_volume(A1, Am, A2, l):
    return float((l / 3) * (A1 + 4 * Am + A2))


def shoelace_area(points):
    n = len(points)
    t = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        t += x1 * y2 - y1 * x2
    return abs(t) / 2


def spot_height_volume(g1, g2, g3, g4, A):
    return float((A / 4) * (sum(g1) + 2 * sum(g2) + 3 * sum(g3) + 4 * sum(g4)))


def triangular_prism(heights, A):
    return float((A / 3) * sum(heights))


def rectangular_prism(heights, A):
    return float(A * sum(heights) / 4)


def build_grid(pts, n=41):
    """pts: array of (x,y,z). Returns gx, gy, GZ, dx, dy on a regular n x n grid."""
    pts = np.asarray(pts, float)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    gx = np.linspace(x.min(), x.max(), n)
    gy = np.linspace(y.min(), y.max(), n)
    GX, GY = np.meshgrid(gx, gy)
    GZ = griddata((x, y), z, (GX, GY), method="linear")
    if np.isnan(GZ).any():
        near = griddata((x, y), z, (GX, GY), method="nearest")
        GZ = np.where(np.isnan(GZ), near, GZ)
    return gx, gy, GZ, gx[1] - gx[0], gy[1] - gy[0]


def spot_height_groups(GZ):
    """Split grid heights into the 1/2/3/4-square groups of eq 13.17."""
    g1 = [GZ[0, 0], GZ[0, -1], GZ[-1, 0], GZ[-1, -1]]
    g2 = list(GZ[0, 1:-1]) + list(GZ[-1, 1:-1]) \
        + list(GZ[1:-1, 0]) + list(GZ[1:-1, -1])
    g3 = []
    g4 = list(GZ[1:-1, 1:-1].ravel())
    return g1, g2, g3, g4


def volume_trapezoidal(gx, gy, GZ, dx, dy, datum):
    H = np.clip(GZ - datum, 0, None)
    rows = [trapezoidal_area(H[i, :], dx) for i in range(H.shape[0])]
    return trapezoidal_area(rows, dy), rows


def volume_simpson(gx, gy, GZ, dx, dy, datum):
    H = np.clip(GZ - datum, 0, None)
    rows = []
    for i in range(H.shape[0]):
        a = simpson_area(H[i, :], dx)
        if a is None:
            return None, None
        rows.append(a)
    return simpson_area(rows, dy), rows