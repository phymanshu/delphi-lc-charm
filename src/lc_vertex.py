"""
lc_vertex.py - secondary-vertex fit from the DELPHI track parameters in the lc dump.

Each track is given by the five perigee parameters of the dump
    d0 (R-phi impact parameter), z0, theta, phi, kappa (= 1/R, signed)
and the 5x5 *weight* matrix (= inverse covariance) stored as the 15 columns w1..w15
(lower triangle, row by row: w1=(1,1), w2=(2,1), w3=(2,2), w4=(3,1), ...).
The real-data check showed the diagonals are ~1e6, 5e4, 1e7, 4e7, 2e9, i.e. weights.

The dump does not tell us the sign conventions, so they are explicit and are fixed
from the data with lc_vertex_scan.py:
    Conv.sd   : the perigee point is  ref + sd * d0 * n,  n = (-sin phi, cos phi)
    Conv.sk   : kappa used = sk * exported curvature (kappa > 0 = counter-clockwise)
    Conv.zrel : z0 is relative to the reference z (True) or absolute (False)
ref is the point the exported d0 is measured from. REAL DATA: this is the ORIGIN (0,0,0), not the primary vertex
(found with lc_vertex_scan.py: d0 - ip_pv is +-0.33 cm = the beam offset; ip_pv/zip_pv are the PV-referenced
quantities). Pass ref = zeros for the fit and measure decay lengths from the event PV with decay_length().

Fit: Billoir-style. The helix must pass through the vertex v: two constraints per track
g_i(v, alpha_i) = (transverse distance of v from the circle, z mismatch) = 0. With the
first-order propagation S_i = B_i C_i B_i^T (B = dg/dalpha, C = W^-1), minimise
sum_i g_i^T S_i^-1 g_i over v. ndf = 2*ntracks - 3.
"""
from dataclasses import dataclass
from math import erfc, exp, pi, sqrt

import numpy as np

# step sizes for the numerical derivatives: d0, z0, theta, phi (kappa is relative, see below)
_STEPS = np.array([1e-4, 1e-4, 1e-5, 1e-5])


@dataclass(frozen=True)
class Conv:
    sd: int = 1
    sk: int = 1
    zrel: bool = False

    def __str__(self):
        return f"sd={self.sd:+d} sk={self.sk:+d} z0 {'relative' if self.zrel else 'absolute'}"


# fixed from the real data (D*-tagged D0 scan, 50 files of short94_c2): d0 sign -1, z0 absolute.
# sk is not constrained by mm-long flights; sign(curv) == -q for every track, i.e. kappa > 0 is counter-clockwise.
DEFAULT_CONV = Conv(sd=-1, sk=+1, zrel=False)
# robust core width of the PV-referenced d0 / z0 pulls on real data is 1.3-1.5, flat in p (lc_pull_check.py)
DEFAULT_ERR_SCALE = 1.4

ALL_CONVENTIONS = [Conv(sd, sk, zr) for sd in (1, -1) for sk in (1, -1) for zr in (False, True)]


def unpack_w(w):
    """(..., 15) packed lower triangle -> (..., 5, 5) symmetric matrix."""
    w = np.asarray(w, dtype=float)
    out = np.zeros(w.shape[:-1] + (5, 5))
    k = 0
    for i in range(5):
        for j in range(i + 1):
            out[..., i, j] = w[..., k]
            out[..., j, i] = w[..., k]
            k += 1
    return out


def _geom(v, par, ref, conv):
    """returns g (N,2) and delta (N,) = angle swept along the helix from the perigee to the vertex."""
    d0, z0, th, ph, ka = (par[:, i] for i in range(5))
    ka = conv.sk * ka
    ka = np.where(np.abs(ka) < 1e-9, np.where(ka < 0, -1e-9, 1e-9), ka)
    sp, cp = np.sin(ph), np.cos(ph)
    xp = ref[:, 0] - conv.sd * d0 * sp
    yp = ref[:, 1] + conv.sd * d0 * cp
    zp = (ref[:, 2] + z0) if conv.zrel else z0
    ux, uy = v[:, 0] - xp, v[:, 1] - yp
    a = ux * cp + uy * sp                 # along the track at the perigee
    b = -ux * sp + uy * cp                # to the left of the track
    inv = 1.0 / ka
    wn = np.hypot(a, b - inv)             # |v - circle centre|
    g1 = (ux * ux + uy * uy - 2.0 * b * inv) / (wn + np.abs(inv))   # = |w| - R, without cancellation
    delta = np.arctan2(ka * a, 1.0 - ka * b)                        # angle swept from the perigee to v
    s = delta * inv                       # transverse arc length
    g2 = zp + s / np.tan(th) - v[:, 2]
    return np.stack([g1, g2], axis=1), delta


def constraint(v, par, ref, conv):
    """g = (distance of v from the track circle, z mismatch), shape (N, 2).
    v (N,3), par (N,5) = d0,z0,theta,phi,kappa(exported), ref (N,3)."""
    return _geom(v, par, ref, conv)[0]


def phi_at_vertex(v, par, ref, conv):
    """Azimuth of the momentum at the vertex v: the perigee phi rotated by the angle swept along the helix.
    The sign of the rotation follows the convention conv.sk (kappa > 0 = counter-clockwise)."""
    return par[:, 3] + _geom(v, par, ref, conv)[1]


def _derivs(v, par, ref, conv):
    N = len(v)
    A = np.zeros((N, 2, 3))
    for k in range(3):
        dv = np.zeros_like(v)
        dv[:, k] = 1e-4
        A[:, :, k] = (constraint(v + dv, par, ref, conv) - constraint(v - dv, par, ref, conv)) / 2e-4
    B = np.zeros((N, 2, 5))
    for k in range(5):
        h = np.full(N, _STEPS[k]) if k < 4 else 1e-4 * np.abs(par[:, 4]) + 1e-10
        dp = np.zeros_like(par)
        dp[:, k] = h
        B[:, :, k] = (constraint(v, par + dp, ref, conv) - constraint(v, par - dp, ref, conv)) / (2 * h)[:, None]
    return A, B


def _inv2(S):
    a, b, c, d = S[:, 0, 0], S[:, 0, 1], S[:, 1, 0], S[:, 1, 1]
    det = a * d - b * c
    det = np.where(np.abs(det) < 1e-300, np.nan, det)
    out = np.empty_like(S)
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1] = d / det, -b / det, -c / det, a / det
    return out


def fit_vertex(par, W, ref, conv, niter=5, v0=None, err_scale=1.0):
    """par (N,n,5), W (N,n,5,5), ref (N,3) -> v (N,3), chi2 (N,), Cv (N,3,3). NaN where the fit failed.
    err_scale multiplies all track errors (the real-data pull check gave a core width of about 1.4)."""
    N, n, _ = par.shape
    with np.errstate(all="ignore"):
        C = np.linalg.inv(W) * err_scale ** 2
        v = ref.copy() if v0 is None else v0.copy()
        for _ in range(niter):
            H = np.zeros((N, 3, 3))
            bvec = np.zeros((N, 3, 1))
            for i in range(n):
                g0 = constraint(v, par[:, i], ref, conv)
                A, B = _derivs(v, par[:, i], ref, conv)
                S = B @ C[:, i] @ B.transpose(0, 2, 1)
                Si = _inv2(S)
                At = A.transpose(0, 2, 1)
                H += At @ Si @ A
                bvec += At @ Si @ g0[:, :, None]
            dv = -np.linalg.solve(H + 1e-12 * np.eye(3), bvec)[:, :, 0]
            v = v + dv
        chi2 = np.zeros(N)
        for i in range(n):
            g0 = constraint(v, par[:, i], ref, conv)
            A, B = _derivs(v, par[:, i], ref, conv)
            Si = _inv2(B @ C[:, i] @ B.transpose(0, 2, 1))
            chi2 += np.einsum("ni,nij,nj->n", g0, Si, g0)
        Cv = np.linalg.inv(H + 1e-12 * np.eye(3))
    bad = ~np.isfinite(chi2) | ~np.isfinite(v).all(axis=1)
    chi2[bad] = np.nan
    return v, chi2, Cv


def chi2_sf(x, ndf):
    """P(chi2 > x) for ndf = 1 or 3 (2 or 3 track vertices); closed forms, no scipy needed."""
    x = np.asarray(x, dtype=float)
    f = np.vectorize(lambda t: erfc(sqrt(t / 2.0)) if ndf == 1
                     else erfc(sqrt(t / 2.0)) + sqrt(2.0 * t / pi) * exp(-t / 2.0))
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & (x >= 0)
    if ok.any():
        out[ok] = f(x[ok])
    return out


def decay_length(v, ref, mom):
    """Transverse decay length: (v - ref)_xy projected on the xy direction of mom (N,3). Positive = downstream."""
    d = v[:, :2] - ref[:, :2]
    m = mom[:, :2]
    return (d * m).sum(axis=1) / np.hypot(m[:, 0], m[:, 1])
