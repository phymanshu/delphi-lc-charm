"""
Synthetic dump in the lc_schema format, with injected Lambda_c -> p K pi or D+ -> K pi pi decays whose tracks come
from a displaced vertex. The tracks are written the way the real dump behaves: perigee parameters measured from the
ORIGIN (sd = -1, z0 absolute), ip_pv / zip_pv measured from the primary vertex, diagonal weight matrices.
Used only by the smoke tests; it says nothing about real-data performance.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from lc_schema import EVENT_COLS, TRACK_COLS  # noqa: E402
from lc_vertex import Conv  # noqa: E402

TRUE = Conv(sd=-1, sk=+1, zrel=False)
MP, MK, MPI, MLC, MD = 0.93827, 0.49368, 0.13957, 2.28646, 1.86965


def helix_to_perigee(v, p3, q, ref, conv=TRUE, B=1.23):
    """perigee parameters (d0, z0, theta, phi, kappa) of the helix through v with momentum p3, seen from ref"""
    pt = np.hypot(p3[:, 0], p3[:, 1])
    ph = np.arctan2(p3[:, 1], p3[:, 0])
    th = np.arccos(p3[:, 2] / np.linalg.norm(p3, axis=1))
    ka = q * 0.2998 * B * 0.01 / pt
    nx, ny = -np.sin(ph), np.cos(ph)
    cx, cy = v[:, 0] + nx / ka, v[:, 1] + ny / ka
    rx, ry = ref[:, 0] - cx, ref[:, 1] - cy
    rn = np.hypot(rx, ry)
    ux, uy = rx / rn, ry / rn
    px_, py_ = cx + ux / np.abs(ka), cy + uy / np.abs(ka)
    sg = np.sign(ka)
    psi = np.arctan2(sg * ux, -sg * uy)
    d0 = conv.sd * ((px_ - ref[:, 0]) * (-np.sin(psi)) + (py_ - ref[:, 1]) * np.cos(psi))
    delta = (ph - psi + np.pi) % (2 * np.pi) - np.pi
    z_p = v[:, 2] - (delta / ka) / np.tan(th)
    return np.stack([d0, z_p - (ref[:, 2] if conv.zrel else 0), th, psi, conv.sk * ka], axis=1)


def _boost(v, mass, b):
    b2 = b @ b
    g = 1 / np.sqrt(1 - b2)
    e = np.sqrt(v @ v + mass ** 2)
    return v + ((g - 1) * (b @ v) / b2 + g * e) * b


def _decay2(rng, M, m1, m2):
    pst = np.sqrt(max((M ** 2 - (m1 + m2) ** 2) * (M ** 2 - (m1 - m2) ** 2), 0)) / (2 * M)
    c, f = rng.uniform(-1, 1), rng.uniform(0, 2 * np.pi)
    s = np.sqrt(1 - c * c)
    d = np.array([pst * s * np.cos(f), pst * s * np.sin(f), pst * c])
    return d, np.sqrt(d @ d + m1 ** 2), -d, np.sqrt(d @ d + m2 ** 2)


def _three_body(rng, M, m1, m2, m3, Pv):
    """isotropic decay of M (momentum Pv) into three bodies via a (m2 m3) system; returns the three lab momenta"""
    E = np.sqrt(Pv @ Pv + M ** 2)
    bP = Pv / E
    mX = rng.uniform(m2 + m3 + 0.02, M - m1 - 0.02)
    d1, _, dX, EX = _decay2(rng, M, m1, mX)
    d2, _, d3, _ = _decay2(rng, mX, m2, m3)
    bX = dX / EX
    return (_boost(d1, m1, bP), _boost(_boost(d2, m2, bX), m2, bP), _boost(_boost(d3, m3, bX), m3, bP))


def make(prefix, channel="lc", n_events=3000, frac=0.15, seed=1, duplicate_event_ids=False):
    """writes <prefix>.events.csv and <prefix>.tracks.csv; returns the number of injected decays"""
    rng = np.random.default_rng(seed)
    rows, evrows, n_inj = [], [], 0
    org = np.zeros((1, 3))
    for e in range(n_events):
        pv = np.array([[rng.normal(-0.3, 0.012), rng.normal(0.14, 0.0012), rng.normal(-0.7, 0.7)]])
        recs = []

        def add(p3, q, vtx, idx, tags=None):
            par = helix_to_perigee(vtx[None, :], p3[None, :], np.array([q * 1.0]), org)[0]
            ipz = helix_to_perigee(vtx[None, :], p3[None, :], np.array([q * 1.0]), pv)[0]
            sig = np.array([2e-3, 6e-3, 4e-4, 3e-4, 2e-6 + 2e-3 * abs(par[4])])
            noise = rng.normal(size=5) * sig
            W = np.diag(1 / sig ** 2)
            d = dict.fromkeys(TRACK_COLS, 0.0)
            d.update(run=1, evt=e, idx=idx, q=q, lock=0, px=p3[0], py=p3[1], pz=p3[2], trklen=-100.0,
                     ip_pv=ipz[0] + noise[0], zip_pv=ipz[1] - pv[0, 2] + noise[1], eerr=0.03, vd_rphi=1, vd_rz=1,
                     d0=par[0] + noise[0], z0=par[1] + noise[1], theta=par[2] + noise[2], phi=par[3] + noise[3],
                     curv=par[4] + noise[4])
            for k, (i, j) in enumerate((i, j) for i in range(5) for j in range(i + 1)):
                d[f"w{k + 1}"] = W[i, j]
            if tags is None:      # background tracks: mostly untagged
                d["h_rich_p"] = rng.choice([-1, 0, 1, 2, 3], p=[0.2, 0.72, 0.03, 0.03, 0.02])
                d["h_rich_k"] = rng.choice([-1, 0, 1, 2, 3], p=[0.2, 0.70, 0.04, 0.04, 0.02])
            else:
                d["h_rich_p"], d["h_rich_k"] = tags
            d["h_dedx_p"] = d["h_dedx_k"] = -1
            recs.append(d)

        idx = 1
        for _ in range(14):
            p = rng.exponential(2.5) + 1.0
            ct, ph = rng.uniform(-.8, .8), rng.uniform(-np.pi, np.pi)
            st = np.sqrt(1 - ct ** 2)
            add(np.array([p * st * np.cos(ph), p * st * np.sin(ph), p * ct]), rng.choice([-1, 1]), pv[0], idx)
            idx += 1
        if rng.random() < frac:
            P = rng.uniform(28, 42)
            ct, ph = rng.uniform(-.8, .8), rng.uniform(-np.pi, np.pi)
            st = np.sqrt(1 - ct ** 2)
            Pv = np.array([P * st * np.cos(ph), P * st * np.sin(ph), P * ct])
            q = rng.choice([-1, 1])
            good = rng.random() < 0.8
            if channel == "lc":
                legs = _three_body(rng, MLC, MP, MK, MPI, Pv)
                vtx = pv[0] + rng.exponential(0.09) * Pv / P
                add(legs[0], +q, vtx, idx, (2 if good else 0, -1)); idx += 1
                add(legs[1], -q, vtx, idx, (-1, 2 if good else 0)); idx += 1
                add(legs[2], +q, vtx, idx, (-1, -1)); idx += 1
            else:
                legs = _three_body(rng, MD, MK, MPI, MPI, Pv)
                vtx = pv[0] + rng.exponential(0.31 * P / MD * 0.5) * Pv / P
                add(legs[0], -q, vtx, idx, (-1, 2 if good else 0)); idx += 1
                add(legs[1], +q, vtx, idx, (-1, -1)); idx += 1
                add(legs[2], +q, vtx, idx, (-1, -1)); idx += 1
            n_inj += 1
        rows += recs
        ev = dict.fromkeys(EVENT_COLS, 0.0)
        ev.update(run=1, evt=e, ihad4=1, nctr4=idx - 1, echar=60.0, ecm=91.2, nwrit=idx - 1, pv_x=pv[0, 0],
                  pv_y=pv[0, 1], pv_z=pv[0, 2], pv_xx=1e-6, pv_yy=1e-6, pv_zz=1e-5, btn3=0.5, btp3=0.4, bts3=0.4)
        evrows.append(ev)
    tr, ev = pd.DataFrame(rows)[TRACK_COLS], pd.DataFrame(evrows)[EVENT_COLS]
    if duplicate_event_ids:       # the real dump has (run, evt) pairs that occur twice in a file
        tr.loc[tr.evt == 7, "evt"] = 6
        ev.loc[ev.evt == 7, "evt"] = 6
    ev.to_csv(prefix + ".events.csv", header=False, index=False)
    tr.to_csv(prefix + ".tracks.csv", header=False, index=False)
    return n_inj
