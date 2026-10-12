#!/usr/bin/env python3
"""
lc_pull_check.py - are the exported track errors (the weight matrix) the right size?

Most tracks in a hadronic event come from the primary vertex, so for them d0/sigma(d0) and
zip_pv/sigma(z0) should be ~N(0,1) in their core. The core width (1.4826 * MAD, robust
against the secondary-vertex tails) tells how much the errors are over/under-estimated:
  width ~ 1      errors are right
  width ~ 3      errors too small by 3: vertex pulls (L/sigma_L) are inflated by about that factor
It also reports the primary-vertex uncertainty, which the vertex fit does not include.

Usage:
    python lc_pull_check.py 'dump_val/csv/*.events.csv' [--files 20]

Needs lc_schema.py and lc_vertex.py (unpack_w) in the same folder.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse
import glob
import sys

import numpy as np
import pandas as pd

from lc_schema import EVENT_COLS, TRACK_COLS
from lc_vertex import DEFAULT_CONV, unpack_w

WCOLS = [f"w{k}" for k in range(1, 16)]
NEED = ["run", "evt", "idx", "q", "lock", "px", "py", "pz", "trklen", "ip_pv", "zip_pv", "eerr", "vd_rphi",
        "vd_rz", "d0", "z0", "phi", "ip_pv", "zip_pv"] + WCOLS


def core_width(x):
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("--files", type=int, default=20)
    a = ap.parse_args()
    files = sorted(glob.glob(a.events))[:a.files]
    if not files:
        sys.exit("no files match")
    rows = []
    for f in files:
        try:
            ev = pd.read_csv(f, names=EVENT_COLS)
            tr = pd.read_csv(f.replace(".events.csv", ".tracks.csv"), names=TRACK_COLS, usecols=NEED)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            continue
        ev = ev[ev.ihad4 == 1]
        tr = tr.merge(ev[["run", "evt", "pv_x", "pv_y", "pv_z", "pv_xx", "pv_xy", "pv_yy", "pv_xz", "pv_yz", "pv_zz"]], on=["run", "evt"])
        p = np.sqrt(tr.px ** 2 + tr.py ** 2 + tr.pz ** 2)
        th = np.degrees(np.arccos((tr.pz / p).clip(-1, 1)))
        ok = ((p > 1.0) & (th > 25) & (th < 155) & (tr.lock == 0) & (tr.eerr < 1) & (tr.vd_rphi > 0) &
              (tr.trklen.abs() > 30) & (tr.ip_pv != -999) & (tr.zip_pv != -999))
        tr = tr[ok].assign(p=p[ok])
        if len(tr):
            rows.append(tr)
    if not rows:
        sys.exit("no tracks")
    t = pd.concat(rows, ignore_index=True)
    W = unpack_w(t[WCOLS].to_numpy())
    C = np.linalg.inv(W)
    sd = np.sqrt(C[:, 0, 0])
    sz = np.sqrt(C[:, 1, 1])
    phi = np.arctan2(t.py, t.px).to_numpy()
    nx, ny = -np.sin(phi), np.cos(phi)
    var_pv_d = nx ** 2 * t.pv_xx + 2 * nx * ny * t.pv_xy + ny ** 2 * t.pv_yy
    spv_d = np.sqrt(np.maximum(var_pv_d.to_numpy(), 0))
    spv_z = np.sqrt(np.maximum(t.pv_zz.to_numpy(), 0))
    # d0 and z0 are measured from the origin; ip_pv / zip_pv are the primary-vertex-referenced impact parameters
    d_pull = t.ip_pv.to_numpy() / sd
    d_pull_tot = t.ip_pv.to_numpy() / np.sqrt(sd ** 2 + spv_d ** 2)
    zres = t.zip_pv.to_numpy()
    z_pull = zres / sz
    z_pull_tot = zres / np.sqrt(sz ** 2 + spv_z ** 2)
    use = np.isfinite(d_pull) & np.isfinite(z_pull) & (np.abs(zres) < 1.0)    # tracks consistent with the PV in z

    print(f"tracks used: {int(use.sum())}  (p>1 GeV, VD R-phi hit, unlocked, 25<theta<155 deg, |zip_pv|<1 cm)")
    q = lambda x: " ".join(f"{v:.4f}" for v in np.nanquantile(x, [0.1, 0.5, 0.9]))
    print(f"sigma(d0) from the weights [cm] 10/50/90%:      {q(sd[use])}")
    print(f"sigma(z0) from the weights [cm] 10/50/90%:      {q(sz[use])}")
    print(f"PV error along the d0 direction [cm] 10/50/90%: {q(spv_d[use])}")
    print(f"PV error in z [cm] 10/50/90%:                   {q(spv_z[use])}")
    print("\ncore width of the pulls (1.4826*MAD; 1.0 = errors right), and fraction beyond 3 sigma:")
    print("p bin [GeV]     N     d0/sig_trk  d0/sig_tot   z0/sig_trk  z0/sig_tot   frac|d0 pull|>3 (trk)")
    p = t.p.to_numpy()
    for lo, hi in ((1, 2), (2, 5), (5, 50), (1, 50)):
        m = use & (p >= lo) & (p < hi)
        print(f"{lo:>3}-{hi:<8}{int(m.sum()):7d}{core_width(d_pull[m]):12.2f}{core_width(d_pull_tot[m]):12.2f}"
              f"{core_width(z_pull[m]):13.2f}{core_width(z_pull_tot[m]):12.2f}{np.mean(np.abs(d_pull[m]) > 3):16.3f}")
    print("\nIf the d0 core width is well above 1 (especially at low p), the fit's L/sigma_L is too large by about that factor.")


if __name__ == "__main__":
    main()
