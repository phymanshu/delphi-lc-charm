#!/usr/bin/env python3
"""
lc_dstar_check.py - D*+ -> D0 pi+, D0 -> K- pi+ validation of the lc dump.

No PID and no vertex fit are needed: the paper itself reconstructs this channel
without kaon identification (the D0 mass window removes most background), and
the D*+ - D0 mass difference peaks very narrowly at 145.4 MeV. Charm hadrons
fly only ~1 mm, so the momenta at the primary-vertex perigee are good enough
(unlike K0S/Lambda, which fly cm and get smeared).

Usage:
    python lc_dstar_check.py 'dump_val/csv/*.events.csv' --out dstar [--xe-min 0.4]

The matching tracks files are found by replacing '.events.csv' with '.tracks.csv'.
You need roughly 50+ files of the dump (with -p 0.4) for a clear peak: a single
file only has a handful of D*+.

Selection (from the DELPHI charm paper, Eur.Phys.J. C12 (2000) 225):
  hadronic event: ihad4 == 1 and charged energy > 12% of sqrt(s)
  tracks: 0.4 < p < 50 GeV, 20 < theta < 160 deg, |trklen| > 30 cm,
          |ip_pv| < 4 cm, |zip_pv| < 10 cm, relative error (eerr) < 1
  K and pi from the D0: p > 1 GeV; slow pion: p > 0.4 GeV
  D0 mass 1.79-1.94 GeV, X_E(D0) > --xe0 and X_E(D*) > --xe-min
Right-sign = slow pion has the charge of the D0's pion; wrong-sign (same charge
as the kaon) is the background control. NOTE: a true D* with its K and pi swapped
also lands in the wrong-sign sample as a smeared peak near 145 MeV, so the
right-minus-wrong excess is conservative; the right-sign peak must still stand
out clearly above its own neighbouring bins.
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

MPI, MK, DM0 = 0.13957, 0.49368, 0.14543
NEED = ["run", "evt", "idx", "q", "lock", "px", "py", "pz", "trklen", "ip_pv", "zip_pv", "eerr"]


def prep_tracks(tr):
    tr = tr.copy()
    tr["p"] = np.sqrt(tr.px ** 2 + tr.py ** 2 + tr.pz ** 2)
    cost = tr.pz / tr.p
    th = np.degrees(np.arccos(cost.clip(-1, 1)))
    ok = ((tr.p > 0.4) & (tr.p < 50) & (th > 20) & (th < 160) & (tr.trklen.abs() > 30) &
          (tr.ip_pv != -999) & (tr.zip_pv != -999) & (tr.ip_pv.abs() < 4) & (tr.zip_pv.abs() < 10) &
          (tr.eerr < 1) & (tr.q != 0))
    return tr[ok]


def process(ev_file, args, hist_rs, hist_ws, edges):
    tr_file = ev_file.replace(".events.csv", ".tracks.csv")
    try:
        ev = pd.read_csv(ev_file, names=EVENT_COLS)
        tr = pd.read_csv(tr_file, names=TRACK_COLS, usecols=NEED)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return 0
    ecm = ev.ecm.where(ev.ecm > 1, 91.2)
    ev = ev[(ev.ihad4 == 1) & (ev.echar / ecm > 0.12)].assign(ecm=ecm)
    if ev.empty or tr.empty:
        return 0
    tr = prep_tracks(tr).merge(ev[["run", "evt", "ecm"]], on=["run", "evt"])
    kp = tr[tr.p > 1.0]
    c = ["run", "evt", "idx", "q", "px", "py", "pz", "p", "ecm"]
    pr = kp[c].merge(kp[c[:-1]], on=["run", "evt"], suffixes=("_a", "_b"))
    pr = pr[(pr.q_a < 0) & (pr.q_b > 0)]           # a = negative track, b = positive track
    for hyp in (1, 2):
        # hyp 1: K = negative (a), pi = positive (b), slow pion must be + (D*+ -> D0 pi+)
        # hyp 2: K = positive (b), pi = negative (a), slow pion must be - (D*- -> D0bar pi-)
        mk_k, mk_pi = (("a", "b") if hyp == 1 else ("b", "a"))
        ek = np.sqrt(pr[f"p_{mk_k}"] ** 2 + MK ** 2)
        epi = np.sqrt(pr[f"p_{mk_pi}"] ** 2 + MPI ** 2)
        px = pr.px_a + pr.px_b; py = pr.py_a + pr.py_b; pz = pr.pz_a + pr.pz_b
        e = ek + epi
        m = np.sqrt(np.maximum(e ** 2 - px ** 2 - py ** 2 - pz ** 2, 0))
        xe = 2 * e / pr.ecm
        keep = (m > 1.79) & (m < 1.94) & (xe > args.xe0)
        d0 = pr[keep].assign(m0=m[keep], e0=e[keep], sx=px[keep], sy=py[keep], sz=pz[keep])
        if d0.empty:
            continue
        q_pi = 1 if hyp == 1 else -1
        slow = tr[(tr.p > 0.4)][["run", "evt", "idx", "q", "px", "py", "pz", "p"]]
        slow = slow.rename(columns={c_: c_ + "_s" for c_ in ["idx", "q", "px", "py", "pz", "p"]})
        cand = d0.merge(slow, on=["run", "evt"])
        cand = cand[(cand.idx_s != cand.idx_a) & (cand.idx_s != cand.idx_b)]
        es = np.sqrt(cand.p_s ** 2 + MPI ** 2)
        et = cand.e0 + es
        mt = np.sqrt(np.maximum(et ** 2 - (cand.sx + cand.px_s) ** 2 - (cand.sy + cand.py_s) ** 2 - (cand.sz + cand.pz_s) ** 2, 0))
        dm = mt - cand.m0
        good = (2 * et / cand.ecm > args.xe_min) & (dm > edges[0]) & (dm < edges[-1])
        rs = dm[good & (cand.q_s == q_pi)]
        ws = dm[good & (cand.q_s == -q_pi)]
        hist_rs += np.histogram(rs, edges)[0]
        hist_ws += np.histogram(ws, edges)[0]
    return len(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", help="glob of *.events.csv files")
    ap.add_argument("--out", default="dstar")
    ap.add_argument("--xe0", type=float, default=0.3, help="min X_E of the D0")
    ap.add_argument("--xe-min", type=float, default=0.4, help="min X_E of the D*")
    ap.add_argument("--files", type=int, default=0, help="use only the first N files")
    args = ap.parse_args()
    files = sorted(glob.glob(args.events))
    if args.files:
        files = files[:args.files]
    if not files:
        sys.exit("no files match")
    edges = np.arange(0.135, 0.1751, 0.001)
    rs = np.zeros(len(edges) - 1); ws = np.zeros(len(edges) - 1)
    nev = 0
    for i, f in enumerate(files, 1):
        nev += process(f, args, rs, ws, edges)
        if i % 10 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)} files, hadronic events so far: {nev}", flush=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    win = np.abs(centres - DM0) < 0.0015
    s, b = rs[win].sum(), ws[win].sum()
    lines = [f"files: {len(files)}   hadronic events used: {nev}",
             f"cuts: X_E(D0) > {args.xe0}, X_E(D*) > {args.xe_min}",
             "dM bin starts at 0.135 GeV, 1 MeV bins; right-sign counts:", "  " + " ".join(str(int(x)) for x in rs),
             "wrong-sign counts:", "  " + " ".join(str(int(x)) for x in ws),
             f"window |dM - {DM0}| < 1.5 MeV: right-sign {int(s)}, wrong-sign {int(b)}, "
             f"excess {s - b:.0f} +- {np.sqrt(s + b):.1f}  ({(s - b) / max(np.sqrt(s + b), 1e-9):.1f} sigma)",
             "PASS if the excess is several sigma and the right-sign peak sits at 145.4 MeV (bin 0.145-0.146 is the 11th number)"]
    txt = "\n".join(lines)
    print(txt)
    open(f"{args.out}_summary.txt", "w").write(txt + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.stairs(rs, edges, label="right-sign"); ax.stairs(ws, edges, label="wrong-sign")
        ax.axvline(DM0, color="r", ls="--", lw=0.8)
        ax.set_xlabel(r"M(K pi pi_s) - M(K pi)  [GeV]"); ax.set_ylabel("candidates / MeV"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{args.out}.png", dpi=110)
        print(f"wrote {args.out}.png and {args.out}_summary.txt")
    except Exception as exc:
        print("plot failed:", exc)


if __name__ == "__main__":
    main()
