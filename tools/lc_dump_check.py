#!/usr/bin/env python3
"""
lc_dump_check.py - sanity checks of the CSVs written by `delphi_dump.sh -m lc`.

Usage:
    python lc_dump_check.py 'dump_lc_test/csv/*.events.csv' 'dump_lc_test/csv/*.tracks.csv' [--out check]

It prints summaries of the event and track columns (so we can see what the
tags and matrices really contain) and runs physics checks that need no PID and no vertex fit:
  A) K0S -> pi+ pi- from displaced pairs (sentinel impact parameters excluded)
  B) K0S -> pi+ pi- from pairs where both tracks are locked by SKELANA as
     'charged outgoing' (lock bit 2 = secondary-vertex daughters)
  C) K0S -> pi+ pi- from all small-angle opposite-sign pairs (no IP cut)
  D) Lambda -> p pi from the lock-bit-2 pairs
A clear peak at 0.4976 (K0S) / 1.1157 GeV (Lambda) means momenta and charges are sane.
Needs lc_schema.py (same folder). Writes <out>.png and <out>_summary.txt.
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

MPI = 0.13957
MK0 = 0.49761


def read_all(pattern, names, nrows=None):
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no files match {pattern}")
    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f, names=names, nrows=nrows))
        except pd.errors.EmptyDataError:
            pass
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("tracks")
    ap.add_argument("--out", default="check")
    ap.add_argument("--max-events", type=int, default=4000, help="events used for the K0S pair check")
    a = ap.parse_args()

    out = open(f"{a.out}_summary.txt", "w")

    def say(*x):
        s = " ".join(str(i) for i in x)
        print(s)
        out.write(s + "\n")

    pd.set_option("display.width", 200, "display.max_columns", 60, "display.max_rows", 100)
    ev = read_all(a.events, EVENT_COLS)
    tr = read_all(a.tracks, TRACK_COLS)
    say(f"events: {len(ev)}   track rows: {len(tr)}   tracks/event: {len(tr) / max(len(ev), 1):.1f}")
    bad = tr[TRACK_COLS[1:]].isna().any(axis=1).sum()
    say(f"rows with NaN fields: {bad}")

    say("\n== events ==")
    say("ihad4 values:", ev.ihad4.value_counts().to_dict())
    say("nwrit == tracks written per event:", int((ev.groupby(["run", "evt"]).nwrit.sum().sum()) == len(tr)))
    ev["ech_frac"] = ev.echar / ev.ecm.where(ev.ecm > 1, 91.2)
    say(ev[["nctr4", "nctrk", "ecm", "echar", "ech_frac", "pv_x", "pv_y", "pv_z", "pv_chi2",
            "bs_x", "bs_y", "bs_z", "bs_sx", "bs_sy", "bs_sz",
            "btn3", "btp3", "bts3"]].describe().T[["mean", "std", "min", "50%", "max"]].round(4))

    say("\n== tracks ==")
    tr["p"] = np.sqrt(tr.px ** 2 + tr.py ** 2 + tr.pz ** 2)
    tr["pt"] = np.hypot(tr.px, tr.py)
    tr["cost"] = tr.pz / tr.p
    say("charge:", tr.q.value_counts().to_dict())
    say("lock  :", tr.lock.value_counts().to_dict(), " (1=failed selection, 2=multivertex lock)")
    say("fraction p>=1 GeV: %.3f   p>=2 GeV: %.3f" % ((tr.p >= 1).mean(), (tr.p >= 2).mean()))
    say("mass column (VECP(5)) values:", tr.m.round(4).value_counts().head(3).to_dict())
    say("E^2-p^2 vs m^2 max deviation: %.2e" % np.abs(tr.E ** 2 - tr.p ** 2 - tr.m ** 2).max())
    say("VD hits: frac with R-phi hits %.3f, with R-z hits %.3f, with any %.3f" %
        ((tr.vd_rphi > 0).mean(), (tr.vd_rz > 0).mean(), ((tr.vd_rphi > 0) | (tr.vd_rz > 0)).mean()))
    cov = [f"w{k}" for k in range(1, 16)]
    say("weight-matrix elements, fraction nonzero:", {c: round(float((tr[c] != 0).mean()), 3) for c in cov})
    say(tr[["d0", "z0", "theta", "phi", "curv", "trklen", "chi2_novd", "chi2_vd", "ip_pv", "zip_pv",
            "ip_bs", "eerr", "dedx", "dedx_w", "xkaon", "xproton"]].describe().T[["mean", "std", "min", "50%", "max"]].round(4))
    say("detmask values (top):", tr.detmask.value_counts().head(6).to_dict())

    say("\n== PID tags for tracks with p >= 1 GeV ==")
    t1 = tr[tr.p >= 1]
    for c in ["h_isign", "h_dedx_k", "h_dedx_p", "h_rich_k", "h_rich_p", "h_rich_pi", "rich_flag", "dedx_nw"]:
        vc = t1[c].value_counts().sort_index()
        say(f"{c:10s}", vc.head(12).to_dict())

    say("\n== sentinels and sign conventions ==")
    say("trklen: fraction negative %.3f (SKELANA: negative = not in PV fit); |trklen| median %.1f cm" %
        ((tr.trklen < 0).mean(), tr.trklen.abs().median()))
    for c in ["ip_pv", "zip_pv", "ip_bs"]:
        say(f"{c}: fraction == -999 (undefined): {(tr[c] == -999).mean():.3f}   by lock:",
            {int(k): round(float(v), 3) for k, v in (tr[c] == -999).groupby(tr.lock).mean().items()})

    say("\n== paper-style track cuts (p 0.4-50, 20<theta<160 deg, |trklen|>30 cm, |d0|<4, |z0|<10, eerr<1) ==")
    th = np.degrees(np.arccos(tr.cost.clip(-1, 1)))
    ok_ip = (tr.ip_pv != -999) & (tr.zip_pv != -999)
    base = (tr.p > 0.4) & (tr.p < 50) & (th > 20) & (th < 160) & (tr.trklen.abs() > 30) & (tr.eerr < 1)
    sel_pv = base & ok_ip & (tr.ip_pv.abs() < 4) & (tr.zip_pv.abs() < 10)     # w.r.t. primary vertex
    sel_or = base & (tr.d0.abs() < 4) & (tr.z0.abs() < 10)                    # w.r.t. detector origin
    say("pass (IPs w.r.t. PV): %.3f   pass (d0,z0 as exported): %.3f   of the PV-pass rows lock==0: %.3f" %
        (sel_pv.mean(), sel_or.mean(), (tr[sel_pv].lock == 0).mean()))

    # ---- resonance checks without PID or vertex fit
    say("\n== resonance checks (no PID, no vertex fit) ==")
    keys = ev[["run", "evt"]].drop_duplicates().head(a.max_events)
    base_d = tr.merge(keys, on=["run", "evt"])
    cols = ["run", "evt", "idx", "q", "px", "py", "pz", "p", "ip_pv", "lock"]

    def make_pairs(d):
        pr = d[cols].merge(d[cols], on=["run", "evt"], suffixes=("1", "2"))
        pr = pr[(pr.q1 > 0) & (pr.q2 < 0)].copy()
        pr["cosa"] = (pr.px1 * pr.px2 + pr.py1 * pr.py2 + pr.pz1 * pr.pz2) / (pr.p1 * pr.p2)
        return pr

    def mass_of(pr, m1, m2, swap_by_p=False):
        if swap_by_p:      # heavier hypothesis (proton) goes to the higher-momentum track
            hi = pr.p1 >= pr.p2
            ma = np.where(hi, m1, m2); mb = np.where(hi, m2, m1)
        else:
            ma, mb = m1, m2
        e1 = np.sqrt(pr.p1 ** 2 + ma ** 2); e2 = np.sqrt(pr.p2 ** 2 + mb ** 2)
        px, py, pz = pr.px1 + pr.px2, pr.py1 + pr.py2, pr.pz1 + pr.pz2
        return np.sqrt(np.maximum((e1 + e2) ** 2 - px ** 2 - py ** 2 - pz ** 2, 0))

    def excess(m, name, centre, half):
        """signal window centre +- half; sidebands 1.5*half..3*half on both sides (same total width)"""
        d = np.abs(np.asarray(m) - centre)
        sig = int((d < half).sum())
        sb = int(((d > 1.5 * half) & (d < 3 * half)).sum()) * (2 * half) / (2 * 1.5 * half)
        err = np.sqrt(sig + sb * (2 * half) / (2 * 1.5 * half) + 1e-9)
        say(f"{name}: signal window {centre - half:.4f}-{centre + half:.4f}: {sig}; sideband estimate: {sb:.1f}; "
            f"excess: {sig - sb:.1f} +- {err:.1f}")

    def table(m, lo, hi, nb, name):
        h, e = np.histogram(m, bins=nb, range=(lo, hi))
        say(f"{name} counts per {1000 * (hi - lo) / nb:.0f} MeV bin, {lo:.3f}-{hi:.3f} GeV:")
        say("   " + " ".join(f"{int(x)}" for x in h))

    sel_all = base_d[(base_d.p > 0.5) & (base_d.q != 0)]
    pa = make_pairs(sel_all[(sel_all.ip_pv != -999) & (sel_all.ip_pv.abs() > 0.3)])
    mA = mass_of(pa[pa.cosa > 0.8], MPI, MPI); mA = mA[(mA > 0.40) & (mA < 0.60)]
    lk = sel_all[(sel_all.lock.astype(int) & 2) != 0]
    pb = make_pairs(lk)
    mB = mass_of(pb, MPI, MPI); mB = mB[(mB > 0.40) & (mB < 0.60)]
    pc = make_pairs(sel_all[sel_all.p > 0.8])
    mC = mass_of(pc[pc.cosa > 0.95], MPI, MPI); mC = mC[(mC > 0.40) & (mC < 0.60)]
    pd_ = pb[pb.cosa > 0.9]
    mD = mass_of(pd_, 0.93827, MPI, swap_by_p=True); mD = mD[(mD > 1.07) & (mD < 1.17)]
    excess(mA, "A displaced pairs        ", 0.4976, 0.0175)
    excess(mB, "B both lock-bit-2        ", 0.4976, 0.0175)
    excess(mC, "C all, cos>0.95          ", 0.4976, 0.0175)
    excess(mD, "D Lambda->p pi (lock2)   ", 1.1157, 0.0060)
    table(mA, 0.40, 0.60, 20, "A")
    table(mB, 0.40, 0.60, 20, "B")
    table(mC, 0.40, 0.60, 20, "C")
    table(mD, 1.07, 1.17, 20, "D")
    say("PASS if at least one of A-C shows a clearly positive excess with the peak at 0.4976 GeV")
    say("Report the numbers above even if all are negative: it tells us where the problem is.")

    # ---- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 4, figsize=(19, 8))
        for k, (m, ttl, rng_) in enumerate(((mA, "A: displaced pairs", (0.40, 0.60)), (mB, "B: both lock-bit-2", (0.40, 0.60)),
                                            (mC, "C: all, cos>0.95", (0.40, 0.60)), (mD, "D: Lambda -> p pi (lock2)", (1.07, 1.17)))):
            ax[0, k].hist(m, bins=50, range=rng_, histtype="step")
            ax[0, k].axvline(MK0 if k < 3 else 1.1157, color="r", ls="--"); ax[0, k].set_title(ttl)
            ax[0, k].set_xlabel("mass [GeV]")
        ax[1, 0].hist(tr.p, bins=100, range=(0, 30), log=True, histtype="step"); ax[1, 0].set_xlabel("p [GeV]")
        ax[1, 1].hist(tr.cost, bins=50, range=(-1, 1), histtype="step"); ax[1, 1].set_xlabel("cos theta")
        ok = tr.ip_pv != -999
        ax[1, 2].hist(tr.ip_pv[ok].clip(-1, 1), bins=100, histtype="step", log=True); ax[1, 2].set_xlabel("ip_pv [cm] (sentinels removed)")
        ax[1, 3].hist(ev.ech_frac.clip(0, 1.5), bins=60, histtype="step"); ax[1, 3].set_xlabel("E_ch / sqrt(s)")
        fig.tight_layout(); fig.savefig(f"{a.out}.png", dpi=110)
        say(f"wrote {a.out}.png and {a.out}_summary.txt")
    except Exception as exc:  # plotting must never hide the text summary
        say("plot failed:", exc)
    out.close()


if __name__ == "__main__":
    main()
