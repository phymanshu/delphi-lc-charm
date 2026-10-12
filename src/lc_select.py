#!/usr/bin/env python3
"""
lc_select.py - vary the selection offline on the candidate table written by lc_reco.py and look for the
best signal significance, with a hold-out half so the tuning does not bias the result.

The candidate table holds loosely selected candidates (L > 0, P(chi2) > 1e-4, X_E > X_E,min - 0.05, mass window,
PID tag >= the --tag-min used in lc_reco.py) with the PID tags, so every cut below can be tightened here.

Usage:
    python lc_select.py lc_full_candidates.csv.gz --channel lc                    # paper cuts, spectrum + fit
    python lc_select.py lc_full_candidates.csv.gz --channel lc --tag-p 3 --xe-min 0.4 --lmin 0.03
    python lc_select.py lc_full_candidates.csv.gz --channel lc --scan              # grid scan, tuned on even events, shown on odd

Significance = (events in |m - m0| < 25 MeV minus the sideband estimate) / sqrt(window + sideband estimate),
sidebands 50-100 MeV on either side of the peak. The scan tunes on events with even event number and reports the
same selection on the odd ones, which are not used for the choice.
"""
import argparse
import itertools
import sys

import numpy as np
import pandas as pd

from lc_reco import CHANNELS, fit_mass, hel_cut, tag_ok


def apply(df, ch, o):
    """boolean mask of the offline selection o (dict)"""
    c = (df.mass > ch["window"][0]) & (df.mass < ch["window"][1]) & (df.xe > o["xe_min"]) & (df.P > o["pmin"]) & (df.L > o["lmin"])
    if o["xe_slope"] is not None:     # energy-dependent L cut of the paper (D+)
        x, y = o["xe_slope"]
        c &= df.L > x * (df.xe - o["xe_min"]) ** 2 + y
    if o["hel"]:
        c &= hel_cut(df.xe, df.cosh, ch["hel"])
    if ch["dstar_veto"]:
        c &= df.veto
    if ch["roles"][0][3] == "p":        # Lambda_c: proton = role 0 ("p_" columns), kaon = role 1 ("k_" columns)
        c &= tag_ok(df.p_h_rich_p, df.p_h_dedx_p, o["tag_p"]) & tag_ok(df.k_h_rich_k, df.k_h_dedx_k, o["tag_k"])
    else:                               # D+: the kaon is role 0 and is stored with the "k_" prefix
        c &= tag_ok(df.k_h_rich_k, df.k_h_dedx_k, o["tag_k"])
    return c


def fit_poly(m, lo, hi, nb, mass, sigma_fixed=None, half=0.10):
    """Gaussian (integrated over the bins) + quadratic background; the real spectra have broad structure that a
    single exponential cannot follow. Returns dict or a message."""
    try:
        from math import erf
        from scipy.optimize import curve_fit
    except ImportError:
        return None
    bw = (hi - lo) / nb
    lo2, hi2 = max(lo, mass - half), min(hi, mass + half)      # fit only near the peak: the global shape has broad structure
    nb2 = max(int(round((hi2 - lo2) / bw)), 6)
    h, e = np.histogram(m, bins=nb2, range=(lo2, hi2))
    nb = nb2
    verf = np.vectorize(erf)
    x = 0.5 * (e[1:] + e[:-1])
    u = (x - mass) / 0.2

    def model(_, n, mu, sg, a0, a1, a2):
        cdf = 0.5 * (1 + verf((e - mu) / (sg * np.sqrt(2))))
        return n * np.diff(cdf) + a0 + a1 * u + a2 * u ** 2

    if sigma_fixed:
        f = lambda xx, n, mu, a0, a1, a2: model(xx, n, mu, sigma_fixed / 1000.0, a0, a1, a2)
        p0 = [max(h.max() - np.median(h), 1), mass, np.median(h), 0, 0]
        lb = [-1e6, mass - 0.015, 0, -1e6, -1e6]; ub = [1e6, mass + 0.015, 1e7, 1e6, 1e6]
    else:
        f = model
        p0 = [max(h.max() - np.median(h), 1), mass, 0.015, np.median(h), 0, 0]
        lb = [-1e6, mass - 0.015, 0.008, 0, -1e6, -1e6]; ub = [1e6, mass + 0.015, 0.040, 1e7, 1e6, 1e6]
    try:
        popt, pcov = curve_fit(f, x, h, p0=p0, sigma=np.sqrt(np.maximum(h, 1)), maxfev=40000, bounds=(lb, ub))
    except Exception as exc:
        return f"fit failed: {exc}"
    sg = sigma_fixed / 1000.0 if sigma_fixed else popt[2]
    full = popt if not sigma_fixed else np.insert(popt, 2, sg)
    chi2 = float((((h - model(None, *full)) / np.sqrt(np.maximum(h, 1))) ** 2).sum())
    return dict(yield_=popt[0], err=np.sqrt(pcov[0, 0]), mean=popt[1], sigma=sg, chi2=chi2, ndf=nb - len(popt),
                curve=lambda: model(None, *full), x=x, h=h)


def significance(m, m0, win=0.025, sb_lo=0.05, sb_hi=0.10):
    """signal window |m - m0| < win; sidebands sb_lo < |m - m0| < sb_hi on both sides"""
    d = np.abs(np.asarray(m) - m0)
    sig = int((d < win).sum())
    sb = ((d > sb_lo) & (d < sb_hi)).sum() * (win / (sb_hi - sb_lo))
    return sig, sb, sig - sb, (sig - sb) / np.sqrt(max(sig + sb, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates")
    ap.add_argument("--channel", choices=sorted(CHANNELS), default="lc")
    ap.add_argument("--tag-p", type=int, default=1)
    ap.add_argument("--tag-k", type=int, default=1)
    ap.add_argument("--xe-min", type=float, default=None)
    ap.add_argument("--lmin", type=float, default=None)
    ap.add_argument("--pmin", type=float, default=None, help="min P(chi2)")
    ap.add_argument("--no-helicity", action="store_true")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--charge", type=int, choices=[-1, 1], default=None, help="only Lc+ (+1) or Lc- (-1): sign of the proton")
    ap.add_argument("--bins", type=int, default=None, help="number of mass bins (default: the channel default)")
    ap.add_argument("--sigma", type=float, default=None, help="fix the Gaussian width in MeV (default: free, 8-40 MeV)")
    ap.add_argument("--png", default="lc_select_mass.png")
    ap.add_argument("--win", type=float, default=0.025, help="half-width of the signal window [GeV] (the free-width fit gave sigma ~ 9 MeV: try 0.015)")
    ap.add_argument("--sb-lo", type=float, default=0.05, help="inner edge of the sidebands, distance from the centre [GeV]")
    ap.add_argument("--sb-hi", type=float, default=0.10, help="outer edge of the sidebands [GeV]")
    ap.add_argument("--btag-var", choices=["btp3", "bts3", "btn3"], default="btp3", help="event b-tag probability to cut on (btp3 = positive-IP tracks, whole event; small = lifetime-like)")
    ap.add_argument("--btag-lo", type=float, default=None, help="keep events with btag-var >= this")
    ap.add_argument("--btag-hi", type=float, default=None, help="keep events with btag-var < this")
    ap.add_argument("--lhist", action="store_true", help="sideband-subtracted decay-length distribution of the peak")
    ap.add_argument("--mass-shift", type=float, default=0.0, help="centre of the signal window = nominal mass + this shift [GeV] (the fits give about +0.005)")
    ap.add_argument("--fit-half-range", type=float, default=0.10, help="fit window mass +- this many GeV (default 0.10)")
    a = ap.parse_args()
    ch = CHANNELS[a.channel]
    df = pd.read_csv(a.candidates)
    n0 = len(df)
    if not a.no_dedupe:
        df = df.drop_duplicates(["run", "evt", "ctrl", "mass", "xe"])
    print(f"{n0} candidate rows, {len(df)} after removing exact duplicates (same run, event, mass and X_E)")
    base = dict(xe_min=ch["xe_min"] if a.xe_min is None else a.xe_min, lmin=ch["lcut"][2] if a.lmin is None else a.lmin,
                pmin=ch["pcut"] if a.pmin is None else a.pmin, hel=not a.no_helicity,
                xe_slope=(ch["lcut"][0], ch["lcut"][1]) if ch["lcut"][0] is not None else None,
                tag_p=a.tag_p, tag_k=a.tag_k)
    if a.btag_lo is not None or a.btag_hi is not None:
        lo_b = -1.0 if a.btag_lo is None else a.btag_lo
        hi_b = 2.0 if a.btag_hi is None else a.btag_hi
        df = df[(df[a.btag_var] >= lo_b) & (df[a.btag_var] < hi_b)]
        print(f"events with {lo_b} <= {a.btag_var} < {hi_b}: {len(df)} rows")
    if a.charge is not None:
        df = df[df.q == a.charge]
        print(f"only q(proton) = {a.charge:+d}: {len(df)} rows")
    nb = a.bins or ch["nbins"]
    ph, ct = df[df.ctrl == 0], df[df.ctrl == 1]
    if not a.scan:
        mp, mc = apply(ph, ch, base), apply(ct, ch, base)
        lo, hi = ch["window"]
        edges = np.linspace(lo, hi, nb + 1)
        print("selection:", {k: v for k, v in base.items() if k != "xe_slope"})
        print(f"candidates: physical {int(mp.sum())}, control {int(mc.sum())}")
        print("  physical: " + " ".join(str(int(x)) for x in np.histogram(ph[mp].mass, edges)[0]))
        print("  control : " + " ".join(str(int(x)) for x in np.histogram(ct[mc].mass, edges)[0]))
        r = fit_poly(ph[mp].mass.to_numpy(), lo, hi, nb, ch["mass"], a.sigma, a.fit_half_range)
        if isinstance(r, dict):
            print(f"fit in mass +- {a.fit_half_range} GeV (Gaussian + quadratic background): signal yield {r['yield_']:.0f} +- {r['err']:.0f}, mean {r['mean']:.4f} GeV, "
                  f"sigma {r['sigma'] * 1000:.1f} MeV, chi2/ndf {r['chi2']:.0f}/{r['ndf']}")
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7.5, 5))
                ax.errorbar(r["x"], r["h"], yerr=np.sqrt(r["h"]), fmt="ko", ms=3, label="data")
                ax.plot(r["x"], r["curve"](), "r-", label="Gauss + pol2")
                hc = np.histogram(ct[mc].mass, edges)[0]
                ax.stairs(hc, edges, color="tab:orange", label="control")
                ax.set_xlabel("mass [GeV]"); ax.set_ylabel(f"candidates / {(hi - lo) / nb * 1000:.0f} MeV"); ax.legend()
                fig.tight_layout(); fig.savefig(a.png, dpi=120); print("wrote", a.png)
            except Exception as exc:
                print("plot failed:", exc)
        elif isinstance(r, str):
            print(r)
        if a.lhist:
            sel = ph[mp]
            c0 = ch["mass"] + a.mass_shift
            dm = np.abs(sel.mass - c0)
            win, sbd = sel[dm < a.win], sel[(dm > a.sb_lo) & (dm < a.sb_hi)]
            le = np.array([0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0])
            hw = np.histogram(win.L, le)[0]
            hs = np.histogram(sbd.L, le)[0] * (a.win / (a.sb_hi - a.sb_lo))
            diff = hw - hs
            err = np.sqrt(hw + hs * (a.win / (a.sb_hi - a.sb_lo)))
            print(f"decay length L [cm] of the peak (window {c0 - a.win:.4f}-{c0 + a.win:.4f}, sidebands {a.sb_lo}-{a.sb_hi} GeV from the centre) minus sidebands (scaled), selection as above:")
            print("  L bin edges:     " + " ".join(f"{x:5.2f}" for x in le))
            print("  window:          " + " ".join(f"{int(x):5d}" for x in hw))
            print("  sidebands scaled:" + " ".join(f"{x:5.0f}" for x in hs))
            print("  signal:          " + " ".join(f"{x:5.0f}" for x in diff))
            print("  +-               " + " ".join(f"{x:5.0f}" for x in err))
            mid = 0.5 * (le[1:] + le[:-1])
            if diff.sum() > 0:
                print(f"  mean L of the signal (bin midpoints, truncated at 1 cm): {np.sum(diff * mid) / diff.sum():.3f} cm")
        sig, sb, ex, z = significance(ph[mp].mass, ch["mass"] + a.mass_shift, a.win, a.sb_lo, a.sb_hi)
        print(f"window {sig}, sideband estimate {sb:.1f}, excess {ex:.1f}, significance {z:.1f}")
        return
    # ---- grid scan: tune on even events, report on odd events
    tune, test = ph[ph.evt % 2 == 0], ph[ph.evt % 2 == 1]
    grid = dict(tag_p=[1, 2, 3], tag_k=[1, 2, 3], xe_min=[0.3, 0.35, 0.4, 0.5, 0.6], lmin=[0.015, 0.03, 0.05, 0.08],
                pmin=[0.001, 0.01, 0.05], hel=[True, False])
    if ch["roles"][0][3] != "p":
        grid["tag_p"] = [1]
    rows = []
    for vals in itertools.product(*grid.values()):
        o = dict(base)
        o.update(dict(zip(grid.keys(), vals)))
        mt = apply(tune, ch, o)
        if mt.sum() < 20:
            continue
        s_t = significance(tune[mt].mass, ch["mass"] + a.mass_shift, a.win, a.sb_lo, a.sb_hi)
        mo = apply(test, ch, o)
        s_o = significance(test[mo].mass, ch["mass"] + a.mass_shift, a.win, a.sb_lo, a.sb_hi)
        rows.append(dict(zip(grid.keys(), vals), tune_sig=s_t[3], tune_excess=s_t[2], test_sig=s_o[3], test_excess=s_o[2], test_window=s_o[0]))
    if not rows:
        sys.exit("no selection left")
    res = pd.DataFrame(rows).sort_values("tune_sig", ascending=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\nbest selections on the tuning half (even events), with the same cuts on the held-out odd events:")
    print(res.head(15).round(2).to_string(index=False))
    print("\nIf the held-out significance is much smaller than the tuned one, the gain is mostly a tuning fluctuation.")
    res.to_csv("lc_scan.csv", index=False)


if __name__ == "__main__":
    main()
