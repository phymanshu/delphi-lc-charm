#!/usr/bin/env python3
"""
lc_vertex_scan.py - fix the track-parameter conventions of the dump with D*+ -> D0 pi+ tagged
D0 -> K- pi+ candidates, and test the vertex fit (lc_vertex.py) on them.

Why D0 from D*: the D*+ tag (right-sign, |dM - 145.4 MeV| < window) gives a sample that is
mostly real D0, which come from a ~1 mm displaced vertex. With the right conventions the
2-track vertex fit gives chi2 ~ chi2(ndf=1) (median 0.45, 95% below 3.84), a flat P(chi2) for
the signal, a positive decay length and a vertex z close to the primary vertex z.

The wrong-sign slow-pion sample (same window) is fitted too and used as a background control:
right-minus-wrong decay-length counts show the real-D0 lifetime distribution.

Usage:
    python lc_vertex_scan.py 'dump_val/csv/*.events.csv' [--files 50] [--window 0.002]

Needs lc_schema.py and lc_vertex.py in the same folder. Prints a table over the conventions
(sd = sign of d0, sk = sign of curvature, z0 relative/absolute) and the diagnostics for the best one.
What to expect: sd is decided by chi2; the z convention by median |v_z - pv_z| (a z shift common
to both tracks is absorbed by the vertex z, so chi2 cannot tell it); sk has almost no effect on
mm-long flights, so it is irrelevant for charm and the table will show no preference.
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
from lc_vertex import ALL_CONVENTIONS, DEFAULT_ERR_SCALE, chi2_sf, decay_length, fit_vertex, unpack_w

MPI, MK, DM0 = 0.13957, 0.49368, 0.14543
WCOLS = [f"w{k}" for k in range(1, 16)]
PCOLS = ["d0", "z0", "theta", "phi", "curv"]
NEED = ["run", "evt", "idx", "q", "lock", "px", "py", "pz", "trklen", "ip_pv", "zip_pv", "eerr", "vd_rphi", "ip_bs"] + PCOLS + WCOLS


def good_tracks(tr):
    tr = tr.copy()
    tr["p"] = np.sqrt(tr.px ** 2 + tr.py ** 2 + tr.pz ** 2)
    th = np.degrees(np.arccos((tr.pz / tr.p).clip(-1, 1)))
    ok = ((tr.p > 0.4) & (tr.p < 50) & (th > 20) & (th < 160) & (tr.trklen.abs() > 30) &
          (tr.ip_pv != -999) & (tr.zip_pv != -999) & (tr.ip_pv.abs() < 4) & (tr.zip_pv.abs() < 10) &
          (tr.eerr < 1) & (tr.q != 0))
    return tr[ok]


def collect_file(ev_file, args):
    tr_file = ev_file.replace(".events.csv", ".tracks.csv")
    try:
        ev = pd.read_csv(ev_file, names=EVENT_COLS)
        tr = pd.read_csv(tr_file, names=TRACK_COLS, usecols=NEED)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return None, None
    ecm = ev.ecm.where(ev.ecm > 1, 91.2)
    ev = ev[(ev.ihad4 == 1) & (ev.echar / ecm > 0.12)].assign(ecm=ecm)
    if ev.empty or tr.empty:
        return None, None
    tr = good_tracks(tr).merge(ev[["run", "evt", "ecm", "pv_x", "pv_y", "pv_z", "bs_x", "bs_y", "bs_z"]], on=["run", "evt"])
    kp = tr[tr.p > 1.0]
    c = ["run", "evt", "idx", "q", "px", "py", "pz", "p", "ecm"]
    pr = kp[c].merge(kp[c[:-1]], on=["run", "evt"], suffixes=("_a", "_b"))
    pr = pr[(pr.q_a < 0) & (pr.q_b > 0)]
    rows = []
    for hyp in (1, 2):
        k, pi_ = ("a", "b") if hyp == 1 else ("b", "a")
        ek = np.sqrt(pr[f"p_{k}"] ** 2 + MK ** 2)
        epi = np.sqrt(pr[f"p_{pi_}"] ** 2 + MPI ** 2)
        px, py, pz = pr.px_a + pr.px_b, pr.py_a + pr.py_b, pr.pz_a + pr.pz_b
        e = ek + epi
        m = np.sqrt(np.maximum(e ** 2 - px ** 2 - py ** 2 - pz ** 2, 0))
        keep = (m > 1.79) & (m < 1.94) & (2 * e / pr.ecm > args.xe0)
        d0 = pr[keep].assign(m0=m[keep], e0=e[keep], sx=px[keep], sy=py[keep], sz=pz[keep], hyp=hyp,
                             idx_k=pr[keep][f"idx_{k}"], idx_pi=pr[keep][f"idx_{pi_}"])
        if d0.empty:
            continue
        slow = tr[tr.p > 0.4][["run", "evt", "idx", "q", "px", "py", "pz", "p"]]
        slow = slow.rename(columns={x: x + "_s" for x in ["idx", "q", "px", "py", "pz", "p"]})
        cand = d0.merge(slow, on=["run", "evt"])
        cand = cand[(cand.idx_s != cand.idx_a) & (cand.idx_s != cand.idx_b)]
        es = np.sqrt(cand.p_s ** 2 + MPI ** 2)
        et = cand.e0 + es
        mt = np.sqrt(np.maximum(et ** 2 - (cand.sx + cand.px_s) ** 2 - (cand.sy + cand.py_s) ** 2 -
                                (cand.sz + cand.pz_s) ** 2, 0))
        cand = cand.assign(dm=mt - cand.m0)
        q_pi = 1 if hyp == 1 else -1
        cand = cand[np.abs(cand.dm - DM0) < args.window]
        cand = cand[2 * (cand.e0 + np.sqrt(cand.p_s ** 2 + MPI ** 2)) / cand.ecm > args.xe_min]
        cand = cand[(cand.q_s == q_pi) | (cand.q_s == -q_pi)].assign(ws=lambda d: (d.q_s != q_pi).astype(int))
        if not cand.empty:
            rows.append(cand[["run", "evt", "idx_k", "idx_pi", "hyp", "dm", "m0", "sx", "sy", "sz", "ws"]])
    if not rows:
        return None, None
    cd = pd.concat(rows, ignore_index=True).drop_duplicates(["run", "evt", "idx_k", "idx_pi", "ws"])
    ti = tr.set_index(["run", "evt", "idx"])
    evi = ev.set_index(["run", "evt"])
    out = {}
    for name in ("k", "pi"):
        mi = pd.MultiIndex.from_arrays([cd.run, cd.evt, cd[f"idx_{name}"]])
        t = ti.reindex(mi)
        for col in PCOLS + WCOLS + ["px", "py", "pz", "q", "vd_rphi", "ip_pv", "ip_bs"]:
            out[f"{name}_{col}"] = t[col].to_numpy()
    mi = pd.MultiIndex.from_arrays([cd.run, cd.evt])
    for col in ("pv_x", "pv_y", "pv_z", "bs_x", "bs_y", "bs_z"):
        out[col] = evi.reindex(mi)[col].to_numpy()
    for col in ("sx", "sy", "sz", "dm", "m0", "hyp", "ws"):
        out[col] = cd[col].to_numpy()
    return pd.DataFrame(out), len(ev)


def build_arrays(cd, angles_from_momentum=True):
    pars, ws = [], []
    for name in ("k", "pi"):
        p = np.stack([cd[f"{name}_{c}"] for c in PCOLS], axis=1).astype(float)
        if angles_from_momentum:
            px, py, pz = cd[f"{name}_px"].to_numpy(), cd[f"{name}_py"].to_numpy(), cd[f"{name}_pz"].to_numpy()
            pp = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
            p[:, 2] = np.arccos(np.clip(pz / pp, -1, 1))
            p[:, 3] = np.arctan2(py, px)
        pars.append(p)
        ws.append(unpack_w(cd[[f"{name}_{c}" for c in WCOLS]].to_numpy()))
    ref = cd[["pv_x", "pv_y", "pv_z"]].to_numpy(float)
    mom = cd[["sx", "sy", "sz"]].to_numpy(float)
    return np.stack(pars, axis=1), np.stack(ws, axis=1), ref, mom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("--files", type=int, default=0)
    ap.add_argument("--window", type=float, default=0.002, help="|dM - 145.4 MeV| window")
    ap.add_argument("--xe0", type=float, default=0.3)
    ap.add_argument("--xe-min", type=float, default=0.4)
    ap.add_argument("--out", default="vertex_scan")
    ap.add_argument("--err-scale", type=float, default=DEFAULT_ERR_SCALE, help="multiply the track errors (pull-check core width ~1.4)")
    ap.add_argument("--vd-min", type=int, default=1, help="min exported vd_rphi value for both tracks (paper: at least one VD hit); 0 = off")
    ap.add_argument("--d0-halfwidth", type=float, default=0.03, help="|M(K pi) - 1.8648| window [GeV]")
    ap.add_argument("--exported-angles", action="store_true", help="use exported theta/phi instead of the momentum angles")
    a = ap.parse_args()
    files = sorted(glob.glob(a.events))
    if a.files:
        files = files[:a.files]
    if not files:
        sys.exit("no files match")
    parts = []
    for i, f in enumerate(files, 1):
        cd, _ = collect_file(f, a)
        if cd is not None:
            parts.append(cd)
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)} files, D* candidates so far: {sum(len(p) for p in parts)}", flush=True)
    if not parts:
        sys.exit("no D* candidates found")
    cd = pd.concat(parts, ignore_index=True)
    cd = cd[np.isfinite(cd[[c for c in cd.columns if c.startswith(("k_", "pi_"))]].to_numpy()).all(axis=1)].reset_index(drop=True)
    n0 = len(cd)
    cd = cd[(np.abs(cd.m0 - 1.8648) < a.d0_halfwidth) & (cd.k_vd_rphi >= a.vd_min) & (cd.pi_vd_rphi >= a.vd_min)].reset_index(drop=True)
    print(f"candidates after |M(Kpi)-1.8648|<{a.d0_halfwidth} and vd_rphi>={a.vd_min} for both tracks: {len(cd)} of {n0}")
    if cd.empty:
        sys.exit("no candidates left after the cuts")
    par, W, ref, mom = build_arrays(cd, not a.exported_angles)
    pv = ref.copy()                 # event primary vertex: decay lengths are measured from here
    ref = np.zeros_like(pv)         # the exported d0 and z0 are measured from the ORIGIN (found with the reference test), not from the PV
    ws = cd.ws.to_numpy().astype(bool)
    out = []

    def say(*x):
        s = " ".join(str(i) for i in x)
        print(s)
        out.append(s)

    say(f"track errors scaled by {a.err_scale}")
    say(f"\nD* tagged D0 candidates: {int((~ws).sum())} right-sign, {int(ws.sum())} wrong-sign  (window +-{a.window * 1000:.1f} MeV around 145.4, X_E(D0)>{a.xe0}, X_E(D*)>{a.xe_min})")

    # ---- quick diagnostics of the conventions that do not need a fit
    for name in ("k", "pi"):
        q = cd[f"{name}_q"].to_numpy()
        kap = cd[f"{name}_curv"].to_numpy()
        phi_e = cd[f"{name}_phi"].to_numpy()
        phi_p = np.arctan2(cd[f"{name}_py"], cd[f"{name}_px"]).to_numpy()
        dphi = np.abs(((phi_e - phi_p + np.pi) % (2 * np.pi)) - np.pi)
        pt = np.hypot(cd[f"{name}_px"], cd[f"{name}_py"]).to_numpy()
        say(f"{name}: sign(curv)==q for {np.mean(np.sign(kap) == q):.3f}; exported phi == momentum phi (mod 2pi, <1e-3) for "
            f"{np.mean(dphi < 1e-3):.3f}; median |curv|*pT = {np.median(np.abs(kap) * pt):.3g} (1/cm*GeV; 3.7e-3 expected for B=1.23 T in cm^-1)")

    # ---- scan
    say("\nconvention                          median chi2  frac<3.84  median |vz-pvz| [cm]  median Lxy [cm]  frac Lxy>0")
    best = None
    for c in ALL_CONVENTIONS:
        v, chi2, _ = fit_vertex(par[~ws], W[~ws], ref[~ws], c, err_scale=a.err_scale)
        ok = np.isfinite(chi2)
        L = decay_length(v, pv[~ws], mom[~ws])
        frac = float(np.mean(chi2[ok] < 3.84)) if ok.any() else 0.0
        dz = float(np.nanmedian(np.abs(v[:, 2] - pv[~ws][:, 2])))
        say(f"{str(c):36s}{np.nanmedian(chi2):10.3g}{frac:11.3f}{dz:19.3g}{np.nanmedian(L):17.3g}{np.mean(L[ok] > 0):12.3f}")
        key = (round(frac, 2), round(-dz, 1))      # coarse, so ties keep the first (sk=+1) entry
        if c.sk == +1 and (best is None or key > best[0]):     # sk has no effect on mm-long flights: report sk=+1
            best = (key, c)
    _, c = best
    say(f"\nbest convention: {c}")
    v, chi2, Cv = fit_vertex(par, W, ref, c, err_scale=a.err_scale)
    L = decay_length(v, pv, mom)
    ok = np.isfinite(chi2)
    dirxy = mom[:, :2] / np.hypot(mom[:, 0], mom[:, 1])[:, None]
    sigL = np.sqrt(np.einsum('ni,nij,nj->n', dirxy, Cv[:, :2, :2], dirxy))
    pull = L / sigL
    for name, m in (("right-sign", ~ws), ("wrong-sign", ws)):
        sel = m & ok
        good = sel & (chi2 < 3.84)
        say(f"{name}: {int(sel.sum())} fitted, chi2<3.84: {int(good.sum())} ({good.sum() / max(sel.sum(), 1):.3f}); "
            f"Lxy median {np.nanmedian(L[good]):.3f} cm, frac Lxy>0 {np.mean(L[good] > 0):.3f}")
    P = chi2_sf(chi2, 1)
    h = np.histogram(P[~ws & ok], bins=10, range=(0, 1))[0]
    say("P(chi2), right-sign, 10 bins (first-bin spike = background; the rest should be roughly flat):")
    say("  " + " ".join(str(int(x)) for x in h))
    edges = np.array([-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5])
    hr = np.histogram(L[~ws & ok & (chi2 < 3.84)], edges)[0]
    hw = np.histogram(L[ws & ok & (chi2 < 3.84)], edges)[0]
    say("Lxy [cm] bin edges (chi2<3.84 only): " + " ".join(f"{x:g}" for x in edges))
    say("  right-sign:           " + " ".join(f"{int(x):4d}" for x in hr))
    say("  wrong-sign:           " + " ".join(f"{int(x):4d}" for x in hw))
    say("  right - wrong:        " + " ".join(f"{int(x):4d}" for x in hr - hw))
    pos = edges[:-1] >= 0.0
    say(f"signal-like excess at Lxy > 0: {int((hr - hw)[pos].sum())}, at Lxy < 0: {int((hr - hw)[~pos].sum())} "
        "(a real lifetime signal sits at positive Lxy; the wrong-sign sample still holds the swapped-K/pi reflection of real D0)")
    say("\n== reference point of the exported d0 ==")
    for name in ("k", "pi"):
        d0v = cd[f"{name}_d0"].to_numpy(); ipp = cd[f"{name}_ip_pv"].to_numpy(); ipb = cd[f"{name}_ip_bs"].to_numpy()
        q = lambda x: " ".join(f"{t:+.4f}" for t in np.nanquantile(x, [0.1, 0.5, 0.9]))
        say(f"{name}: quantiles 10/50/90% [cm] of  d0-ip_pv: {q(d0v - ipp)}   d0+ip_pv: {q(d0v + ipp)}   d0-ip_bs: {q(d0v - ipb)}   ip_pv-ip_bs: {q(ipp - ipb)}")
    pvr = cd[["pv_x", "pv_y", "pv_z"]].to_numpy(float)
    bsr = cd[["bs_x", "bs_y", "bs_z"]].to_numpy(float)
    org = np.zeros_like(pvr)
    say("fit reference (where d0 is taken to be measured from); L is always measured from the event PV. Real data: the origin wins:")
    say("reference   chi2<3.84   median sigma_L   frac L/sigma<-3   frac |L/sigma|<2   median L [cm]")
    for rname, rf in (("event PV", pvr), ("beam spot", bsr), ("origin", org)):
        vv, cc, CC = fit_vertex(par[~ws], W[~ws], rf[~ws], c, err_scale=a.err_scale)
        okk = np.isfinite(cc)
        LL = decay_length(vv, pvr[~ws], mom[~ws])
        dd = mom[~ws][:, :2] / np.hypot(mom[~ws][:, 0], mom[~ws][:, 1])[:, None]
        sg = np.sqrt(np.einsum("ni,nij,nj->n", dd, CC[:, :2, :2], dd))
        pl = LL / sg
        gg = okk & (cc < 3.84)
        say(f"{rname:11s}{np.mean(cc[okk] < 3.84):9.3f}{np.nanmedian(sg[gg]):16.3f}{np.mean(pl[gg] < -3):17.3f}{np.mean(np.abs(pl[gg]) < 2):18.3f}{np.nanmedian(LL[gg]):16.3f}")
    say("the right reference gives the most candidates with |L/sigma| < 2 (prompt-like pairs) and almost none below -3\n")
    pe = np.array([-100, -6, -4, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12, 20, 50, 1e9])
    gk = ok & (chi2 < 3.84)
    hr = np.histogram(pull[~ws & gk], pe)[0]; hw = np.histogram(pull[ws & gk], pe)[0]
    say("\nL/sigma_L (chi2<3.84): median sigma_L right-sign %.3f cm, wrong-sign %.3f cm" % (np.nanmedian(sigL[~ws & gk]), np.nanmedian(sigL[ws & gk])))
    say("bin edges: " + " ".join(f"{x:g}" for x in pe[1:-1]))
    say("  right-sign:  " + " ".join(f"{int(x):4d}" for x in hr))
    say("  wrong-sign:  " + " ".join(f"{int(x):4d}" for x in hw))
    say("  fraction with L/sigma_L < -3: right-sign %.3f, wrong-sign %.3f  (a Gaussian resolution alone would give 0.001)" %
        (np.mean(pull[~ws & gk] < -3), np.mean(pull[ws & gk] < -3)))
    say("  fraction with L/sigma_L > 3:  right-sign %.3f, wrong-sign %.3f" % (np.mean(pull[~ws & gk] > 3), np.mean(pull[ws & gk] > 3)))
    say("expected: real D0 with p_D0 > 27 GeV (X_E>0.3): mean Lxy ~ beta*gamma*c*tau = 0.17 cm, resolution ~ 0.03 cm")
    open(f"{a.out}_summary.txt", "w").write("\n".join(out) + "\n")
    print(f"wrote {a.out}_summary.txt")


if __name__ == "__main__":
    main()
