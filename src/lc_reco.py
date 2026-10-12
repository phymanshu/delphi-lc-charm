#!/usr/bin/env python3
"""
lc_reco.py - charm-hadron reconstruction on the lc dump, following the selection of the DELPHI charm
paper (Eur. Phys. J. C12 (2000) 225; draft CERN-EP/Paper215).

Channels (--channel):
  lc     Lambda_c+ -> p K- pi+          (the target)
  dplus  D+ -> K- pi+ pi+               (validation: same code path, 7018 candidates in the paper, so it is
                                         visible already in ~40 files; checks the 3-track vertex fit, the kaon
                                         PID logic, the helicity and decay-length cuts)

Selection (paper -> here):
  event      hadronic: ihad4 == 1 and charged energy > 12% sqrt(s), sqrt(s) within 2 GeV of 91.19 GeV
  tracks     0.4 < p < 50 GeV, 20 < theta < 160 deg, |trklen| > 30 cm, |ip_pv| < 4 cm, |zip_pv| < 10 cm,
             relative momentum error (eerr) < 1; every candidate track needs >= 1 VD R-phi hit
  candidate  all combinations with p > 1 GeV (K, pi) or p > 2 GeV (proton) and the right charges
  PID        K and p identified: RICH tag >= --tag-min, or the dE/dx tag if there is no RICH information;
             a track with neither is rejected
  vertex     3-track secondary-vertex fit (lc_vertex.py), pull-calibrated errors, momenta at the vertex
  cuts       mass window and X_E minimum (table 1), helicity-angle X_E cut (table 2; leading particle = p or K),
             decay length (table 3), P(chi2) > 0.001 (table text); D+ also needs M(K pi pi) - M(K pi) > 0.15 GeV
A control sample uses the unphysical charge pattern (Lc: K charge = p charge; D+: K+ pi+ pi+). It has no
signal and shows the combinatorial background.

Not done: the 3D fit with X_E and tr(P_ev), the simulation efficiency; 1991/1992 can be excluded with
--exclude-sets (the paper did not use 1992 for Lambda_c).

Usage:
    python lc_reco.py 'dump/csv/*.events.csv' --channel dplus --files 40 --out dplus_test --nproc 8
    python lc_reco.py 'dump/csv/*.events.csv' --channel lc --out lc_full --nproc 8 \\
        --setmap all_list.txt --exclude-sets short91_f1,short92_e2
Writes <out>_candidates.csv.gz (loosely selected, with the PID tags, so cuts can be varied offline),
<out>_summary.txt and <out>_mass.png.
"""
import argparse
import glob
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from lc_schema import EVENT_COLS, TRACK_COLS
from lc_vertex import DEFAULT_CONV, DEFAULT_ERR_SCALE, chi2_sf, fit_vertex, phi_at_vertex, unpack_w

MP, MK, MPI = 0.93827, 0.49368, 0.13957
WCOLS = [f"w{k}" for k in range(1, 16)]
PCOLS = ["d0", "z0", "theta", "phi", "curv"]
TRACK_NEED = (["run", "evt", "idx", "q", "px", "py", "pz", "trklen", "ip_pv", "zip_pv", "eerr", "vd_rphi",
               "h_rich_k", "h_rich_p", "h_dedx_k", "h_dedx_p"] + PCOLS + WCOLS)

# roles: (name, mass, pmin, tag, charge relative to role 0 for the physical / the control combination)
CHANNELS = {
    "lc": dict(label="Lambda_c -> p K pi", mass=2.28646, window=(2.10, 2.50), nbins=25, xe_min=0.30,
               hel=(0.5, 3.0, 0.15), lead=0, lcut=(None, None, 0.015), pcut=0.001, dstar_veto=False,
               roles=[("p", MP, 2.0, "p", +1, +1), ("k", MK, 1.0, "k", -1, +1), ("pi", MPI, 1.0, None, +1, -1)],
               sym=False),
    "dplus": dict(label="D+ -> K pi pi", mass=1.86965, window=(1.70, 2.05), nbins=50, xe_min=0.20,
                  hel=(0.5, 3.0, 0.10), lead=0, lcut=(-1.0, 0.230, 0.125), pcut=0.001, dstar_veto=True,
                  roles=[("k", MK, 1.0, "k", +1, +1), ("pi1", MPI, 1.0, None, -1, +1), ("pi2", MPI, 1.0, None, -1, +1)],
                  sym=True),
}


def stages(ch):
    return ["triplets (PID, p, VD)", "pre-selection at the perigee (mass window +-0.1, XE-0.05)", "vertex fit converged",
            "mass window and XE min at the vertex", "helicity-angle XE cut", "decay-length cut",
            "P(chi2) > 0.001" + (" and D* veto" if ch["dstar_veto"] else "")]


def tag_ok(rich, dedx, tmin):
    """identified: RICH tag if there is RICH information, else the dE/dx tag, else not identified"""
    has_rich = rich != -1
    has_dedx = dedx != -1
    return np.where(has_rich, rich >= tmin, np.where(has_dedx, dedx >= tmin, False))


def select_tracks(tr):
    tr = tr.copy()
    tr["p"] = np.sqrt(tr.px ** 2 + tr.py ** 2 + tr.pz ** 2)
    th = np.degrees(np.arccos((tr.pz / tr.p).clip(-1, 1)))
    ok = ((tr.p > 0.4) & (tr.p < 50) & (th > 20) & (th < 160) & (tr.trklen.abs() > 30) &
          (tr.ip_pv != -999) & (tr.zip_pv != -999) & (tr.ip_pv.abs() < 4) & (tr.zip_pv.abs() < 10) &
          (tr.eerr < 1) & (tr.q != 0) & (tr.vd_rphi > 0))
    return tr[ok]


def role_table(t, role, tmin):
    name, _, pmin, tag, _, _ = role
    sel = t.p >= pmin
    if tag:
        sel = sel & tag_ok(t[f"h_rich_{tag}"], t[f"h_dedx_{tag}"], tmin)
    return t[sel][["eid", "idx", "q"]]


def triplets(t, ch, tmin, control):
    r0, r1, r2 = ch["roles"]
    c = 1 if control else 0
    a = role_table(t, r0, tmin).rename(columns={"idx": "i0", "q": "q0"})
    b = role_table(t, r1, tmin).rename(columns={"idx": "i1", "q": "q1"})
    cc = role_table(t, r2, tmin).rename(columns={"idx": "i2", "q": "q2"})
    ab = a.merge(b, on="eid")
    ab = ab[(ab.i0 != ab.i1) & (ab.q1 == ab.q0 * r1[4 + c])]
    tri = ab.merge(cc, on="eid")
    tri = tri[(tri.i2 != tri.i0) & (tri.i2 != tri.i1) & (tri.q2 == tri.q0 * r2[4 + c])]
    if ch["sym"]:
        tri = tri[tri.i1 < tri.i2]
    return tri


def kinematics(px, py, pz, masses):
    E = np.zeros(len(px))
    for i, m in enumerate(masses):
        E += np.sqrt(px[:, i] ** 2 + py[:, i] ** 2 + pz[:, i] ** 2 + m ** 2)
    sx, sy, sz = px.sum(1), py.sum(1), pz.sum(1)
    return E, sx, sy, sz, np.sqrt(np.maximum(E ** 2 - sx ** 2 - sy ** 2 - sz ** 2, 0))


def helicity(E, sx, sy, sz, M, lx, ly, lz, El):
    """cos of the angle between the leading particle in the candidate rest frame and the flight direction"""
    P = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)
    n = np.stack([sx, sy, sz], axis=1) / P[:, None]
    beta, gamma = P / E, E / M
    lpar = lx * n[:, 0] + ly * n[:, 1] + lz * n[:, 2]
    lperp2 = np.maximum(lx ** 2 + ly ** 2 + lz ** 2 - lpar ** 2, 0)
    lpar_star = gamma * (lpar - beta * El)
    return lpar_star / np.sqrt(lpar_star ** 2 + lperp2)


def hel_cut(xe, ch_, abc):
    a, b, c = abc
    return (xe > a * np.exp(b * (ch_ - 1)) + c) & (xe > a * np.exp(-b * (ch_ + 1)) + c)


def length_cut(L, xe, ch, xe_min):
    x, y, lmin = ch["lcut"]
    lim = np.full(len(L), lmin)
    if x is not None:
        lim = np.maximum(lim, x * (np.asarray(xe) - xe_min) ** 2 + y)
    return L > lim


def process_file(args_tuple):
    ev_file, a = args_tuple
    ch = CHANNELS[a.channel]
    st = stages(ch)
    flow = {"ctrl": Counter(), "phys": Counter()}
    empty = pd.DataFrame()
    tr_file = ev_file.replace(".events.csv", ".tracks.csv")
    try:
        ev = pd.read_csv(ev_file, names=EVENT_COLS)
        tr = pd.read_csv(tr_file, names=TRACK_COLS, usecols=TRACK_NEED)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return flow, empty
    # unique event id: track rows follow the event order, each event has `nwrit` rows (checked by the dump).
    # (run, evt) is NOT unique inside every file, so it cannot be used as the event key.
    ev["eid"] = np.arange(len(ev))
    if int(ev.nwrit.sum()) != len(tr):
        print(f"WARNING: {ev_file}: track rows do not match nwrit, skipped", file=sys.stderr)
        return flow, empty
    tr["eid"] = np.repeat(ev.eid.to_numpy(), ev.nwrit.to_numpy().astype(int))
    ecm = ev.ecm.where(ev.ecm > 1, 91.19)
    ev = ev.assign(ecm=ecm)
    ev = ev[(ev.ihad4 == 1) & (ev.echar / ecm > 0.12) & (np.abs(ecm - 91.19) < a.ecm_window)]
    flow["phys"]["hadronic events"] += len(ev)
    flow["ctrl"]["hadronic events"] += len(ev)
    if ev.empty or tr.empty:
        return flow, empty
    t = select_tracks(tr).merge(ev[["eid", "ecm", "pv_x", "pv_y", "pv_z", "echar", "btn3", "btp3", "bts3"]], on="eid")
    ti = t.set_index(["eid", "idx"])
    runmap = dict(zip(ev.eid, ev.run))
    evtmap = dict(zip(ev.eid, ev.evt))
    names = [r[0] for r in ch["roles"]]
    masses = tuple(r[1] for r in ch["roles"])
    lo, hi = ch["window"]
    out = []
    for control in (False, True):
        key = "ctrl" if control else "phys"
        tri = triplets(t, ch, a.tag_min, control)
        flow[key][st[0]] += len(tri)
        if tri.empty:
            continue
        legs = {n: ti.reindex(pd.MultiIndex.from_arrays([tri.eid, tri[f"i{k}"]])) for k, n in enumerate(names)}
        px = np.stack([legs[n].px.to_numpy() for n in names], axis=1)
        py = np.stack([legs[n].py.to_numpy() for n in names], axis=1)
        pz = np.stack([legs[n].pz.to_numpy() for n in names], axis=1)
        ecm_c = legs[names[0]].ecm.to_numpy()
        E, sx, sy, sz, M = kinematics(px, py, pz, masses)
        xe = 2 * E / ecm_c
        pre = (M > lo - 0.1) & (M < hi + 0.1) & (xe > ch["xe_min"] - 0.05)
        flow[key][st[1]] += int(pre.sum())
        if not pre.any():
            continue
        tri = tri[pre].reset_index(drop=True)
        legs = {n: legs[n][pre] for n in names}
        ecm_c = ecm_c[pre]
        par, W = [], []
        for n in names:
            pp = np.stack([legs[n][c].to_numpy() for c in PCOLS], axis=1).astype(float)
            pp[:, 2] = np.arccos(np.clip(legs[n].pz.to_numpy() / legs[n].p.to_numpy(), -1, 1))
            pp[:, 3] = np.arctan2(legs[n].py.to_numpy(), legs[n].px.to_numpy())
            par.append(pp)
            W.append(unpack_w(legs[n][WCOLS].to_numpy()))
        par, W = np.stack(par, axis=1), np.stack(W, axis=1)
        pv = legs[names[0]][["pv_x", "pv_y", "pv_z"]].to_numpy(float)
        org = np.zeros_like(pv)
        v, chi2, Cv = fit_vertex(par, W, org, DEFAULT_CONV, err_scale=a.err_scale)
        okf = np.isfinite(chi2)
        flow[key][st[2]] += int(okf.sum())
        if not okf.any():
            continue
        pxv, pyv, pzv = [], [], []
        for i, n in enumerate(names):
            phiv = phi_at_vertex(v, par[:, i], org, DEFAULT_CONV)
            pt = np.hypot(legs[n].px.to_numpy(), legs[n].py.to_numpy())
            pxv.append(pt * np.cos(phiv)); pyv.append(pt * np.sin(phiv)); pzv.append(legs[n].pz.to_numpy())
        pxv, pyv, pzv = np.stack(pxv, 1), np.stack(pyv, 1), np.stack(pzv, 1)
        E, sx, sy, sz, M = kinematics(pxv, pyv, pzv, masses)
        xe = 2 * E / ecm_c
        ld = ch["lead"]
        El = np.sqrt(pxv[:, ld] ** 2 + pyv[:, ld] ** 2 + pzv[:, ld] ** 2 + masses[ld] ** 2)
        with np.errstate(all="ignore"):
            chl = helicity(E, sx, sy, sz, M, pxv[:, ld], pyv[:, ld], pzv[:, ld], El)
            d = np.stack([sx, sy], axis=1) / np.hypot(sx, sy)[:, None]
            L = ((v[:, :2] - pv[:, :2]) * d).sum(axis=1)
            sigL = np.sqrt(np.einsum("ni,nij,nj->n", d, Cv[:, :2, :2], d))
        P = chi2_sf(chi2, 3)
        veto = np.ones(len(M), dtype=bool)
        if ch["dstar_veto"]:       # M(K pi pi) - M(K pi_i) must exceed 150 MeV for both pions
            for j in (1, 2):
                Ej = np.sqrt(pxv[:, 0] ** 2 + pyv[:, 0] ** 2 + pzv[:, 0] ** 2 + MK ** 2) + \
                    np.sqrt(pxv[:, j] ** 2 + pyv[:, j] ** 2 + pzv[:, j] ** 2 + MPI ** 2)
                mj = np.sqrt(np.maximum(Ej ** 2 - (pxv[:, 0] + pxv[:, j]) ** 2 - (pyv[:, 0] + pyv[:, j]) ** 2 -
                                        (pzv[:, 0] + pzv[:, j]) ** 2, 0))
                veto &= (M - mj) > 0.150
        c1 = okf & (M > lo) & (M < hi) & (xe > ch["xe_min"])
        c2 = c1 & hel_cut(xe, chl, ch["hel"])
        c3 = c2 & length_cut(L, xe, ch, ch["xe_min"])
        c4 = c3 & (P > ch["pcut"]) & veto
        for s, c in zip(st[3:], (c1, c2, c3, c4)):
            flow[key][s] += int(c.sum())
        xe_store = ch["xe_min"] - 0.05 if a.store_xe_min is None else a.store_xe_min
        keep = okf & (L > 0) & (P > a.store_p_min) & (xe > xe_store) & (M > lo) & (M < hi)
        if keep.any():
            eids = tri.eid.to_numpy()
            df = pd.DataFrame({"run": [runmap[i] for i in eids], "evt": [evtmap[i] for i in eids],
                               "ctrl": int(control), "mass": M, "xe": xe, "cosh": chl, "L": L, "sigL": sigL,
                               "chi2": chi2, "P": P, "veto": veto, "q": tri.q0.to_numpy(),
                               "p0": legs[names[0]].p.to_numpy(), "p1": legs[names[1]].p.to_numpy(),
                               "p2": legs[names[2]].p.to_numpy(), "echar": legs[names[0]].echar.to_numpy(),
                               "pvz": pv[:, 2], "btn3": legs[names[0]].btn3.to_numpy(),
                               "btp3": legs[names[0]].btp3.to_numpy(), "bts3": legs[names[0]].bts3.to_numpy()})
            for n in names[:2]:
                for c in ("h_rich_p", "h_dedx_p", "h_rich_k", "h_dedx_k"):
                    df[f"{n}_{c}"] = legs[n][c].to_numpy()
            out.append(df[keep])
    return flow, (pd.concat(out, ignore_index=True) if out else empty)


def fit_mass(m, lo, hi, nb, mass, sg_lo=0.004, sg_hi=0.04):
    """Gaussian + exponential on the binned spectrum, model integrated over each bin; needs scipy"""
    try:
        from math import erf
        from scipy.optimize import curve_fit
    except ImportError:
        return None
    h, e = np.histogram(m, bins=nb, range=(lo, hi))
    verf = np.vectorize(erf)
    bw = (hi - lo) / nb

    def counts(n, mu, sg, b, a):
        cdf = 0.5 * (1 + verf((e - mu) / (sg * np.sqrt(2))))
        b = b if abs(b) > 1e-6 else 1e-6
        bkg = a * (np.exp(-b * (e[:-1] - lo)) - np.exp(-b * (e[1:] - lo))) / b
        return n * np.diff(cdf) + bkg

    x = 0.5 * (e[1:] + e[:-1])
    f = lambda xx, n, mu, sg, b, a: counts(n, mu, sg, b, a)
    try:
        popt, pcov = curve_fit(f, x, h, p0=[max(h.max() - h.mean(), 1), mass, 0.012, 1.0, max(h[0], 1) / bw],
                               sigma=np.sqrt(np.maximum(h, 1)), maxfev=40000,
                               bounds=([0, mass - 0.03, sg_lo, 1e-3, 0], [1e7, mass + 0.03, sg_hi, 30, 1e8]))
    except Exception as exc:
        return f"fit failed: {exc}"
    n, mu, sg = popt[:3]
    plot_f = lambda xx, *p: (p[0] * bw * np.exp(-0.5 * ((xx - p[1]) / p[2]) ** 2) / (p[2] * np.sqrt(2 * np.pi)) +
                             p[4] * np.exp(-p[3] * (xx - lo)) * bw)
    return dict(yield_=n, err=np.sqrt(pcov[0, 0]), mean=mu, sigma=sg, popt=popt, f=plot_f)


def final_selection(df, ch, ctrl):
    if df.empty:
        return df
    d = df[df.ctrl == ctrl]
    lo, hi = ch["window"]
    c = ((d.mass > lo) & (d.mass < hi) & (d.xe > ch["xe_min"]) & hel_cut(d.xe, d.cosh, ch["hel"]) &
         length_cut(d.L.to_numpy(), d.xe.to_numpy(), ch, ch["xe_min"]) & (d.P > ch["pcut"]) & d.veto)
    return d[c]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events", help="glob of *.events.csv files")
    ap.add_argument("--channel", choices=sorted(CHANNELS), default="lc")
    ap.add_argument("--out", default="lc_out")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--files", type=int, default=0, help="only the first N files")
    ap.add_argument("--tag-min", type=int, default=1, help="min RICH / dE/dx tag for K and p (1 loose, 2 standard, 3 tight)")
    ap.add_argument("--err-scale", type=float, default=DEFAULT_ERR_SCALE)
    ap.add_argument("--ecm-window", type=float, default=2.0)
    ap.add_argument("--setmap", help="all_list.txt: maps dump file names to their data set")
    ap.add_argument("--store-xe-min", type=float, default=None, help="loosest X_E kept in the candidate table (default: channel X_E min - 0.05)")
    ap.add_argument("--store-p-min", type=float, default=1e-4, help="loosest P(chi2) kept in the candidate table")
    ap.add_argument("--exclude-sets", default="", help="comma-separated data sets to skip, e.g. short91_f1,short92_e2")
    return ap.parse_args()


def run(a):
    ch = CHANNELS[a.channel]
    files = sorted(glob.glob(a.events))
    if a.setmap and a.exclude_sets:
        bad = {s for s in a.exclude_sets.split(",") if s}
        skip = set()
        for line in open(a.setmap):
            parts = line.strip().split("/")
            if len(parts) >= 2 and parts[-2] in bad:
                skip.add(os.path.splitext(parts[-1])[0])
        n0 = len(files)
        files = [f for f in files if os.path.basename(f).replace(".events.csv", "") not in skip]
        print(f"excluded {n0 - len(files)} files of the sets {sorted(bad)}")
    if a.files:
        files = files[:a.files]
    if not files:
        sys.exit("no files")
    print(f"{ch['label']}: {len(files)} files; track error scale {a.err_scale}; tag >= {a.tag_min}", flush=True)
    flow = {"ctrl": Counter(), "phys": Counter()}
    frames = []
    jobs = [(f, a) for f in files]

    def collect(i, fl, df):
        for k in flow:
            flow[k].update(fl[k])
        if len(df):
            frames.append(df)
        if i % 50 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} files", flush=True)

    if a.nproc > 1:
        with ProcessPoolExecutor(max_workers=a.nproc) as ex:
            for i, (fl, df) in enumerate(ex.map(process_file, jobs, chunksize=1), 1):
                collect(i, fl, df)
    else:
        for i, j in enumerate(jobs, 1):
            collect(i, *process_file(j))
    cand = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    st = stages(ch)
    lines = [f"channel {ch['label']}; files: {len(files)}   track error scale {a.err_scale}   PID tag >= {a.tag_min}", "",
             f"{'stage':62s}{'physical':>12s}{'control':>12s}", f"{'hadronic events':62s}{flow['phys']['hadronic events']:>12,d}"]
    for s in st:
        lines.append(f"{s:62s}{flow['phys'][s]:>12,d}{flow['ctrl'][s]:>12,d}")
    fp, fc = final_selection(cand, ch, 0), final_selection(cand, ch, 1)
    lines += ["", f"final candidates: physical {len(fp)}, control {len(fc)}"]
    lo, hi = ch["window"]
    edges = np.linspace(lo, hi, ch["nbins"] + 1)
    res = None
    if len(fp):
        hp = np.histogram(fp.mass, edges)[0]
        hc = np.histogram(fc.mass, edges)[0] if len(fc) else np.zeros(ch["nbins"], int)
        bw = (hi - lo) / ch["nbins"]
        lines += [f"mass spectrum, {bw * 1000:.0f} MeV bins from {lo:.2f} GeV:",
                  "  physical: " + " ".join(str(int(x)) for x in hp), "  control : " + " ".join(str(int(x)) for x in hc)]
        res = fit_mass(fp.mass.to_numpy(), lo, hi, ch["nbins"], ch["mass"])
        if isinstance(res, dict):
            lines.append(f"fit (Gaussian + exponential): yield {res['yield_']:.0f} +- {res['err']:.0f}, mean {res['mean']:.4f} GeV, "
                         f"sigma {res['sigma'] * 1000:.1f} MeV")
        elif isinstance(res, str):
            lines.append(res)
        d_ = np.abs(fp.mass - ch["mass"])
        sig, sb = int((d_ < 0.025).sum()), ((d_ > 0.05) & (d_ < 0.10)).sum() * 0.5
        lines.append(f"counting: |m - {ch['mass']:.4f}| < 25 MeV: {sig}; sideband estimate {sb:.1f}; excess {sig - sb:.1f} +- {np.sqrt(sig + sb * 0.5):.1f}")
    lines.append("paper (3.5M hadronic events 1992-95): D+ 7018 +- 200, Lambda_c 336 +- 42 candidates after all cuts")
    txt = "\n".join(lines)
    print("\n" + txt)
    open(f"{a.out}_summary.txt", "w").write(txt + "\n")
    if len(cand):
        cand.to_csv(f"{a.out}_candidates.csv.gz", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5))
        c = 0.5 * (edges[1:] + edges[:-1])
        if len(fp):
            ax.errorbar(c, hp, yerr=np.sqrt(hp), fmt="ko", ms=3, label=f"data {ch['label']} (+c.c.)")
            ax.stairs(hc, edges, color="tab:orange", label="control (wrong charge pattern)")
            if isinstance(res, dict):
                xx = np.linspace(lo, hi, 300)
                ax.plot(xx, res["f"](xx, *res["popt"]), "r-", label="Gauss + exp")
        ax.axvline(ch["mass"], color="gray", ls=":")
        ax.set_xlabel("mass [GeV]"); ax.set_ylabel(f"candidates / {(hi - lo) / ch['nbins'] * 1000:.0f} MeV"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{a.out}_mass.png", dpi=120)
    except Exception as exc:
        print("plot failed:", exc)



def main():
    run(parse_args())


if __name__ == "__main__":
    main()
