#!/usr/bin/env python3
"""
lc_run.py - run the charm reconstruction and look at the final signal, all steered by a config file.

    python lc_run.py lc_config.ini reco      # heavy: reads the CSV dump, writes <out>_candidates.csv.gz (minutes to hours)
    python lc_run.py lc_config.ini select    # instant: applies the [select] and [fit] settings to that table
    python lc_run.py lc_config.ini all

Play with the selection without editing the file:
    python lc_run.py lc_config.ini select --set select.tag_p=3 --set select.xe_min=0.4 --set select.label=_tight

The select stage prints a cut flow (physical and control), the mass spectrum, the counting and fit results
and writes <out>_select<label>.png and <out>_select<label>_summary.txt.
"""
import argparse
import configparser
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

import lc_reco
from lc_reco import CHANNELS, hel_cut, tag_ok
from lc_select import fit_poly, significance
from lc_vertex import DEFAULT_ERR_SCALE


def load_cfg(path, overrides):
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not cp.read(path):
        sys.exit(f"cannot read {path}")
    for o in overrides:
        key, _, v = o.partition("=")
        sec, _, k = key.partition(".")
        if not cp.has_section(sec):
            cp.add_section(sec)
        cp.set(sec, k, v)
    return cp


def val(cp, sec, key, typ=str, default=None):
    if not cp.has_option(sec, key):
        return default
    raw = cp.get(sec, key).strip()
    if raw == "":
        return default
    if typ is bool:
        return raw.lower() in ("1", "yes", "true", "on")
    return typ(raw)


# ---------------------------------------------------------------------------------------------- reco
def stage_reco(cp):
    a = SimpleNamespace(
        events=val(cp, "general", "data_glob"), channel=val(cp, "general", "channel", str, "lc"),
        out=val(cp, "general", "out", str, "lc_out"), setmap=val(cp, "general", "setmap"),
        nproc=val(cp, "reco", "nproc", int, 1), files=val(cp, "reco", "files", int, 0),
        tag_min=val(cp, "reco", "tag_min", int, 1), err_scale=val(cp, "reco", "err_scale", float, DEFAULT_ERR_SCALE),
        ecm_window=val(cp, "reco", "ecm_window", float, 2.0), exclude_sets=val(cp, "reco", "exclude_sets", str, ""),
        store_xe_min=val(cp, "reco", "store_xe_min", float, None), store_p_min=val(cp, "reco", "store_p_min", float, 1e-4))
    if not a.events:
        sys.exit("general.data_glob is empty")
    lc_reco.run(a)


# -------------------------------------------------------------------------------------------- select
def stage_select(cp):
    channel = val(cp, "general", "channel", str, "lc")
    ch = CHANNELS[channel]
    out = val(cp, "general", "out", str, "lc_out")
    label = val(cp, "select", "label", str, "")
    df = pd.read_csv(f"{out}_candidates.csv.gz")
    n_raw = len(df)
    df = df.drop_duplicates(["run", "evt", "ctrl", "mass", "xe"])
    S = lambda k, t, d: val(cp, "select", k, t, d)
    F = lambda k, t, d: val(cp, "fit", k, t, d)
    xe_min = S("xe_min", float, ch["xe_min"])
    l_min = S("l_min", float, ch["lcut"][2])
    p_min = S("p_min", float, ch["pcut"])
    use_hel = S("helicity", bool, True)
    hel = (S("hel_a", float, ch["hel"][0]), S("hel_b", float, ch["hel"][1]), S("hel_c", float, ch["hel"][2]))
    tag_p, tag_k = S("tag_p", int, 1), S("tag_k", int, 1)
    charge = S("charge", int, 0)
    holdout = S("holdout", str, "none").lower()
    bvar, blo, bhi = S("btag_var", str, "btp3"), S("btag_lo", float, None), S("btag_hi", float, None)
    veto = S("dstar_veto", bool, ch["dstar_veto"])
    m_lo, m_hi = S("mass_lo", float, ch["window"][0]), S("mass_hi", float, ch["window"][1])
    nb = F("bins", int, ch["nbins"])
    sigma = F("sigma_mev", float, 0.0)
    half = F("fit_half_range", float, 0.10)
    win, sb_lo, sb_hi = F("signal_window", float, 0.025), F("sideband_lo", float, 0.05), F("sideband_hi", float, 0.10)
    shift = F("mass_shift", float, 0.0)
    m0 = ch["mass"] + shift

    def flow(d):
        steps = [("candidates in the table", len(d))]
        m = np.ones(len(d), dtype=bool)

        def step(name, mask):
            nonlocal m
            m = m & np.asarray(mask)
            steps.append((name, int(m.sum())))
        if holdout in ("even", "odd"):
            step(f"event number {holdout} only", (d.evt % 2 == (0 if holdout == "even" else 1)).to_numpy())
        if charge:
            step(f"charge of the first particle = {charge:+d}", (d.q == charge).to_numpy())
        if blo is not None or bhi is not None:
            step(f"{bvar} in [{blo}, {bhi})", ((d[bvar] >= (-1 if blo is None else blo)) & (d[bvar] < (2 if bhi is None else bhi))).to_numpy())
        step(f"mass in [{m_lo}, {m_hi}]", ((d.mass > m_lo) & (d.mass < m_hi)).to_numpy())
        if ch["roles"][0][3] == "p":
            step(f"proton tag >= {tag_p}", tag_ok(d.p_h_rich_p, d.p_h_dedx_p, tag_p))
        step(f"kaon tag >= {tag_k}", tag_ok(d.k_h_rich_k, d.k_h_dedx_k, tag_k))
        step(f"X_E > {xe_min}", (d.xe > xe_min).to_numpy())
        if use_hel:
            step(f"helicity-angle cut (a,b,c = {hel})", hel_cut(d.xe, d.cosh, hel).to_numpy())
        lim = np.full(len(d), l_min)
        if ch["lcut"][0] is not None:
            lim = np.maximum(lim, ch["lcut"][0] * (d.xe.to_numpy() - xe_min) ** 2 + ch["lcut"][1])
        step(f"decay length L > {l_min} cm" + (" and the energy-dependent cut" if ch["lcut"][0] is not None else ""), (d.L.to_numpy() > lim))
        step(f"P(chi2) > {p_min}", (d.P > p_min).to_numpy())
        if veto:
            step("D* veto", d.veto.to_numpy())
        return m, steps

    ph, ct = df[df.ctrl == 0], df[df.ctrl == 1]
    mp, sp = flow(ph)
    mc, sc = flow(ct)
    lines = [f"{ch['label']}   table: {out}_candidates.csv.gz   ({n_raw} rows, {len(df)} after removing exact duplicates)",
             f"effective cuts: tag_p={tag_p} tag_k={tag_k} xe_min={xe_min} l_min={l_min} p_min={p_min} helicity={use_hel}{hel if use_hel else ''} "
             f"charge={charge} holdout={holdout} btag={bvar}[{blo},{bhi}) veto={veto}",
             f"fit/counting: bins={nb} sigma_mev={sigma} fit_half_range={half} window=+-{win} sidebands={sb_lo}-{sb_hi} mass_shift={shift}",
             "", f"{'cut flow':60s}{'physical':>12s}{'control':>12s}"]
    for (n1, v1), (_, v2) in zip(sp, sc):
        lines.append(f"{n1:60s}{v1:>12,d}{v2:>12,d}")
    sel, ctl = ph[mp], ct[mc]
    edges = np.linspace(m_lo, m_hi, nb + 1)
    hp, hc = np.histogram(sel.mass, edges)[0], np.histogram(ctl.mass, edges)[0]
    lines += ["", f"mass spectrum, {(m_hi - m_lo) / nb * 1000:.0f} MeV bins from {m_lo}:",
              "  physical: " + " ".join(str(int(x)) for x in hp), "  control : " + " ".join(str(int(x)) for x in hc)]
    sig, sb, ex, z = significance(sel.mass, m0, win, sb_lo, sb_hi)
    lines.append(f"counting: |m - {m0:.4f}| < {win * 1000:.0f} MeV: {sig}; sidebands {sb_lo * 1000:.0f}-{sb_hi * 1000:.0f} MeV estimate {sb:.1f}; "
                 f"excess {ex:.1f}, significance {z:.1f} sigma")
    res = fit_poly(sel.mass.to_numpy(), m_lo, m_hi, nb, ch["mass"], sigma if sigma > 0 else None, half) if F("fit", bool, True) else None
    if isinstance(res, dict):
        lines.append(f"fit in mass +- {half} GeV, Gaussian ({'sigma fixed to ' + str(sigma) + ' MeV' if sigma > 0 else 'sigma free'}) + quadratic background: "
                     f"yield {res['yield_']:.0f} +- {res['err']:.0f}, mean {res['mean']:.4f} GeV, sigma {res['sigma'] * 1000:.1f} MeV, chi2/ndf {res['chi2']:.0f}/{res['ndf']}")
        lines.append("  (a free sigma is poorly determined on this statistics: compare sigma_mev = 12, 16, 20)")
    elif isinstance(res, str):
        lines.append(res)
    if F("lhist", bool, False) and len(sel):
        dm = np.abs(sel.mass - m0)
        w_, s_ = sel[dm < win], sel[(dm > sb_lo) & (dm < sb_hi)]
        le = np.array([0.0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0])
        hw, hs = np.histogram(w_.L, le)[0], np.histogram(s_.L, le)[0] * (win / (sb_hi - sb_lo))
        lines += ["", "decay length L [cm] of the peak minus sidebands:", "  L bin edges:     " + " ".join(f"{x:5.2f}" for x in le),
                  "  window:          " + " ".join(f"{int(x):5d}" for x in hw), "  sidebands scaled:" + " ".join(f"{x:5.0f}" for x in hs),
                  "  signal:          " + " ".join(f"{x:5.0f}" for x in hw - hs)]
        if (hw - hs).sum() > 0:
            lines.append(f"  mean L of the excess: {np.sum((hw - hs) * 0.5 * (le[1:] + le[:-1])) / (hw - hs).sum():.3f} cm (a pure Lambda_c would give ~0.05-0.09 cm)")
    txt = "\n".join(l for l in lines if l is not None)
    print(txt)
    open(f"{out}_select{label}_summary.txt", "w").write(txt + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5))
        c = 0.5 * (edges[1:] + edges[:-1])
        ax.errorbar(c, hp, yerr=np.sqrt(hp), fmt="ko", ms=3, label=f"data {ch['label']} (+c.c.)")
        if F("control_scaled", bool, True) and len(ctl):
            d1 = np.abs(sel.mass - m0); d2 = np.abs(ctl.mass - m0)
            nb_s = ((d1 > sb_lo) & (d1 < sb_hi)).sum(); nc_s = ((d2 > sb_lo) & (d2 < sb_hi)).sum()
            k = nb_s / nc_s if nc_s else 1.0
            ax.stairs(hc * k, edges, color="tab:orange", label=f"control, scaled x{k:.2f} to the sidebands")
        else:
            ax.stairs(hc, edges, color="tab:orange", label="control")
        if isinstance(res, dict):
            xx = res["x"]
            ax.plot(xx, res["curve"](), "r-", label="Gauss + pol2 (fit window)")
        ax.axvline(ch["mass"], color="gray", ls=":")
        ax.set_xlabel("mass [GeV]"); ax.set_ylabel(f"candidates / {(m_hi - m_lo) / nb * 1000:.0f} MeV")
        ax.set_title(f"{ch['label']}: {ex:.0f} excess ({z:.1f} sigma)")
        ax.legend(); fig.tight_layout(); fig.savefig(f"{out}_select{label}.png", dpi=120)
        print(f"\nwrote {out}_select{label}.png and {out}_select{label}_summary.txt")
    except Exception as exc:
        print("plot failed:", exc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("stage", choices=["reco", "select", "all"])
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value", help="override a config value")
    a = ap.parse_args()
    cp = load_cfg(a.config, a.set)
    if a.stage in ("reco", "all"):
        stage_reco(cp)
    if a.stage in ("select", "all"):
        stage_select(cp)


if __name__ == "__main__":
    main()
