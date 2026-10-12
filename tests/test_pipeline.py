"""
Smoke tests on synthetic data (no DELPHI data needed): run with  python tests/test_pipeline.py  or  pytest tests/.

  * the vertex fit gives chi2(ndf=1) for two tracks from a common vertex and recovers the sign convention
  * the Lambda_c and the D+ channels find their injected signal, with an empty control sample, through the same
    config-driven CLI that is used on real data (lc_run.py reco / select)
  * (run, evt) pairs that occur twice in a file do not break the reconstruction
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

import synthetic  # noqa: E402
from lc_vertex import ALL_CONVENTIONS, Conv, chi2_sf, fit_vertex  # noqa: E402

CONFIG = """
[general]
channel = {channel}
data_glob = {glob}
setmap =
out = {out}
[reco]
nproc = 1
tag_min = 1
err_scale = 1.0
[select]
[fit]
sigma_mev = 16
"""


def run_cli(tmp, channel, glob, out, extra=()):
    ini = os.path.join(tmp, f"{channel}.ini")
    open(ini, "w").write(CONFIG.format(channel=channel, glob=glob, out=out))
    cmd = [sys.executable, os.path.join(SRC, "lc_run.py"), ini, "all", *extra]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout


def excess(stdout):
    m = re.search(r"counting: .*excess ([-\d.]+), significance", stdout)
    assert m, stdout
    return float(m.group(1))


def test_vertex_fit_convention():
    """two tracks from one vertex: chi2 ~ chi2(1 dof) and the true sign of d0 is preferred"""
    rng = np.random.default_rng(3)
    N = 3000
    ref = np.stack([rng.normal(-0.3, 0.012, N), rng.normal(0.14, 0.001, N), rng.normal(-0.7, 0.7, N)], axis=1)
    org = np.zeros_like(ref)
    ph0, ct0 = rng.uniform(-np.pi, np.pi, N), rng.uniform(-.8, .8, N)
    st0 = np.sqrt(1 - ct0 ** 2)
    vtx = ref + rng.exponential(0.1, N)[:, None] * np.stack([st0 * np.cos(ph0), st0 * np.sin(ph0), ct0], axis=1)
    pars, Ws = [], []
    for q in (-1, 1):
        p = rng.uniform(2, 20, N)
        th, ph = np.arccos(ct0) + rng.normal(0, .15, N), ph0 + rng.normal(0, .15, N)
        p3 = np.stack([p * np.sin(th) * np.cos(ph), p * np.sin(th) * np.sin(ph), p * np.cos(th)], axis=1)
        par = synthetic.helix_to_perigee(vtx, p3, q * np.ones(N), org)
        sig = np.stack([np.full(N, 1.5e-3), np.full(N, 6e-3), np.full(N, 4e-4), np.full(N, 3e-4),
                        2e-6 + 2e-3 * np.abs(par[:, 4])], axis=1)
        par = par + rng.normal(size=par.shape) * sig
        W = np.zeros((N, 5, 5))
        for i in range(5):
            W[:, i, i] = 1 / sig[:, i] ** 2
        pars.append(par)
        Ws.append(W)
    par, W = np.stack(pars, axis=1), np.stack(Ws, axis=1)
    frac = {}
    for c in ALL_CONVENTIONS:
        v, chi2, _ = fit_vertex(par, W, org, c)
        frac[c] = float(np.mean(chi2[np.isfinite(chi2)] < 3.84))
    assert frac[Conv(-1, 1, False)] > 0.93, frac          # the generating convention: ~95% below the 1-dof 95% point
    assert frac[Conv(1, 1, False)] < 0.6, frac            # the wrong d0 sign is clearly worse
    v, chi2, _ = fit_vertex(par, W, org, Conv(-1, 1, False))
    assert abs(np.nanmean(chi2) - 1.0) < 0.15
    p = chi2_sf(chi2, 1)
    assert np.all(np.histogram(p[np.isfinite(p)], bins=10, range=(0, 1))[0] > 0.07 * N)   # flat P(chi2)


def test_lc_channel():
    with tempfile.TemporaryDirectory() as tmp:
        n = synthetic.make(os.path.join(tmp, "s0"), "lc", n_events=2500, seed=1)
        out = run_cli(tmp, "lc", os.path.join(tmp, "s0.events.csv"), os.path.join(tmp, "lc"))
        assert excess(out) > 0.4 * n, (excess(out), n)
        cand = pd.read_csv(os.path.join(tmp, "lc_candidates.csv.gz"))
        ctrl = cand[(cand.ctrl == 1) & (cand.mass > 2.25) & (cand.mass < 2.32) & (cand.P > 0.001)]
        assert len(ctrl) <= 3, len(ctrl)                    # no K charge = p charge combination from the injected signal


def test_dplus_channel():
    with tempfile.TemporaryDirectory() as tmp:
        n = synthetic.make(os.path.join(tmp, "s0"), "dplus", n_events=2500, frac=0.12, seed=2)
        out = run_cli(tmp, "dplus", os.path.join(tmp, "s0.events.csv"), os.path.join(tmp, "dp"))
        assert excess(out) > 0.3 * n, (excess(out), n)


def test_duplicate_event_numbers():
    with tempfile.TemporaryDirectory() as tmp:
        synthetic.make(os.path.join(tmp, "s0"), "lc", n_events=800, seed=4, duplicate_event_ids=True)
        run_cli(tmp, "lc", os.path.join(tmp, "s0.events.csv"), os.path.join(tmp, "dup"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
