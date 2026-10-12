# Λc → p K π on DELPHI open data

Reconstruction of Λc⁺ → p K⁻ π⁺ (+c.c.) in e⁺e⁻ → Z → hadrons, following the selection of the DELPHI charm paper
(Eur. Phys. J. C12 (2000) 225, the analysis behind arXiv:1404.3888). I worked from the paper's draft text, so
check the published cuts if they matter.

**Result.** A Λc peak of about **780–870 events at 6–7σ** at 2.2865 GeV (3.0M hadronic events), with a flat control
sample, the same excess in Λc⁺ and Λc⁻ (about 390 each, 5.3σ each), and a D⁺ → Kππ cross-check that reproduces the
paper's rate. This is a signal yield, **not** R_c·P(c→Λc)·BR: that needs simulation.

## Layout

```
dump/        delphi_download.sh  delphi_dump.sh  lc_user02.f      download the data; dump it to CSV with SKELANA
src/         lc_schema.py  lc_vertex.py  lc_reco.py  lc_select.py  lc_run.py     the analysis
config/      lc_config.ini                                          every selection you can change
tools/       lc_dump_check.py  lc_dstar_check.py  lc_vertex_scan.py  lc_pull_check.py     validation of the dump
catalogue/   delphi_keep_zpeak.csv  delphi_collision_classified.csv   which data sets were chosen, and why
tests/       synthetic.py  test_pipeline.py                         smoke tests on synthetic data (no DELPHI data needed)
```

## Setup

```bash
pip install -r requirements.txt
python tests/test_pipeline.py        # about a minute; also works with pytest tests/
```

## Pipeline

| Step | Tool | What happens |
|---|---|---|
| 1. Choose data | `catalogue/` | The portal lists 200 collision sets (17.8 TiB). RAW and TANAGRA sets are not usable; I kept the full Short/XShort DST set of each Z-peak period: 9 sets, 16.6M events, 1,138 files, 105 GiB (LEP1 1991–95 + Z-calibration runs of 1997, 1998, 2000). |
| 2. Download | `dump/delphi_download.sh` | `cernopendata-client` + `xrdcp`, size-checked, resumable. Put the data on EOS (AFS is too small). |
| 3. Dump to CSV | `dump/delphi_dump.sh -m lc` | Builds a SKELANA job in the DELPHI environment and writes one `events.csv` + `tracks.csv` per input file: hadronic events (≥5 charged tracks), charged tracks with p ≥ 0.4 GeV, impact parameters, perigee parameters + weight matrix, VD hits, RICH/dE/dx tags, primary vertex, b-tag. Columns: `src/lc_schema.py`. About 29 s per file, 59 GB in total. |
| 4. Reconstruct | `src/lc_run.py … reco` | Paper selection, 3-track vertex fit, PID, helicity and decay-length cuts. Writes a loose candidate table with the PID tags. |
| 5. Select + look | `src/lc_run.py … select` | Applies the cuts of `config/lc_config.ini` to the table in seconds: cut flow, spectrum, fit, plot. |

```bash
# 2-3: data and dump (once; on lxplus)
dump/delphi_download.sh -j 4 catalogue/delphi_keep_zpeak.csv /eos/user/<you>/delphi_zpeak
find /eos/user/<you>/delphi_zpeak -mindepth 2 -type f ! -name '*.part' ! -path '*/lists/*' | sort > all_list.txt
source /cvmfs/delphi.cern.ch/setup.sh
dump/delphi_dump.sh -m lc -p 0.4 -j 4 -b <build dir> -w /tmp/$USER/dump_jobs -o /eos/user/<you>/dump_lc_zpeak all_list.txt
# 4-5: edit config/lc_config.ini (data path), then
python src/lc_run.py config/lc_config.ini reco        # ~15-25 min for 921 files with nproc = 8 (estimate)
python src/lc_run.py config/lc_config.ini select
```

## Playing with the selection: `config/lc_config.ini`

* `[general]`, `[reco]` need a re-run of the **reco** stage: data path, excluded data sets, loosest PID tag stored,
  track-error scale, loosest X_E / P(χ²) kept.
* `[select]`, `[fit]` are **instant** and can only tighten what the table holds: PID tags for p and K, X_E, L,
  P(χ²), helicity cut, charge, b-tag split, event-parity hold-out, mass range, bins, fit width, signal window and sidebands.
* An empty value means the paper's value. One-off change:
  `python src/lc_run.py config/lc_config.ini select --set select.tag_p=3 --set select.label=_tight`
* `--set general.channel=dplus` (with its own `reco` run) gives the D⁺ → Kππ validation channel.
* `python src/lc_select.py <out>_candidates.csv.gz --scan` tunes the cuts on even events and shows them on odd ones.

## What was checked

* **Conventions found from the data:** the 15 `w` columns are the **inverse covariance** of (d0, z0, θ, φ, 1/R), packed
  lower triangle; `d0`, `z0` are measured from the **origin**, not the primary vertex (`ip_pv`, `zip_pv` are PV-referenced);
  `trklen` is always negative (use |trklen|); the track errors are about **1.4× too small** (`tools/lc_pull_check.py`).
* **D\*⁺ → D⁰π⁺ (50 files, no PID, no vertex):** ΔM peak at 145.4 MeV, 117 ± 20 events over the wrong-sign control.
* **Vertex fit on D⁰:** only 0.8% of candidates below L/σ_L = −3; the excess is at positive L.
* **D⁺ → K⁻π⁺π⁺ (40 files):** excess 307 ± 31 (11σ) against about 290 expected from the paper's rate; flat control.
* **Λc (921 files, 1991/92 excluded):** paper cuts, loose tags: excess 869 ± 120 (6.3σ); with P(χ²) > 0.01, L > 0.05 cm: 780 (7.5σ);
  fit 690–915 ± 120–170 depending on the width. Tight PID did not help on the held-out half.

## Caveats

* The yield is about **2×** the paper's 336 ± 42 (looser PID, no tr(P_ev) fit, background structure under the peak).
* **Width and mass are not determined:** a free width is unstable on this statistics (σ = 9–20 MeV); the fitted mean ranges
  from 2.288 to 2.294 GeV with the settings. Do not quote a Λc mass.
* The excess has a **long decay-length tail** (mean 0.21 cm against 0.05–0.09 cm expected for a pure Λc). Split by the event
  b-tag it is long in b-rich events (564 ± 81 events, mean L 0.25 cm) and Λc-like in b-poor ones (98 ± 55, 0.065 cm): this
  looks like Λc from b decays. The b-tag here includes the candidate's own tracks, so the split is only indicative.
* No simulation: no efficiencies, no 3D fit in (mass, X_E, tr(P_ev)), hence no R_c·P·BR.
* The dump contains duplicate (run, event) numbers; the code keys events by their position in the file, and the selection
  removes exact duplicate candidates.
* The real-data numbers above come from runs on lxplus; the sandbox tests use synthetic data only.

## Licence

None chosen yet: add one before publishing. The DELPHI open data have their own terms on opendata.cern.ch.
