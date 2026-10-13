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
src/         lc_schema.py  lc_vertex.py  lc_reco.py  lc_cuts.py  lc_select.py  lc_run.py  lc_batch.py  lc_rootio.py     the analysis
config/      lc_config.ini  lc_scan.ini                             every selection you can change; the variations of a scan
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

* `general.out` is a prefix **relative to the directory you run from**. Run from the directory that holds the candidate table,
  or use `--workdir DIR` (also for `lc_batch.py`), or `--set general.out=/full/path/prefix`.
* `[general]`, `[reco]` need a re-run of the **reco** stage: data path, excluded data sets, loosest PID tag stored,
  track-error scale, loosest X_E / P(χ²) kept.
* `[select]`, `[fit]` are **instant** and can only tighten what the table holds: PID tags for p and K, X_E, L,
  P(χ²), helicity cut, charge, b-tag split, event-parity hold-out, mass range, bins, fit width, signal window and sidebands.
* **Fit range and background:** `[fit] fit_lo`, `fit_hi` (GeV; the fit uses the plotted bins whose centres lie in that range, so the
  edges follow the bin grid) or, if empty, the peak ± `fit_half_range`; `bkg_order` = polynomial order of the background (1 linear,
  2 = pol2, 3 = pol3, …). Compare orders and ranges as a fit systematic: `lc_scan.ini` varies both.
* An empty value means the paper's value. One-off change:
  `python src/lc_run.py config/lc_config.ini select --set select.tag_p=3 --set select.label=_tight`
* `--set general.channel=dplus` (with its own `reco` run) gives the D⁺ → Kππ validation channel.
* Each `select` run writes `<out>_select<label>.png`: a **square** plot with the data (8 MeV bins by default, `fit.bins = 50`)
  and the fit only, drawn as a smooth curve (Gauss + quadratic background, background dashed), with a panel of the fit metrics:
  N_sig ± error, mean, σ, S and B inside mean ± 2σ, S/B, S/√(S+B), N_sig/δN, χ²/ndf with its p-value, and the sideband-counting
  excess. S/B and S/√(S+B) use the fitted background; N_sig/δN also includes the fit's uncertainty on it, so it is the more
  conservative number. The same numbers are in `<out>_select<label>_summary.txt`.
* It also writes `<out>_select<label>.root` (`fit.root_output = yes`): `TH1F h_mass` (the plotted spectrum, 8 MeV bins) and
  `h_mass_ctrl`, the smooth fit as finely binned `TH1F`s (`h_fit_total`, `h_fit_bkg`, `h_fit_sig`), a TTree `fit_metrics` (one entry)
  and a TTree `candidates` (mass, xe, L, P, cosh, q, run, evt, ctrl flag). PyROOT gives real `TH1F`; without it `uproot` is used
  (histograms become `TH1D`). If neither can be imported the rest is still written and a message says so.
* `python src/lc_select.py <out>_candidates.csv.gz --scan` tunes the cuts on even events and shows them on odd ones; `lc_batch.py` (below) keeps every configuration.

## Scanning many selections: `src/lc_batch.py`

Runs 20–50 selection settings in a loop on the stored candidate table (seconds to minutes) and keeps the signal-extraction
metrics of each one:

```bash
python src/lc_batch.py config/lc_config.ini config/lc_scan.ini        # baseline from lc_config.ini, variations from lc_scan.ini
```

* `lc_scan.ini` `mode`: `oat` (baseline + one value of one cut at a time, 29 configurations in the shipped file), `grid`
  (all combinations, cut to `max_configs` by a random subset), `random`, or only the `[config:NAME]` sections (`explicit`).
  Any `[select]`/`[fit]` key can be varied, plus `btag = all,brich,bpoor`.
* Per configuration, one row of the DataFrame: the effective cuts; selected physical and control candidates; the counting
  excess with its error, `signif_sb` = S/√(S+B) and `signif_err` = excess/error, S/B; a control-sample null test (`ctrl_signif`);
  the excess and significance of the even and odd event halves; the mean decay length of the excess; a Gaussian + quadratic fit
  with fixed width and with free width (yield, error, mean, σ, χ²/ndf, p-value, significance, S, B, S/B and S/√(S+B) in mean ± 2σ); signal and background retention
  relative to the baseline.
* Outputs `<out>_batch_*`: `results.csv` and `results.pkl` (the DataFrame), `hists.npz` (mass spectra), `candidates.npz`
  (selected candidates of every configuration), `overview.png`, and `.root`: a `TH1F` per configuration (`h_cfgNNN`, control
  `hctrl_cfgNNN`), a TTree `results` (one entry per configuration) and a TTree `candidates` (cfg, ctrl, mass, xe, L, P, cosh, q,
  run, evt for every selected candidate).
* ROOT writing uses PyROOT (`TH1F`) if it can be imported, otherwise `uproot` (`pip install uproot`; histograms become `TH1D`).
  The pandas outputs are written first and never depend on ROOT; if there is no ROOT backend where the scan ran, convert later
  with `python src/lc_rootio.py <out>_batch` in an environment that has one. **The ROOT writers (here and in `select`) have not been run against a
  real ROOT or uproot** (neither was installable in my sandbox); the tests only check what they are handed.
* Ranking configurations by significance on the same data biases the winner. The even/odd columns are independent halves: a
  real improvement shows in both.

```python
import pandas as pd
df = pd.read_pickle("lc_full_batch_results.pkl")
df.sort_values("signif_sb", ascending=False)[["label", "excess", "excess_err", "signif_sb", "signif_even", "signif_odd"]].head(10)
```
```
root -l lc_full_batch.root
root [1] results->Scan("cfg_id:excess:signif_sb:signif_odd")
root [2] h_cfg000->Draw()
root [3] candidates->Draw("mass", "cfg==0 && ctrl==0")
```

## Data structure (the CSV files)

The dump CSVs have **no header row**: load them with the column names of `src/lc_schema.py`.

```python
import sys; sys.path.insert(0, "src")
import pandas as pd
from lc_schema import EVENT_COLS, TRACK_COLS
ev = pd.read_csv("Y13710.125.events.csv", names=EVENT_COLS)
tr = pd.read_csv("Y13710.125.tracks.csv", names=TRACK_COLS)
```

### 1. Dump, one pair per input file: `<name>.events.csv` and `<name>.tracks.csv`

**`events.csv`**: one row per hadronic event (at least 5 Team-4 charged tracks), 37 columns.

| Columns | Meaning |
|---|---|
| `run`, `evt` | Run and event number. The pair is **not unique** within a file, so events and tracks are linked by order instead. |
| `ihad4` | Team-4 hadronic tag (1 = hadronic). |
| `nctr4`, `nctrk`, `nvecp` | Team-4 charged multiplicity, total charged multiplicity, number of particles in SKELANA's vector. |
| `nwrit` | Number of rows this event owns in `tracks.csv`: the tracks of event *i* are the next `nwrit` rows. |
| `ecm`, `echar` | √s and total charged energy [GeV]. |
| `pv_x`, `pv_y`, `pv_z`, `pv_chi2` | Primary vertex [cm] and its fit χ². |
| `pv_xx`, `pv_xy`, `pv_yy`, `pv_xz`, `pv_yz`, `pv_zz` | Covariance of the primary vertex. |
| `bs_x/y/z`, `bs_sx/sy/sz` | Beam-spot position and size [cm]. |
| `btn1..3`, `btp1..3`, `bts1..3` | b-tag probabilities for hemisphere 1, hemisphere 2 and the whole event, from negative-, positive- and all-impact-parameter tracks. A small `btp3` means lifetime-like. |
| `thr1..3` | Thrust axis from the b-tag package (x, y, z). |

**`tracks.csv`**: one row per charged track with p ≥ 0.4 GeV (the `-p` of the dump), 54 columns.

| Columns | Meaning |
|---|---|
| `run`, `evt`, `idx` | Event, and the track's index in SKELANA's vector (charged tracks come first). |
| `q` | Charge ±1. |
| `lock` | Bit 1 = failed SKELANA's track selection; bit 2 = locked as a secondary-vertex daughter. Values 0–3. |
| `px`, `py`, `pz`, `E`, `m` | Momentum [GeV] and energy; `m` is the pion mass assumed for `E`. |
| `d0`, `z0` | Rφ and z impact parameters [cm], **measured from the origin (0,0,0)**, not from the primary vertex. |
| `theta`, `phi` | Perigee angles [rad]; not wrapped to the physical range, so use the momentum for directions. |
| `curv` | Signed curvature 1/R [cm⁻¹]; `sign(curv) = −q`. |
| `w1..w15` | **Weight matrix = inverse covariance** of (d0, z0, θ, φ, 1/R), stored as the lower triangle row by row: `w1`=(1,1), `w2`=(2,1), `w3`=(2,2), … The diagonals are `w1`, `w3`, `w6`, `w10`, `w15`. |
| `trklen` | Track length [cm]; **negative for every track** here, use `abs(trklen)`. |
| `detmask` | Bit pattern of the detectors used. |
| `chi2_vd`, `ndf_vd`, `ndf_novd` | Track-fit χ² with the vertex detector and degrees of freedom. `chi2_novd` has unphysical values: do not use it. |
| `vd_rphi`, `vd_rz` | Vertex-detector hit counts, bit-packed per layer; a value above 0 means at least one hit. |
| `ip_pv`, `zip_pv`, `ip_bs` | Signed impact parameters [cm] relative to the primary vertex and the beam spot. |
| `eerr` | Relative momentum error (the Team-4 cut uses `eerr < 1`). |
| `h_dedx_k`, `h_dedx_p` | Kaon / proton tag from dE/dx: −1 no measurement, 0 pion-like, 1 loose, 2 standard, 3 tight. |
| `h_rich_k`, `h_rich_p`, `h_rich_pi` | RICH tags: −1 no measurement, 0 not that particle, 1 loose, 2 standard, 3 tight (pion: 0/1). |
| `h_isign`, `rich_flag` | Which detectors went into the combined tag; RICH quality status (bit patterns). |
| `xkaon`, `xproton` | Combined tags, from −1 (none) up to 4 (super tight). |
| `dedx`, `dedx_nw` | dE/dx normalised to 1 for minimum-ionising tracks; number of wires used. `dedx_w` is always 0: not usable. |

The conventions for `d0`/`z0`, the weight matrix, `trklen` and `curv` were verified on the data (`tools/`). The tag values and the
`QTRAC`/`KHAID` indices come from the SKELANA manual; `detmask`, `h_isign`, `rich_flag` and `thr` are taken from it unchecked.

### 2. Candidate table `<out>_candidates.csv.gz` (with header)

One row per three-track candidate passing the loose selection of the reco stage.

| Columns | Meaning |
|---|---|
| `run`, `evt`, `ctrl` | Event; `ctrl` = 0 physical, 1 control (wrong charge pattern). |
| `mass`, `xe`, `cosh` | Candidate mass [GeV], X_E = 2E/√s, helicity cosine of the leading particle. |
| `L`, `sigL`, `chi2`, `P` | Transverse decay length from the primary vertex [cm] and its error; vertex χ² and P(χ²) (3 dof). |
| `veto`, `q` | D\* veto (always true for the Λc); charge of the first particle (the proton for the Λc). |
| `p0`, `p1`, `p2` | Momenta of the three tracks [GeV]. |
| `echar`, `pvz`, `btn3`, `btp3`, `bts3` | Event quantities. |
| `p_h_rich_p`, `p_h_dedx_p`, `p_h_rich_k`, `p_h_dedx_k` and the same with `k_` | PID tags of the first two particles (for the D⁺ channel the prefixes are `k_` and `pi1_`). |

### 3. Batch results `<out>_batch_results.csv` (one row per configuration)

* `cfg_id`, `label`, the effective cuts (`tag_p`, `xe_min`, `l_min`, …) and fit settings (`fit_lo`, `fit_hi`, `bkg_order`, …).
* Counts: `n_phys`, `n_ctrl`, `n_window`, `sideband_est`.
* Counting metrics: `excess`, `excess_err`, `signif_sb` (S/√(S+B)), `signif_err` (excess/error), `s_over_b`; the control null test `ctrl_*`;
  the independent halves `excess_even`, `signif_even`, `excess_odd`, `signif_odd`; `mean_L_excess`.
* Fit metrics, prefix `fit_` (fixed width) and `freefit_` (free width): `ok`, `yield`, `yield_err`, `mean`, `mean_err`, `sigma_mev`,
  `chi2`, `ndf`, `pvalue`, `signif`, and S, B, S/B, S/√(S+B) in mean ± 2σ (`s_2sigma`, `b_2sigma`, `s_over_b_2sigma`, `signif_2sigma`).
* `signal_eff_vs_baseline`, `background_eff_vs_baseline`.

### 4. Catalogue (`catalogue/`)

* `delphi_keep_zpeak.csv`: `recid, name, files, GiB` of the 9 data sets that were downloaded.
* `delphi_collision_classified.csv`: all 200 portal collision sets with `recid, name, fam, yr, role, scope, keep, events, files, GiB, reason`;
  `role` says RAW / TANAGRA / subset / usable DST, `reason` why a set was kept or dropped.

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
