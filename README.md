# WCTE NiCf calibration analysis

Analysis pipeline for the NiCf (Ni + <sup>252</sup>Cf) neutron-capture calibration
source of the Water Cherenkov Test Experiment (WCTE) at CERN, in the
Hyper-Kamiokande programme.

The pipeline takes WCTE offline ROOT data and WCSim Monte Carlo for the same
source position and produces two calibration observables:

| Observable | What it measures | Stage |
|---|---|---|
| **Relative quantum efficiency** `RQE_i` | Per-channel hit rate in data divided by the same rate in simulation. Quantifies how far each PMT is from its simulated efficiency. | `qe` |
| **Angular response** `ε(θ)` | Data/MC hit rate as a function of the photon incidence angle on the photocathode, per mPMT type. Tests whether the simulated angular acceptance describes the detector. | `reco` + `angular` |

Both are ratios of data to simulation, and both rely on the same principle:
**data and MC are put through byte-for-byte the same selection**, so that
everything common to the two samples (trigger efficiency, capture physics, cut
acceptance) cancels and only the detector-response difference survives.

---

## 1. Physics summary

The NiCf source emits neutrons from spontaneous <sup>252</sup>Cf fission. A
neutron thermalises in the water and is then captured either

* **on hydrogen**, emitting a single 2.2 MeV gamma, or
* **on the nickel** of the source capsule, emitting a cascade of gammas summing
  to about 9 MeV.

The gammas Compton-scatter in the water; electrons above the Cherenkov threshold
radiate, and the light is detected by the 106 mPMTs (19 3-inch Hamamatsu R14374
PMTs each, 2014 channels in total).

The data are recorded in 500 µs readout windows, so the capture signal has to be
found inside a long stream of dark-noise hits. Three ingredients make that
possible:

**Time-of-flight correction.** The detector is not spherically symmetric and the
light is emitted from Compton vertices spread over tens of centimetres, so raw
hit times of one capture are not coherent. Every hit time is corrected by the
straight-line flight time from the source to the PMT that recorded it,

$$t_i^{\text{ToF-corr}} = t_i^{\text{meas}} - \frac{|\vec r_{\mathrm{PMT},i} - \vec r_{\text{source}}|}{c/n_{\text{water}}}$$

with `n_water = 1.33`. After this correction the hits of a capture pile up in
time and the background does not.

**Greedy nHits trigger.** A 20 ns sliding window scans the corrected times; a
candidate is seeded when at least `thresh_min = 2` hits fall inside it, and is
then extended window by window while further hits keep arriving. The window
length is set by the detector size: the longest straight path inside the tank is
about 4.07 m, i.e. roughly 18 ns of flight time, so a 20 ns window cannot split
one capture in two.

**tRMS cut.** The RMS of the corrected hit times inside a candidate separates
signal from noise. Real captures give a narrow peak; uncorrelated dark noise
gives a broad flat distribution. The standard selection is `tRMS < 2 ns` plus
`nhits <= 50` to remove rare large pile-up clusters.

A secondary structure at higher tRMS was traced, using MC truth, to nickel
captures: several gammas produce several Compton vertices separated by tens of
centimetres, which broadens the time spread relative to the single-vertex
hydrogen capture. It is signal, not an instrumental artefact, but the standard
cut keeps the cleaner hydrogen-dominated population.

For the angular response the emission point is refined further: each candidate
is passed through the collaboration multilaterator, which fits a vertex
`(x, y, z, t0)` from the hit times, and the incidence angle is then computed from
that vertex rather than from the source position.

---

## 2. Repository layout

```
.
├── run_pipeline.sh                      single entry point (all stages)
├── requirements.txt
├── config/
│   ├── env.sh.example                   site paths and analysis defaults
│   └── positions.csv                    source position per run
├── data/
│   ├── wcte_v11_20250513.json           PMT positions + normals, mm, WCTE frame
│   ├── geofile_NuPRISMBeamTest_16cShort_mPMT.txt   tube -> (slot,pos), cm, WCSim
│   ├── other_mpmt_info.dict             mPMT classification (pickle, as used)
│   └── other_mpmt_info.csv              same, as plain text
├── external/                            vendored collaboration code, see its README
│   ├── multilat_vertex_reconstruction.py
│   ├── functions_multilateration.py
│   ├── functions_bonsai.py
│   └── multilat_smoke_test.py           checks the reconstruction and its units
└── src/
    ├── functions.py                     shared library: I/O, ToF, trigger, observables
    ├── pipeline_utils.py                shared helpers (file globs, time matching, frames)
    ├── run_data_hits.py                 stage 1: per-PMT data summary
    ├── merge_data_hits_npz.py           stage 1b: merge parallel partials
    ├── run_qe.py                        stage 2: relative quantum efficiency
    ├── create_file_for_multilateration.py   stage 3a: data candidate clusters
    ├── build_mc_clusters_for_reco.py    stage 3b: MC candidate clusters
    └── run_angular_vertex.py            stage 4: vertex-based angular response
```

### External code this pipeline depends on

Everything needed is in the repository except two WCTE packages that cannot be
installed with pip:

* **`WCSimFilePackages`** — `npz_to_df.truehits_info_to_df`, the WCSim `.npz`
  reader;
* **`WCTE_BRB_Data_Analysis`** — `wcte.brbtools.sort_run_files` and
  `get_part_files`, the ROOT part-file helpers.

Set `WCTE_SOFTWARE_DIR` to a checkout containing both; the scripts add it to
`sys.path` themselves.

The multilaterator in `external/` is collaboration code that has been vendored
here for convenience. Read `external/README.md` before publishing: it lists the
portability patches applied and the attribution that still has to be settled.

Python 3.9 or newer:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

> A note on parquet engines: writing the large hit-level DataFrames with
> `fastparquet` produced `Corrupted thrift data` on the Lustre scratch
> filesystem. Use `pyarrow`. The current pipeline avoids monolithic parquets
> altogether, which is the more robust fix.

---

## 3. Configuration

Two files hold everything that changes between machines and between runs.

### `config/env.sh`

Copy `config/env.sh.example` to `config/env.sh` and edit the paths. It also
holds the analysis defaults (trigger window, cuts, number of part-files). Any of
them can be overridden per invocation:

```bash
N_PARTS=5 TRMS_CUT=1.5 ./run_pipeline.sh 1767
```

`config/env.sh` should stay out of version control; only the `.example` is
tracked.

### `config/positions.csv`

One row per signal run: background run, label, and the source position in **both
coordinate frames**.

### Coordinate frames — read this before adding a run

The pipeline uses two frames and never converts between them:

| Frame | Units | Where it is used | Column |
|---|---|---|---|
| **WCTE** | mm | PMT positions in `wcte_v11_*.json`, multilaterator vertices, ToF correction of the **data** | `src_mm` |
| **WCSim** | cm | MC hit positions in the `.npz`, ToF correction of the **MC** | `src_cm` |

`src_mm` and `src_cm` are the same physical point expressed in two different
frames, **not** the same number in two units. They therefore have to be derived
independently — `src_mm` from the CDS deployment coordinates
(`(R, φ, Z)_CDS → (x, y, z)_WCTE`, with the vertical offset that maps
CDS `Z = 0` to WCTE `y = 1525 mm`), and `src_cm` from the position written in the
WCSim macro used to generate that MC production.

**Open item.** In the current table the two entries satisfy
`src_mm = 10 × src_cm` for runs 1769, 2336 and 2337, but not for run 1767
(1525 mm versus 80 cm = 800 mm). Both conventions cannot be correct
simultaneously: either the WCSim geometry origin differs between the older
(1767) and newer productions, or one of the pairs is wrong. Since `src_mm`
drives the data ToF correction, an error there directly biases the tRMS and thus
the candidate selection. Re-derive and confirm each pair before publishing, and
record the derivation in the `notes` column.

A cheap consistency check: reconstruct the vertices of a run
(`STAGE=reco`) and compare the mean reconstructed vertex with the tabulated
`src_mm`, after converting the vertices to the WCTE frame (below). They should
agree to within the vertex resolution.

### A third frame: the reconstructed vertices

The multilaterator takes its PMT positions from the WCSim geofile and works in
cm, so **its output vertices are in WCSim centimetres**, not in WCTE
millimetres. The two frames are related by

```
r_WCTE[mm] = 10 * r_WCSim[cm] + (0, 425, 0) mm
```

The offset is not hardcoded: `pipeline_utils.wcsim_cm_to_wcte_mm_offset()`
fits it at run time from the 1843 channels the two geometry files have in
common, and reports the per-channel residual (about 3 cm RMS, from the different
mPMT dome modelling in the two descriptions). The value it recovers,
`y = +425 mm`, is consistent with the tank centre sitting at `y = 424 mm` in the
WCTE frame.

`run_angular_vertex.py` applies the conversion on load, for data and MC alike,
controlled by `--vertex-frame` (default `wcsim-cm`). To reproduce figures made
before this was handled, pass `--vertex-frame wcte-mm`, which uses the CSV
numbers unconverted.

You can verify the frame empirically without any real data:

```bash
python external/multilat_smoke_test.py \
    --geo-file data/geofile_NuPRISMBeamTest_16cShort_mPMT.txt
```

It reconstructs synthetic clusters generated from a vertex at
`(0, 80, 0)` cm and recovers it in centimetres.

---

## 4. Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp config/env.sh.example config/env.sh   # edit RAW_DATA_DIR, MC and OUTPUT paths

# full chain for one source position, serial
./run_pipeline.sh 1767

# same, with one background job per data part-file
STAGE=all-parallel MAX_DATA_HITS_JOBS=5 ./run_pipeline.sh 1767

# individual stages
./run_pipeline.sh 1767 data-hits
./run_pipeline.sh 1767 qe
./run_pipeline.sh 1767 reco
./run_pipeline.sh 1767 angular
```

On SLURM:

```bash
sbatch --cpus-per-task=5 --mem=96G --time=48:00:00 \
       --export=ALL,STAGE=all-parallel,N_PARTS=25,MAX_DATA_HITS_JOBS=5 \
       ./run_pipeline.sh 2337
```

All results land in `${OUTPUT_ROOT}/<run>_<n_cos_bins>bins/`.

---

## 5. The pipeline, stage by stage

```
  ROOT part-files                          WCSim .npz chunks
        │                                          │
        ▼                                          │
  run_data_hits.py  ──►  data_hits_R<run>.npz      │
        │              (per-PMT counts, SIG-BKG)   │
        │                                          │
        ├──────────────►  run_qe.py  ◄─────────────┤
        │                     │                    │
        │                     ▼                    │
        │              relative_qe.parquet         │
        │                                          │
        ▼                                          ▼
  create_file_for_             build_mc_clusters_for_reco.py
  multilateration.py                     │
        │                                │
        ▼                                ▼
  candidates_for_reco.csv      mc_candidates_for_reco.csv
        │                                │
        └────────►  multilat_vertex_reconstruction.py  ◄────────┘
                             │
                             ▼
                  *_multilat_chi2.csv  (vertices)
                             │
                             ▼
                   run_angular_vertex.py
                             │
                             ▼
              angular_response_vertex.csv + figures
```

### Stage 1 — `run_data_hits.py` (data summary)

Reads the signal and background runs part-file by part-file, applies the ToF
correction, runs the trigger, applies the cuts and reduces everything to
length-2014 arrays.

Two points make this stage correct rather than merely convenient:

* **Common events.** Only readout windows present in both the signal and the
  background run enter the subtraction, so the two samples cover the same live
  time. `event_id = window_index + part_index × 10⁶` keeps window *N* of part
  *P* aligned between the two runs.
* **Additivity.** Every stored quantity is a sum over disjoint windows, which is
  what allows the parallel mode (one job per part-file) and the later merge.

The full 25 part-file dataset never exists as a hit-level DataFrame; only one
part-file is resident at a time. This replaces the earlier monolithic
`df_sig`/`df_bkg` parquets, which ran out of memory beyond about five parts.

Output: `data/data_hits_R<run>.npz` — `sig_hits`, `bkg_hits`, `pure_signal`,
candidate counts, `n_events_common`, per-PMT charge sums, and the cuts applied.

Parallel mode writes `data_hits_R<run>_part<NNN>.npz`; `merge_data_hits_npz.py`
sums them and refuses to mix partials produced with different cuts.

### Stage 2 — `run_qe.py` (relative quantum efficiency)

```
RQE_i = [ N_i(data) / N_cand(data) ] / [ N_i(MC) / N_cand(MC) ]
```

The data side comes from stage 1. The MC side is processed chunk by chunk
through the *same* ToF correction, the *same* trigger and the *same* cuts, then
mapped from WCSim tube numbers to WCTE `pmt_id`. Normalising each side by its
own candidate count means the two samples need not have equal statistics.

Errors are Poisson on both sides,
`σ(RQE) = RQE · sqrt(1/N_data + 1/N_MC)`.

Outputs: `figures/background_subtraction.png`, `figures/relative_qe.png`,
`figures/relative_qe_projection.png`, `data/relative_qe.parquet`
(`pmt_id`, `relative_qe`, `error_qe`, `data_hits`, `mc_hits`).

**Statistics.** The MC is normally the limiting sample. With ~9×10⁵ data
candidates against ~4.6×10⁴ MC candidates, the per-channel error is dominated by
the simulation and adding data part-files does not narrow the RQE distribution.
Doubling the MC statistics reduced the spread from 0.334 to 0.305, well short of
the 1/√2 expected for a purely statistical width, which indicates a systematic
floor.

**How much of the RQE width is real?** Measuring the same channels in two runs
with the source at very different positions (1767, near the top endcap, and
2336, low in the barrel) and correlating the two results gives a Pearson
coefficient of about 0.56 over 1532 channels. Roughly a third of the observed
variance is a reproducible property of the channel and the rest is measurement
error, so per-channel RQE values carry an uncertainty comparable to the
channel-to-channel spread. Population averages (In-situ versus Ex-situ, TRIUMF
versus WUT) are robust; individual channel values should be quoted with that
caveat.

### Stage 3 — candidate clusters and vertex reconstruction

`create_file_for_multilateration.py` (data) and `build_mc_clusters_for_reco.py`
(MC) both write the same CSV format, one row per candidate:

```
candidate_id, event_id, hit_times_ns, hit_slot_ids, hit_channel_ids, nhits, trms
```

Two details matter:

* **Raw times go into the CSV.** The multilaterator fits a vertex and derives the
  time of flight itself, so it must receive uncorrected times. The *selection*
  still uses the ToF-corrected `trms`, which is what makes the 2 ns cut work.
* **Same selection on both sides.** `run_pipeline.sh` passes one set of trigger
  and cut values to both scripts. Any asymmetry here would appear as a fake
  angular effect later.

Both CSVs are then reconstructed with the **same** multilaterator, so the
reconstruction bias is common to data and MC and cancels in their ratio.

MC candidate IDs are offset by `chunk_index × 10⁸`, where `chunk_index` is the
position of the `.npz` in the sorted file list. `build_mc_clusters_for_reco.py`
writes that list to `<out-csv>.filelist.txt`; `run_angular_vertex.py` rebuilds it
from the same `--mc-npz` specification. If the two ever disagree, the MC
vertices are silently attached to the wrong candidates, so keep the two
invocations identical — which `run_pipeline.sh` does by construction.

### Stage 4 — `run_angular_vertex.py` (angular response)

The hit count on channel *i* is modelled as

```
N_i = Σ_s [ I_s(θ_s,i) / L²_s,i ] · ε_i(θ_PMT) · exp(−L_s,i / λ)
```

summed over Compton vertices *s*, with `L_s,i` the vertex-to-PMT distance. For
each hit the script computes, from the reconstructed vertex of its candidate,
the incidence cosine on the photocathode and the distance `L`, then histograms
hits against `cos θ`, one histogram per mPMT category. The response per category
is the data/MC ratio of those histograms, renormalised so that the
highest-`cos θ` bin equals 1.

The geometric weight `I_s/L² · exp(−L/λ)` is not applied: it is assumed to cancel
between data and MC inside each `cos θ` bin. Two options exist to test that
assumption:

* `--use-weights` fills the histograms with `exp(−L/λ)/L²` instead of counting
  hits. Agreement between weighted and unweighted results supports the
  cancellation.
* `figures/weight_proxy_check.png` compares the distribution of that weight
  between data and MC bin by bin.

Relevant options:

* `--min-L-mm` (default 1000) drops hits with a short vertex-to-PMT distance on
  **both** sides. At small `L` the `1/L²` weight diverges and the
  vertex-reconstruction error dominates the angle; this cut is the
  collaboration's suggested fix for the run-to-run discrepancy.
* `--cos-min` (default 0) keeps only the forward hemisphere. Light arriving on
  the back of a photocathode is either a reflection or a reconstruction failure.
* `--use-source-pos` recomputes everything from the fixed source position
  instead of the per-candidate vertex, reproducing the older point-source
  `N·R²` method as a cross-check. It requires `--source-pos-mm`.

Outputs: `data/angular_response_vertex.csv` (bin centres, per-category data and
MC contents with their `sumw2`, response and error),
`data/angular_cut_flow_summary.{csv,json}`, and four figures — the response
itself, the normalised and raw `cos θ` histograms, and the weight-proxy check.

---

## 6. Method notes worth keeping in mind

**Why the candidate count, not the event count, normalises the RQE.** The two
samples are compared as hit rates per selected candidate. The pure-signal
denominator is `n_cands_sig − n_cands_bkg`, the difference of the counts entering
the subtracted numerator, so numerator and denominator refer to the same
population.

**The background is subtracted per channel, not globally.** After the tRMS cut
the residual background is small but not spatially uniform, so a single scalar
rate would bias the channels far from the source.

**Charge calibration.** `q_sum_per_pmt` and `q_count_per_pmt` are accumulated
*before* the tRMS and nhits cuts, so `mean_charge_per_pmt` stays a gain
observable and is not sculpted by the signal selection.

**Bin errors.** `fill_hist_by_category` stores `Σw²` alongside the bin content.
Unweighted, `Σw² = N` and the error reduces to the Poisson `√N`; weighted, it
gives `√(Σw²)` correctly.

---

## 7. Known limitations and open points

1. **Coordinate-frame table.** See section 3. The `src_mm`/`src_cm` convention is
   not uniform across runs and must be re-verified.
2. **Vertex frame.** Until this packaging pass, `run_angular_vertex.py` used the
   multilaterator vertices as if they were already in WCTE millimetres. They are
   in WCSim centimetres, so `L` and `cos θ` mixed units and frames — a shift of
   roughly 1.1 m in the assumed emission point. The conversion is now applied by
   default (`--vertex-frame wcsim-cm`); any angular-response figure produced
   before this has to be regenerated. The point-source cross-check
   (`--use-source-pos`, which takes `--source-pos-mm` in the WCTE frame) and the
   relative QE were never affected.
3. **Railed vertices.** The multilaterator bounds the fit at ±300 cm and returns
   `fit_success = True` even when a fit sits on the bound, far outside the tank
   (r ≤ 157 cm, |y| ≤ 139 cm). Those vertices are now dropped by default; pass
   `--keep-railed-vertices` to look at them.
4. **MC hit multiplicity is treated differently in the two stages.** The RQE
   counts each channel once per MC candidate (`all_pmts` holds unique tube
   numbers), which mirrors the front-end merging photoelectrons that arrive
   within a few ns — in data the unique-PMT/hit ratio inside a candidate is
   1.000. The angular stage instead rebuilds per-hit rows from `all_times`, so a
   doubly-hit tube counts twice. The effect is small but the two stages should be
   made consistent.
5. **The WCSim geometry has 1843 channels, the detector 2014.** Nine mPMTs
   present in the JSON are absent from the geofile, so those channels have no MC
   counterpart and no relative QE (hence ~1550 valid channels rather than 2014).
   They also drop out of the vertex fit, which silently uses fewer hits than the
   candidate contains.
6. **Background subtraction in the low-memory angular path.** With
   `--data-candidates-csv`, the background comes from the per-PMT counts in
   `data_hits_R*.npz`. Those were produced with the tRMS and max-nhits cuts but
   *without* the `nhits >= 6` requirement applied to the signal candidates here,
   and over the SIG/BKG common events rather than over the candidates in the CSV.
   The background is therefore slightly over-subtracted. The effect is confined
   to the lowest-statistics bins because the response is renormalised at
   `cos θ = 1`, but the parquet path (`--sig-parquet`/`--bkg-parquet`) remains
   the reference for small samples.
7. **Normalisation at the last bin.** The angular response is divided by its
   highest-`cos θ` bin, so a noisy last bin rescales the whole curve. Check
   `figures/nhits_vs_costheta_raw.png` before quoting numbers; averaging the top
   few bins instead would be more robust.
8. **MC statistics limit the RQE.** See stage 2. More WCSim events, not more
   data part-files, is what narrows the per-channel error.
9. **Delaminated mPMT categories.** `run_angular_vertex.py` will split out two
   "Delaminated" groups if `other_mpmt_info.dict` carries a `delaminated` flag.
   The dictionary shipped in `data/` does not (its 93 entries carry only
   `mpmt_type`, `mpmt_site` and `led_pos`), so those categories always come out
   empty; they are harmless placeholders in the output CSV.
10. **ID offset headroom.** Data event and candidate IDs are made unique with a
   10⁶ multiplier per part-file. A typical part yields ~1.7×10⁴ windows and
   ~7×10⁵ candidates, so the margin is under a factor of two. Both
   `run_data_hits.py` and `create_file_for_multilateration.py` now abort if a
   part-file overflows it rather than silently producing colliding IDs. If that
   ever fires, raise the constant and regenerate the candidate CSV **together
   with** its multilaterator output.
11. **Per-hit MC re-extraction.** `run_mc_trigger` stores a candidate's times and
   its *unique* tube numbers, which loses the time↔tube pairing, so both MC
   stages recover it by matching times back to the event hit table. This is
   exact but it is the slowest part of the MC processing. Storing the per-hit
   indices in `functions.run_mc_trigger` would remove the step entirely.

---

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `TypeError: concatenate() got an unexpected keyword argument 'casting'` | Old numpy/awkward combination (Python 3.7). Use the Python 3.9 environment. |
| `Corrupted thrift data` when reading a parquet | `fastparquet` on Lustre. Use `pyarrow`, or better, use the `--data-hits-npz` path, which writes no large parquet. |
| `No .npz files matched` | The MC glob was expanded by the shell. Quote it, or set `MC_NPZ_DIR`/`MC_NPZ_PATTERN` in `config/env.sh`. |
| `run <N> is not in config/positions.csv` | Add the row, with the source position in both frames. |
| Many MC candidates reported as having no reconstructed vertex | The MC candidate CSV and the MC vertex CSV were produced from different `--mc-npz` lists, so the chunk offsets differ. Compare `mc_candidates_for_reco.csv.filelist.txt` with the pattern used in the angular stage. |
| RQE distribution does not narrow when adding data | Expected: the MC sample is the limiting one. Generate more WCSim events. |
| Reconstructed vertices pile up at exactly ±300 | Fits railed at the multilaterator bound. They are dropped by default in the angular stage; see `--vertex-bound-cm`. |
| Angular response looks different from an older plot | The vertex frame conversion is now applied. Pass `--vertex-frame wcte-mm` (or `VERTEX_FRAME=wcte-mm`) to reproduce the old, unconverted behaviour. |
| `ModuleNotFoundError: WCSimFilePackages` | `WCTE_SOFTWARE_DIR` does not point at a checkout containing it. |

---

## 9. Reproducing the thesis results

For each source position (1767, 1769, 2336, 2337):

```bash
STAGE=all-parallel MAX_DATA_HITS_JOBS=5 ./run_pipeline.sh <run>
```

with `N_PARTS=25`, `WINDOW=20`, `THRESH_MIN=2`, `TRMS_CUT=2.0`,
`MIN_HITS=6`, `MAX_NHITS=50`, `MIN_L_MM=1000`, `N_COS_BINS=100` — the defaults in
`config/env.sh.example`. The relevant figures are

* `figures/background_subtraction.png` — signal, background and their difference
  per channel;
* `figures/relative_qe.png`, `figures/relative_qe_projection.png` — RQE per
  channel and its distribution per mPMT type;
* `angular_vertex/figures/angular_response_vertex.png` — the angular response;
* `angular_vertex/figures/weight_proxy_check.png` — the geometric-weight
  cancellation check.

---

## 10. Authorship and contact

<!-- Fill in before publishing. -->

Written as part of a doctoral thesis on the WCTE water Cherenkov detector and
its radioactive calibration sources (NiCf, AmBe), within the WCTE and
Hyper-Kamiokande collaborations.

* Author: *&lt;name, institute, e-mail&gt;*
* The vertex-based angular-response method implemented in
  `run_angular_vertex.py` follows the approach presented by
  *&lt;originator&gt;* to the collaboration.
* The multilaterator used for vertex reconstruction is external to this
  repository; credit it where appropriate.
* License: *&lt;choose one before making the repository public&gt;*
