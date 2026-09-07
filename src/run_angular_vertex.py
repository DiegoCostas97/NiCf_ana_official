#!/usr/bin/env python3
"""
NiCf Analysis Pipeline — Angular Response (vertex-based method)
================================================================
Implements the alternative angular-response extraction proposed by the
collaboration (D. Costas Rodriguez slides), which removes the point-source
N*R^2 assumption of the previous method (run_angular.py).
 
Idea
----
The number of hits on channel i from the Ni/Cf source is modelled as
 
    N_i = sum_s  [ I_s(theta_s,i) / L_s,i^2 ] * eps_i(theta_PMT_s,i) * exp(-L_s,i / lambda)
 
where the sum runs over gamma -> e-/e+ conversion vertices s, L_s,i is the
distance from that *vertex* (not the source) to PMT i, and eps_i(theta_PMT)
is the per-PMT angular efficiency we want to extract.
 
Instead of correcting each hit by R^2 from the source (the old method), we:
  1. Use the reconstructed vertex of each candidate to compute, per hit,
     the incidence angle cos(theta_PMT) and the vertex->PMT distance L.
  2. Fill a histogram of NHits vs cos(theta_PMT), ONE per mPMT type.
     The geometric weight  I_s/L^2 * exp(-L/lambda)  is *not* applied: it is
     assumed to cancel in the data/MC ratio within each cos(theta) bin
     (valid because exp(-L/lambda) ~ 1 and Compton/pair production is well
     simulated, provided the detector geometry is correct).
  3. Build the SAME histograms for MC (Ni ball at the reconstructed-vertex
     centre, coarse ~50% WUT ex-situ correction already applied in MC).
  4. Angular response per type = (data hist) / (MC hist), normalised so that
     the cos(theta)=1 bin equals 1.
  5. Self-check: per cos(theta) bin, compare the distribution of the weight
     proxy  exp(-L/lambda)/L^2  between data and MC.
 
This script DELIBERATELY reuses the existing pipeline:
  - functions.load_pmt_positions / load_wcsim_tube_mapping / read_mc_truehits /
    apply_tof_correction_mc / run_mc_trigger / mc_cands_to_pmt_id
  - the data candidate parquet produced by run_analysis.py
  - the multilaterator vertex CSV produced by multilat_vertex_reconstruction.py
  - the other_mpmt_info.dict mPMT-type map (same as run_qe.py / run_angular.py)
 
Data inputs
-----------
--sig-parquet      : hit-level signal candidates (run_analysis.py output).
                     MUST contain the RAW (non-ToF-corrected) calibrated times,
                     since L and cos(theta) are geometric and the angle is
                     vertex-based, not source-based. We use the hit positions
                     from the geometry JSON via pmt_id.
--bkg-parquet      : hit-level background candidates (same run as run_qe.py).
--vertex-csv       : multilaterator output for the signal candidates, with
                     columns candidate_id, vertex_x, vertex_y, vertex_z
                     (and ideally fit_success / chi2_ndof for quality cuts).
--geo-json         : WCTE geometry JSON (PMT positions AND direction_z normals).
--mpmt-info        : other_mpmt_info.dict (pickle) classifying each mPMT.
 
MC inputs
---------
--mc-npz           : WCSim .npz for the SAME source position as the data run.
--mc-vertex-csv    : MC multilaterator CSV (candidate_id, vertex_x/y/z). MC
                     vertices are reconstructed with the SAME multilaterator and
                     geometry as data, so the reconstruction bias cancels in the
                     data/MC ratio. Produce it with build_mc_clusters_for_reco.py
                     followed by multilat_vertex_reconstruction.py (see below).
--geo-file         : WCSim geofile for tube_no -> (slot,pos) mapping.
--source-pos-cm    : Ni ball position in MC coordinates [cm], used ONLY to
                     reproduce the candidate selection (ToF for the trigger).
 
Preparing the MC vertices (two steps, mirroring the data flow)
--------------------------------------------------------------
    # 1) build MC candidate clusters in the data CSV format
    python build_mc_clusters_for_reco.py \
        --mc-npz   .../...pos1769...newTuning.npz \
        --geo-file .../geofile_NuPRISMBeamTest_16cShort_mPMT.txt \
        --source-pos-cm 0.0 47.6 117.58 \
        --out-csv  .../1769/data/mc_candidates_for_reco.csv \
        --trms-cut 6.0 --min-hits 6 --window 20 --thresh-min 2
 
    # 2) reconstruct with the SAME multilaterator used for data
    python multilat_vertex_reconstruction.py \
        --csv    .../1769/data/mc_candidates_for_reco.csv \
        --outdir .../1769/data --verbose
    # -> .../1769/data/mc_candidates_for_reco_multilat_chi2.csv
 
IMPORTANT — vertex reference symmetry
-------------------------------------
cos(theta) and L are computed PER CANDIDATE from the reconstructed vertex on
BOTH sides (data and MC). This is what makes the geometric weight I_s/L^2 cancel
in the data/MC ratio. Using the fixed source position for MC (an earlier version
of this script) broke that symmetry and made the weight-proxy self-check fail.

IMPORTANT — coordinate frame of the reconstructed vertices
----------------------------------------------------------
The multilaterator takes its PMT positions from the WCSim geofile, which is in
**cm** with the origin at the centre of the tank, and works internally with
c in cm/ns. Its output vertices are therefore in **WCSim cm**, NOT in the
WCTE mm frame of the geometry JSON used here for the PMT positions and normals.
The two frames are related by

    r_WCTE[mm] = 10 * r_WCSim[cm] + (0, 425, 0) mm

(the offset is fitted at run time from the channels common to both geometry
files; see pipeline_utils.wcsim_cm_to_wcte_mm_offset). Vertices are converted on
load, controlled by --vertex-frame:

    --vertex-frame wcsim-cm  (default)  convert, i.e. treat the CSV as what the
                                        multilaterator actually produces
    --vertex-frame wcte-mm              use the numbers as they are

Earlier versions of this script assumed the CSV was already in WCTE mm and used
it unconverted, which mixed frames and units in L and cos(theta). Use
--vertex-frame wcte-mm only to reproduce those older figures.

Vertices that rail at the multilaterator bounds (|x|,|y|,|z| = 300 cm, far
outside the tank) are failed fits that still report fit_success = True; they are
dropped unless --keep-railed-vertices is given.
 
Usage
-----
    python run_angular_vertex.py \
        --sig-parquet  .../1769/data/df_sig_R1769.parquet \
        --bkg-parquet  .../1769/data/df_bkg_R1766.parquet \
        --vertex-csv   .../1769/data/candidates_for_reco_multilat_chi2.csv \
        --mc-npz       .../...pos1769...newTuning.npz \
        --mc-vertex-csv .../1769/data/mc_candidates_for_reco_multilat_chi2.csv \
        --geo-json     .../wcte_v11_20250513.json \
        --geo-file     .../geofile_NuPRISMBeamTest_16cShort_mPMT.txt \
        --mpmt-info    .../other_mpmt_info.dict \
        --source-pos-cm 0.0 47.6 117.58 \
        --trms-cut 2.0 --max-nhits 50 --min-hits 6 \
        --output-dir   .../angular_vertex_1769
"""
 
import argparse
import ast
import json
import os
import sys
import pickle
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
 
from functions import (
    load_pmt_positions,
    load_wcsim_tube_mapping,
    read_mc_truehits,
    apply_tof_correction_mc,
    run_mc_trigger,
)
from pipeline_utils import (
    CANDIDATES_PER_FILE_MAX,
    N_PMTS,
    match_times_to_hits,
    resolve_npz_files,
    wcsim_cm_to_wcte_mm_offset,
)
 
# WCTE software dir (for WCSimFilePackages), same convention as run_qe.py
sys.path.append(os.path.abspath(
    os.environ.get("WCTE_SOFTWARE_DIR",
                   "/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk")
))
 
# The four (or six) mPMT categories. The first four are the standard ones used
# in run_qe.py / run_angular.py. Two delaminated groups can be appended if the
# dict provides a 'delaminated' flag (handled gracefully if absent).
CATEGORY_COLORS = {
    "TRI In-situ": "tab:blue",
    "TRI Ex-situ": "tab:green",
    "WUT In-situ": "tab:red",
    "WUT Ex-situ": "tab:orange",
    "Delaminated A": "tab:purple",
    "Delaminated B": "tab:brown",
}


# ════════════════════════════════════════════════════════════════════
# Argument parsing
# ════════════════════════════════════════════════════════════════════
 
def parse_args():
    p = argparse.ArgumentParser(
        description="Vertex-based PMT angular response (data/MC) per mPMT type")
    # data
    p.add_argument("--sig-parquet", default=None)
    p.add_argument("--bkg-parquet", default=None)
    p.add_argument("--data-candidates-csv", default=None,
                   help="Optional data candidates_for_reco.csv. If provided, "
                        "the script does not need --sig-parquet/--bkg-parquet "
                        "for the data angular histogram.")
    p.add_argument("--data-hits-npz", default=None,
                   help="Optional data_hits_R*.npz used for per-PMT background "
                        "counts when --data-candidates-csv is used.")
    p.add_argument("--vertex-csv",  default=None,
                   help="DATA multilaterator CSV with candidate_id, vertex_x/y/z. "
                        "Required UNLESS --use-source-pos is set.")
    # geometry / mPMT info
    p.add_argument("--geo-json",  required=True,
                   help="WCTE geometry JSON (positions + direction_z normals)")
    p.add_argument("--mpmt-info", required=True,
                   help="other_mpmt_info.dict pickle")
    # MC
    p.add_argument("--mc-npz",   required=True,
                   help="WCSim .npz, glob pattern, or directory of chunk .npz files")
    p.add_argument("--mc-vertex-csv", default=None,
                   help="MC multilaterator CSV (candidate_id, vertex_x/y/z), "
                        "produced by build_mc_clusters_for_reco.py + "
                        "multilat_vertex_reconstruction.py. Reconstructed with the "
                        "SAME multilaterator as data so the bias cancels in the ratio. "
                        "Required UNLESS --use-source-pos is set.")
    p.add_argument("--geo-file", required=True,
                   help="WCSim geofile for tube mapping")
    p.add_argument("--source-pos-cm", type=float, nargs=3, default=[0, 152.5, 0],
                   help="Ni ball position in MC coords [cm], used ONLY to "
                        "reproduce the candidate selection (ToF for triggering)")
    # physics / selection
    p.add_argument("--n-water",      type=float, default=1.33)
    p.add_argument("--atten-length-mm", type=float, default=100000.0,
                   help="Attenuation length lambda [mm] (~100 m). Used only for "
                        "the weight-proxy self-check; both data and MC L are in mm.")
    p.add_argument("--trms-cut",     type=float, default=2.0)
    p.add_argument("--max-nhits",    type=int,   default=50)
    p.add_argument("--min-hits",     type=int,   default=6,
                   help="Minimum hits per candidate (vertex reco needs >=6)")
    p.add_argument("--chi2ndf-cut",  type=float, default=None,
                   help="Optional max chi2/ndf on the reconstructed vertex")
    p.add_argument("--vertex-frame", choices=("wcsim-cm", "wcte-mm"),
                   default="wcsim-cm",
                   help="Coordinate frame of vertex_x/y/z in the multilaterator "
                        "CSVs. 'wcsim-cm' (default) is what the multilaterator "
                        "actually produces and is converted to WCTE mm on load. "
                        "'wcte-mm' uses the numbers unchanged and reproduces the "
                        "behaviour of earlier versions of this script.")
    p.add_argument("--vertex-bound-cm", type=float, default=300.0,
                   help="Absolute bound used by the multilaterator fit [cm]. "
                        "Vertices sitting on it are failed fits and are dropped.")
    p.add_argument("--keep-railed-vertices", action="store_true",
                   help="Keep vertices that railed at the fit bounds "
                        "(diagnostic only; they are unphysical)")
    p.add_argument("--min-L-mm",     type=float, default=1000.0,
                   help="Discard hits whose vertex->PMT distance L is below this "
                        "value [mm] BEFORE histogramming, on BOTH data and MC. "
                        "At small L the 1/L^2 point-source weight blows up and "
                        "vertex-reconstruction error dominates, which is the "
                        "suspected cause of the run-to-run discrepancy. Set to 0 "
                        "to disable the cut.")
    p.add_argument("--use-weights", action="store_true",
                   help="Fill the NHits-vs-cos(theta) histograms with the per-hit "
                        "weight proxy exp(-L/lambda)/L^2 instead of counting 1 per "
                        "hit. The bin error then becomes sqrt(sum w_i^2) rather "
                        "than Poisson sqrt(N). If the weighted result agrees with "
                        "the unweighted one, the I_s/L^2 cancellation assumption "
                        "holds.")
    p.add_argument("--use-source-pos", action="store_true",
                   help="Compute cos(theta) and L from the FIXED source position "
                        "for every hit (both data and MC), instead of each "
                        "candidate's reconstructed vertex. This is the point-source "
                        "regime of the old N*R^2 method, recast in this per-cos(theta) "
                        "histogram pipeline; useful as a cross-check against the "
                        "vertex-based result. Requires --source-pos-mm (WCTE/mm).")
    p.add_argument("--source-pos-mm", type=float, nargs=3, default=None,
                   help="Ni ball position in WCTE coordinates [mm] (same frame as "
                        "the geometry JSON PMT positions). REQUIRED when "
                        "--use-source-pos is set. NOTE this is the WCTE/mm frame, "
                        "NOT --source-pos-cm (which is the WCSim frame used only "
                        "for the trigger ToF). E.g. run 1767 -> 0 1525 0.")
    # trigger (must match the data pipeline)
    p.add_argument("--window",     type=float, default=20)
    p.add_argument("--thresh-min", type=int,   default=2)
    # binning
    p.add_argument("--n-cos-bins", type=int, default=100)
    p.add_argument("--cos-min",    type=float, default=0.0,
                   help="Only PMTs facing the source (cos theta > cos-min)")
    p.add_argument("--output-dir", type=str, default="./output/angular_vertex")
    return p.parse_args()
 
 
# ════════════════════════════════════════════════════════════════════
# Geometry helpers
# ════════════════════════════════════════════════════════════════════
 
def load_pmt_geometry(geo_json):
    """
    Read the WCTE geometry JSON and return two (N_PMTS, 3) arrays:

        pos    : PMT position [mm], WCTE frame
        normal : PMT OUTWARD normal, i.e. 'direction_z' in the JSON, the unit
                 vector pointing away from the photocathode into the water.

    Light arriving at the PMT travels roughly along -normal, which is why the
    incidence cosine in compute_cos_theta_and_L() is (-vhat).n and not vhat.n.

    Channel indexing follows the pipeline-wide convention, identical to
    functions.load_pmt_positions: pmt_id = mpmt_slot * 19 + pmt_position.
    """
    with open(geo_json, "r") as f:
        geo = json.load(f)
 
    pos    = np.full((N_PMTS, 3), np.nan)
    normal = np.full((N_PMTS, 3), np.nan)
 
    for mpmt_idx, mpmt in geo["mpmts"].items():
        for pmt_idx, pmt in mpmt["pmts"].items():
            pmt_id = int(mpmt_idx) * 19 + int(pmt_idx)
            if pmt_id >= N_PMTS:
                continue
            place = pmt["placement"]
            pos[pmt_id] = np.asarray(place["location"], dtype=np.float64)
            normal[pmt_id] = np.asarray(place["direction_z"], dtype=np.float64)
 
    # normalise the normals (they should already be unit, but be safe)
    nrm = np.linalg.norm(normal, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    normal = normal / nrm
    return pos, normal
 
 
def build_category_map(mpmt_info_path):
    """
    Map each pmt_id -> category string using other_mpmt_info.dict.
    Same logic as run_qe.py, extended to optionally pick up delaminated groups.
    PMTs without info -> 'Other'.
    """
    with open(mpmt_info_path, "rb") as f:
        mpmt_info = pickle.load(f)
 
    cat = np.full(N_PMTS, "Other", dtype=object)
    for pmt_id in range(N_PMTS):
        slot = pmt_id // 19
        info = mpmt_info.get(slot)
        if info is None:
            continue
        mtype = info.get("mpmt_type", "")
        msite = info.get("mpmt_site", "")
        delam = info.get("delaminated", False) or info.get("is_delaminated", False)
 
        base = None
        if   msite == "TRI" and mtype == "In-situ": base = "TRI In-situ"
        elif msite == "TRI" and mtype == "Ex-situ": base = "TRI Ex-situ"
        elif msite == "WUT" and mtype == "In-situ": base = "WUT In-situ"
        elif msite == "WUT" and mtype == "Ex-situ": base = "WUT Ex-situ"
 
        if delam:
            # Two delaminated groups split by site, if the flag exists.
            cat[pmt_id] = "Delaminated A" if msite == "TRI" else "Delaminated B"
        elif base is not None:
            cat[pmt_id] = base
    return cat
 
 
def load_vertices(csv_path, args, frame_offset, label):
    """
    Read a multilaterator CSV, apply the vertex-quality cuts and put the
    vertices in the WCTE mm frame used by the geometry JSON.

    Steps, in order:
      1. keep fits flagged successful, and chi2/ndf below --chi2ndf-cut if given;
      2. drop fits that railed at the +-bound of the multilaterator, which are
         reported as successful but are far outside the tank;
      3. convert WCSim cm -> WCTE mm unless --vertex-frame wcte-mm was given.

    The same function is used for data and MC so that both sides receive an
    identical treatment, which is what makes the reconstruction bias cancel in
    the ratio.
    """
    df = pd.read_csv(csv_path)
    n0 = len(df)

    if "fit_success" in df.columns:
        df = df[df["fit_success"] == True]                        # noqa: E712
    if args.chi2ndf_cut is not None and "chi2_ndof" in df.columns:
        df = df[df["chi2_ndof"] < args.chi2ndf_cut]
    n1 = len(df)

    xyz = df[["vertex_x", "vertex_y", "vertex_z"]].values.astype(np.float64)

    if not args.keep_railed_vertices and args.vertex_bound_cm > 0:
        # The fit bounds are expressed in the frame the multilaterator works in
        # (cm). In wcte-mm mode the CSV is assumed to be in mm already, so the
        # bound has to be scaled to compare against it.
        bound = args.vertex_bound_cm * (10.0 if args.vertex_frame == "wcte-mm"
                                        else 1.0)
        railed = (np.abs(xyz) >= bound - 1e-6).any(axis=1)
        df = df[~railed]
        xyz = xyz[~railed]
    n2 = len(df)

    if args.vertex_frame == "wcsim-cm":
        xyz = xyz * 10.0 + frame_offset

    out = df[["candidate_id"]].copy()
    out["vertex_x"] = xyz[:, 0]
    out["vertex_y"] = xyz[:, 1]
    out["vertex_z"] = xyz[:, 2]

    print(f"  {label}: {n0} rows -> {n1} after fit-quality cuts -> {n2} after "
          f"dropping railed fits; frame = {args.vertex_frame}"
          + (" (converted to WCTE mm)" if args.vertex_frame == "wcsim-cm" else ""))
    return out


def compute_cos_theta_and_L(hit_pmt_ids, vertices, pmt_pos, pmt_normal):
    """
    For a set of hits, given the *reconstructed vertex* of their candidate,
    compute cos(theta_PMT) and the vertex->PMT distance L for each hit.
 
    Parameters
    ----------
    hit_pmt_ids : (M,) int array of pmt_id for each hit
    vertices    : (M, 3) array, the reconstructed vertex of each hit's candidate
                  (already broadcast per-hit)
    pmt_pos     : (N_PMTS, 3) PMT positions
    pmt_normal  : (N_PMTS, 3) PMT outward normals
 
    Returns
    -------
    cos_theta : (M,) incidence angle cosine (1 = light hits photocathode head-on)
    L         : (M,) vertex->PMT distance [same units as geometry, mm]
    """
    p = pmt_pos[hit_pmt_ids]        # (M,3)
    n = pmt_normal[hit_pmt_ids]     # (M,3)
 
    vec = p - vertices              # vertex -> PMT
    L = np.linalg.norm(vec, axis=1)
    Lsafe = np.where(L > 0, L, 1.0)
    vhat = vec / Lsafe[:, None]
 
    # Light travels along vhat toward the PMT. The outward normal points away
    # from the photocathode, so the incidence cosine is the projection of the
    # incoming direction (-vhat) onto the normal:  cos = (-vhat) . n
    cos_theta = np.sum(-vhat * n, axis=1)
    return cos_theta, L
 
 
# ════════════════════════════════════════════════════════════════════
# Data preparation
# ════════════════════════════════════════════════════════════════════
 
def prepare_data_hits(df_sig, df_bkg, vertex_df, pmt_pos, pmt_normal,
                      args):
    """
    Apply cuts, attach reconstructed vertices to signal hits, compute
    per-hit cos(theta) and L, and background-subtract at the histogram level.
 
    Returns a per-hit DataFrame for SIGNAL with columns
        pmt_id, cos_theta, L, weight_proxy
    plus the matched-event background hits (same treatment) so the caller can
    subtract bin by bin. We also return the common-event bookkeeping.
    """
    # tRMS / nhits cuts (same as run_qe.py)
    df_s = df_sig[(df_sig["trms"] < args.trms_cut) &
                  (df_sig["nhits"] <= args.max_nhits) &
                  (df_sig["nhits"] >= args.min_hits)].copy()
    df_b = df_bkg[(df_bkg["trms"] < args.trms_cut) &
                  (df_bkg["nhits"] <= args.max_nhits) &
                  (df_bkg["nhits"] >= args.min_hits)].copy()
 
    # Common events for a statistically valid subtraction (as in run_qe.py)
    common = np.intersect1d(df_s["event_id"].unique(), df_b["event_id"].unique())
    df_s = df_s[df_s["event_id"].isin(common)]
    df_b = df_b[df_b["event_id"].isin(common)]
    n_events_common = len(common)
 
    # ---- attach the per-hit emission point ----
    # Two regimes:
    #   (a) vertex-based (default): each candidate's reconstructed vertex.
    #   (b) source-based (--use-source-pos): the fixed Ni ball position (mm) for
    #       every hit. This is the point-source regime, useful as a cross-check.
    if args.use_source_pos:
        src = np.asarray(args.source_pos_mm, dtype=np.float64)
        n_reco = df_s["candidate_id"].nunique()
        sig_cand = df_s["candidate_id"].values.astype(int)
        sig_pmt = df_s["pmt_id"].values.astype(int)
        sig_vtx = np.broadcast_to(src, (len(sig_pmt), 3))
        cos_s, L_s = compute_cos_theta_and_L(sig_pmt, sig_vtx, pmt_pos, pmt_normal)
    else:
        # vertex_df has already been quality-cut and put in the WCTE mm frame
        # by load_vertices().
        vtx = vertex_df[["candidate_id", "vertex_x", "vertex_y", "vertex_z"]]
        df_s = df_s.merge(vtx, on="candidate_id", how="inner")
        n_reco = df_s["candidate_id"].nunique()
 
        # ---- per-hit cos(theta) and L for SIGNAL (vertex-based) ----
        sig_cand = df_s["candidate_id"].values.astype(int)
        sig_pmt = df_s["pmt_id"].values.astype(int)
        sig_vtx = df_s[["vertex_x", "vertex_y", "vertex_z"]].values.astype(np.float64)
        cos_s, L_s = compute_cos_theta_and_L(sig_pmt, sig_vtx, pmt_pos, pmt_normal)

    df_sig_hits = pd.DataFrame({
        "candidate_id": sig_cand,
        "pmt_id":    sig_pmt,
        "cos_theta": cos_s,
        "L":         L_s,
    })
    df_sig_hits["weight_proxy"] = (
        np.exp(-L_s / args.atten_length_mm) / np.where(L_s > 0, L_s**2, 1.0)
    )
 
    # ---- BACKGROUND hits: no vertex (no Ni source). We still need to subtract
    # them from the signal histogram. The background is isotropic noise, so we
    # cannot define a meaningful per-hit cos(theta) from a vertex. Instead we
    # subtract the background as a per-PMT hit count, distributed across the
    # SAME cos(theta) bins in proportion to where that PMT's signal hits fell.
    # In practice, after the tRMS+nhits cuts the background is tiny (see thesis
    # background_subtraction plot), so we subtract per-(pmt_id) totals.
    bkg_counts = (df_b.groupby("pmt_id").size()
                  .reindex(range(N_PMTS), fill_value=0).values.astype(float))
 
    return df_sig_hits, bkg_counts, n_events_common, n_reco


def _parse_list_cell(value):
    """
    Read back a Python list that was written to CSV with repr().

    literal_eval rather than eval: it parses literals only, so a malformed or
    tampered CSV cannot execute code.
    """
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return ast.literal_eval(value)


def prepare_data_hits_from_candidates(candidate_csv, vertex_df,
                                      pmt_pos, pmt_normal, args):
    """
    Low-memory data path. Uses the candidates_for_reco.csv produced for the
    multilaterator instead of the full hit-level df_sig/df_bkg parquet.

    The CSV already contains one row per selected signal candidate with
    hit_slot_ids and hit_channel_ids lists. Background subtraction is supplied
    as per-PMT counts from data_hits_R*.npz when --data-hits-npz is passed.
    """
    print(f"Loading data candidates CSV: {candidate_csv}")
    df = pd.read_csv(candidate_csv, converters={
        "hit_times_ns": _parse_list_cell,
        "hit_slot_ids": _parse_list_cell,
        "hit_channel_ids": _parse_list_cell,
    })

    if "trms" in df.columns:
        df = df[df["trms"] < args.trms_cut]
    if "nhits" in df.columns:
        df = df[(df["nhits"] >= args.min_hits) & (df["nhits"] <= args.max_nhits)]

    if args.use_source_pos:
        src = np.asarray(args.source_pos_mm, dtype=np.float64)
        vertex_by_cand = None
    else:
        # Already quality-cut and frame-converted by load_vertices().
        vertex_by_cand = {
            int(r.candidate_id): np.array([r.vertex_x, r.vertex_y, r.vertex_z],
                                          dtype=np.float64)
            for r in vertex_df.itertuples()
        }

    rows_cand, rows_pmt, rows_cos, rows_L = [], [], [], []
    n_reco = 0
    for row in df.itertuples():
        cand_id = int(row.candidate_id)
        if args.use_source_pos:
            vertex = src
        else:
            vertex = vertex_by_cand.get(cand_id)
            if vertex is None:
                continue
        n_reco += 1

        slots = list(row.hit_slot_ids)
        chans = list(row.hit_channel_ids)
        for slot, chan in zip(slots, chans):
            pmt_id = int(slot) * 19 + int(chan)
            if pmt_id < 0 or pmt_id >= N_PMTS:
                continue
            p = pmt_pos[pmt_id]
            n = pmt_normal[pmt_id]
            vec = p - vertex
            L = float(np.linalg.norm(vec))
            vhat = vec / (L if L > 0 else 1.0)
            cos_theta = float(np.dot(-vhat, n))
            rows_cand.append(cand_id)
            rows_pmt.append(pmt_id)
            rows_cos.append(cos_theta)
            rows_L.append(L)

    df_sig_hits = pd.DataFrame({
        "candidate_id": np.asarray(rows_cand, dtype=int),
        "pmt_id": np.asarray(rows_pmt, dtype=int),
        "cos_theta": np.asarray(rows_cos, dtype=float),
        "L": np.asarray(rows_L, dtype=float),
    })
    df_sig_hits["weight_proxy"] = (
        np.exp(-df_sig_hits["L"].values / args.atten_length_mm) /
        np.where(df_sig_hits["L"].values > 0, df_sig_hits["L"].values**2, 1.0)
    )

    # ---- background ----
    # CAVEAT. In this low-memory path the background is taken from the per-PMT
    # counts in data_hits_R*.npz, which were produced with the run_data_hits.py
    # selection (tRMS and max-nhits) but WITHOUT the min-hits >= 6 requirement
    # applied to the signal candidates here, and restricted to the SIG/BKG common
    # events rather than to the candidates in this CSV. The background is
    # therefore slightly over-subtracted relative to the signal selection. After
    # the tRMS cut the residual background is a per-mille-level effect and the
    # response is renormalised at cos(theta)=1, so the impact is confined to the
    # lowest-statistics bins — but it is the main reason the parquet path
    # (prepare_data_hits) remains the reference for small samples.
    bkg_counts = np.zeros(N_PMTS, dtype=float)
    n_events_common = -1
    if args.data_hits_npz:
        d = np.load(args.data_hits_npz)
        if "bkg_hits" in d:
            bkg_counts = np.asarray(d["bkg_hits"], dtype=float)
        if "n_events_common" in d:
            n_events_common = int(d["n_events_common"])
        print("  Background counts loaded from data_hits npz. Note: these "
              "counts are per PMT after the data_hits cuts, not vertex-specific.")
    else:
        print("  WARNING: no --data-hits-npz supplied; background subtraction "
              "will be skipped in candidate-CSV mode.")

    return df_sig_hits, bkg_counts, n_events_common, n_reco
 
 
# ════════════════════════════════════════════════════════════════════
# MC preparation
# ════════════════════════════════════════════════════════════════════
 
def prepare_mc_hits(args, pmt_pos, pmt_normal, tube_mapping, frame_offset):
    """
    Process MC through the same trigger + ToF + cuts pipeline, then compute
    per-hit cos(theta) and L from the per-candidate RECONSTRUCTED vertex,
    EXACTLY as for data. Returns a per-hit DataFrame with pmt_id, cos_theta, L,
    weight_proxy.
 
    Why reconstructed vertices (not the source position)
    -----------------------------------------------------
    The angular-response method relies on the geometric weight I_s/L^2 cancelling
    in the data/MC ratio within each cos(theta) bin. For that cancellation to be
    real, BOTH sides must build cos(theta) and L from the SAME kind of vertex
    estimator. In data we use the multilaterator vertex per candidate; therefore
    in MC we must ALSO use the multilaterator vertex per candidate (produced by
    build_mc_clusters_for_reco.py + multilat_vertex_reconstruction.py), so that
    the reconstruction bias is identical on both sides and cancels in the ratio.
    Using the fixed source position here (the previous behaviour) breaks the
    symmetry and makes the weight-proxy self-check fail.
 
    Units
    -----
    The multilaterator reconstructs vertices using the WCTE geometry from
    functions_bonsai, i.e. in the SAME units as the PMT positions in the
    geometry JSON (mm). So MC vertices (mm) + JSON PMT positions (mm) give L in
    mm, consistently with data. There is therefore NO cm/mm mismatch any more:
    both data and MC live in mm. We keep args.source_pos_cm only to reproduce
    the candidate selection (the ToF correction used for triggering).
    """
    from WCSimFilePackages.npz_to_df import truehits_info_to_df
 
    # ─── per-hit emission point: reconstructed vertex (default) or source ───
    use_source = args.use_source_pos
    if use_source:
        # Source-based regime: every hit uses the fixed Ni ball position (mm).
        # No reconstructed vertex is needed, so ALL surviving candidates enter
        # (none are dropped for lacking a reco vertex). This mirrors the data
        # source-based branch and is the point-source cross-check.
        src = np.asarray(args.source_pos_mm, dtype=np.float64)
        print(f"  Using FIXED source position {src.tolist()} mm (WCTE) for all "
              f"MC hits (point-source cross-check)")
        vtx_by_cand = None
    else:
        # ─── load the MC reconstructed vertices (same multilaterator as data) ──
        print(f"  Loading MC reconstructed vertices: {args.mc_vertex_csv}")
        mc_vtx = load_vertices(args.mc_vertex_csv, args, frame_offset, "MC")
        vtx_by_cand = {int(r.candidate_id): (r.vertex_x, r.vertex_y, r.vertex_z)
                       for r in mc_vtx.itertuples()}
        print(f"    {len(vtx_by_cand)} MC candidates with a reconstructed vertex")
 
    # We need per-hit (pmt, position) for each surviving candidate. run_mc_trigger
    # stored all_pmts as UNIQUE tube numbers (loses multiplicity), so we re-extract
    # per-hit rows by matching the stored all_times back to df_mc_hits within each
    # event (same trick as the reflection study), then attach the emission point
    # (reconstructed vertex, or the fixed source position).
    # Resolved with the SAME helper and therefore the SAME ordering as
    # build_mc_clusters_for_reco.py, so the per-chunk candidate-ID offsets
    # below reproduce exactly the IDs stored in the MC vertex CSV. Cross-check
    # against <mc_candidates_for_reco.csv>.filelist.txt if in doubt.
    mc_files = resolve_npz_files(args.mc_npz)
    print(f"  Processing MC from {len(mc_files)} file(s)")

    rows_cand, rows_pmt, rows_cos, rows_L = [], [], [], []
    n_no_vertex = 0
    n_cands_mc = 0

    for file_idx, mc_file in enumerate(mc_files):
        cand_offset = file_idx * CANDIDATES_PER_FILE_MAX
        print(f"  [{file_idx + 1}/{len(mc_files)}] {os.path.basename(mc_file)}")

        print("    Reading MC true hits...")
        df_mc_hits = read_mc_truehits(mc_file, truehits_info_to_df)
        print(f"      {len(df_mc_hits)} hits, {df_mc_hits['event_id'].nunique()} events")

        print(f"    ToF correction to source {args.source_pos_cm} cm (selection only)...")
        df_mc_hits = apply_tof_correction_mc(
            df_mc_hits, args.source_pos_cm, args.n_water)

        print(f"    Running trigger (w={args.window}, thresh_min={args.thresh_min})...")
        df_mc_cands = run_mc_trigger(
            df_mc_hits, w=args.window, thresh_min=args.thresh_min,
            time_col="true_hit_time_tof_corrected")
        print(f"      {len(df_mc_cands)} MC candidates")

        df_mc_cands = df_mc_cands[(df_mc_cands["trms"] < args.trms_cut) &
                                  (df_mc_cands["nhits"] <= args.max_nhits) &
                                  (df_mc_cands["nhits"] >= args.min_hits)]
        print(f"      After cuts: {len(df_mc_cands)} MC candidates")

        grouped = {ev: d for ev, d in df_mc_hits.groupby("event_id")}

        cands_with_vertex = set()
        for _, cand in df_mc_cands.iterrows():
            cand_id = int(cand["candidate_id"]) + cand_offset
            if use_source:
                vertex = src
            else:
                vtx = vtx_by_cand.get(cand_id)
                if vtx is None:
                    n_no_vertex += 1
                    continue
                vertex = np.array(vtx, dtype=np.float64)
                cands_with_vertex.add(cand_id)

            ev = cand["event_id"]
            ev_hits = grouped.get(ev)
            if ev_hits is None:
                continue

            # run_mc_trigger stores only the candidate's sorted times and its
            # UNIQUE tube numbers, so the (time, tube) pairing has to be
            # recovered by looking each stored time back up in the event table.
            times_all = ev_hits["true_hit_time_tof_corrected"].values.astype(np.float64)
            tubes_all = ev_hits["true_hit_pmt"].values
            cand_times = np.asarray(cand["all_times"], dtype=np.float64)
            idx = match_times_to_hits(times_all, cand_times, atol=1e-3)

            # WCSim tube numbers are 0-based in the .npz, 1-based in the geofile.
            pmt_ids = []
            for j in idx:
                if j < 0:
                    continue
                mp = tube_mapping.get(int(tubes_all[j]) + 1)
                if mp is None:
                    continue
                pid = mp[0] * 19 + mp[1]
                if 0 <= pid < N_PMTS:
                    pmt_ids.append(pid)
            if not pmt_ids:
                continue

            # Geometry for all hits of this candidate at once.
            pmt_ids = np.asarray(pmt_ids, dtype=int)
            cos_arr, L_arr = compute_cos_theta_and_L(
                pmt_ids,
                np.broadcast_to(vertex, (pmt_ids.size, 3)),
                pmt_pos, pmt_normal)

            rows_cand.extend([cand_id] * pmt_ids.size)
            rows_pmt.extend(pmt_ids.tolist())
            rows_cos.extend(cos_arr.tolist())
            rows_L.extend(L_arr.tolist())

        if use_source:
            n_cands_mc += len(df_mc_cands)
        else:
            n_cands_mc += len(cands_with_vertex)

        del df_mc_hits, df_mc_cands, grouped

    if (not use_source) and n_no_vertex:
        print(f"    ({n_no_vertex} surviving MC candidates had no reconstructed "
              f"vertex and were skipped)")
 
    df_mc = pd.DataFrame({
        "candidate_id": np.asarray(rows_cand, dtype=int),
        "pmt_id":    np.asarray(rows_pmt, dtype=int),
        "cos_theta": np.asarray(rows_cos, dtype=float),
        "L":         np.asarray(rows_L, dtype=float),
    })
    # MC L is now in mm (multilaterator + JSON), same as data. Use the mm
    # attenuation length so the weight-proxy is directly comparable to data.
    df_mc["weight_proxy"] = (
        np.exp(-df_mc["L"].values / args.atten_length_mm) / np.where(df_mc["L"].values > 0, df_mc["L"].values**2, 1.0)
    )
    return df_mc, n_cands_mc
 
 
# ════════════════════════════════════════════════════════════════════
# Histogramming & angular response
# ════════════════════════════════════════════════════════════════════
 
def fill_hist_by_category(df_hits, category, cos_edges, cos_min,
                          use_weights=False):
    """
    For each category, build a 1D histogram of NHits vs cos(theta).
 
    df_hits has columns pmt_id, cos_theta and (if use_weights) weight_proxy.
 
    Returns two dicts cat -> array (n_bins,):
      hist  : the (weighted or unweighted) bin contents.
      sumw2 : the sum of squared weights per bin, used for the bin error.
              - unweighted: each entry contributes 1, so sumw2 == counts and
                the Poisson error sqrt(counts) is recovered downstream.
              - weighted:   each entry contributes w_i^2, so the bin error is
                sqrt(sumw2) = sqrt(sum w_i^2).
    """
    # `category` is a length-N_PMTS object array, so it can be fancy-indexed
    # directly; the previous per-hit list comprehension was the bottleneck on
    # samples with tens of millions of hits.
    cats = category[df_hits["pmt_id"].values.astype(int)]
    cosv = df_hits["cos_theta"].values
    keep = cosv > cos_min
    if use_weights:
        wv = df_hits["weight_proxy"].values
    else:
        wv = np.ones_like(cosv)
 
    hist, sumw2 = {}, {}
    for c in CATEGORY_COLORS:
        m = keep & (cats == c)
        if m.sum() == 0:
            hist[c]  = np.zeros(len(cos_edges) - 1)
            sumw2[c] = np.zeros(len(cos_edges) - 1)
            continue
        h,  _ = np.histogram(cosv[m], bins=cos_edges, weights=wv[m])
        h2, _ = np.histogram(cosv[m], bins=cos_edges, weights=wv[m] ** 2)
        hist[c]  = h.astype(float)
        sumw2[c] = h2.astype(float)
    return hist, sumw2
 
 
def subtract_background_hist(sig_hist_by_cat, sig_sumw2_by_cat,
                             df_sig_hits, bkg_counts,
                             category, cos_edges, cos_min,
                             use_weights=False):
    """
    Subtract the (tiny) background from each category's NHits-vs-cos(theta)
    histogram. The background is known per pmt_id (bkg_counts). For each PMT we
    distribute its background hits across cos(theta) bins in proportion to that
    PMT's *signal* hit distribution in cos(theta) (the only sensible shape we
    have, since background has no vertex). After the tRMS+nhits cuts this is a
    small correction.
 
    The background carries no reconstructed vertex and (after the cuts) is
    negligible, so we subtract it only from the bin *contents* and leave the
    signal sum-of-squared-weights (sig_sumw2_by_cat) untouched: the bin error
    stays dominated by the signal statistics. Returns (pure_hist, pure_sumw2).
    """
    cosv = df_sig_hits["cos_theta"].values
    pmtv = df_sig_hits["pmt_id"].values.astype(int)
    keep = cosv > cos_min
    if use_weights:
        wv = df_sig_hits["weight_proxy"].values.astype(float)
    else:
        wv = np.ones_like(cosv, dtype=float)    
 
    # signal shape per PMT in cos bins
    pure   = {c: sig_hist_by_cat[c].copy()  for c in sig_hist_by_cat}
    sumw2  = {c: sig_sumw2_by_cat[c].copy() for c in sig_sumw2_by_cat}
 
    # for each PMT with background, find its signal-bin distribution
    bin_idx = np.digitize(cosv, cos_edges) - 1
    nbins = len(cos_edges) - 1
 
    for pmt_id in np.where(bkg_counts > 0)[0]:
        c = category[pmt_id]
        if c not in pure:
            continue
        sel = keep & (pmtv == pmt_id)
        if sel.sum() == 0:
            continue

        # Shape template: where this PMT's SIGNAL hits fell in cos(theta).
        # Normalised to unit area, then scaled by the PMT's background count.
        bins_all = bin_idx[sel]
        valid = (bins_all >= 0) & (bins_all < nbins)
        bins_here = bins_all[valid]
        if len(bins_here) == 0:
            continue

        weights_here = wv[sel][valid]
        weighted_template = np.bincount(
            bins_here, weights=weights_here, minlength=nbins).astype(float)
        weighted_template /= float(len(bins_here))
        pure[c] -= bkg_counts[pmt_id] * weighted_template
 
    for c in pure:
        pure[c][pure[c] < 0] = 0.0
    return pure, sumw2
 
 
def angular_response(data_hist, mc_hist, data_sumw2, mc_sumw2):
    """
    Angular response per type = (data / MC), normalised so the highest-cos bin
    that has MC and data statistics equals 1.
 
    The error is propagated from the per-bin sum of squared weights on each
    side. The relative error of a weighted bin is sqrt(sum w^2) / (sum w); the
    two sides are independent, so
 
        sigma_ratio / ratio = sqrt( dataS2/dataN^2 + mcS2/mcN^2 ).
 
    In the unweighted case sum w^2 == N == sum w, and this collapses to the
    Poisson expression ratio*sqrt(1/dataN + 1/mcN) used before.
    Returns ratio array and its error.
    """
    ratio = np.full_like(data_hist, np.nan, dtype=float)
    err   = np.full_like(data_hist, np.nan, dtype=float)
 
    m = (data_hist > 0) & (mc_hist > 0)
    ratio[m] = data_hist[m] / mc_hist[m]
    rel2 = (data_sumw2[m] / data_hist[m] ** 2 +
            mc_sumw2[m]   / mc_hist[m]   ** 2)
    err[m] = ratio[m] * np.sqrt(rel2)
 
    # normalise to the last valid (highest cos theta) bin
    valid_bins = np.where(m)[0]
    if len(valid_bins) > 0:
        norm = ratio[valid_bins[-1]]
        if norm > 0:
            ratio = ratio / norm
            err = err / norm
    return ratio, err


def cut_flow_stats(label, before_l, after_l, after_cos, min_l_mm, cos_min):
    """
    Summarise the hit-level cuts and the candidate-level losses.

    Candidate rejection means that a candidate had at least one hit before the
    cut and no hits left after that cut. Candidate affected means that at least
    one hit was removed, even if the candidate still has surviving hits.
    """
    before_l_cands = set(before_l["candidate_id"].astype(int).unique())
    after_l_cands = set(after_l["candidate_id"].astype(int).unique())
    after_cos_cands = set(after_cos["candidate_id"].astype(int).unique())

    if len(before_l) > 0:
        l_rejected_hits_by_cand = before_l.loc[
            before_l["L"] < min_l_mm, "candidate_id"].astype(int).unique()
    else:
        l_rejected_hits_by_cand = []
    if len(after_l) > 0:
        cos_rejected_hits_by_cand = after_l.loc[
            after_l["cos_theta"] <= cos_min, "candidate_id"].astype(int).unique()
    else:
        cos_rejected_hits_by_cand = []

    l_hits_rejected = int(len(before_l) - len(after_l))
    cos_hits_rejected = int(len(after_l) - len(after_cos))

    return {
        "sample": label,
        "min_L_mm": float(min_l_mm),
        "cos_min": float(cos_min),
        "hits_before_L": int(len(before_l)),
        "hits_after_L": int(len(after_l)),
        "hits_rejected_by_L_lt_min": l_hits_rejected,
        "hits_before_cos": int(len(after_l)),
        "hits_after_cos": int(len(after_cos)),
        "hits_rejected_by_cos_le_min": cos_hits_rejected,
        "candidates_before_L": int(len(before_l_cands)),
        "candidates_after_L": int(len(after_l_cands)),
        "candidates_rejected_by_L_lt_min": int(len(before_l_cands - after_l_cands)),
        "candidates_affected_by_L_lt_min": int(len(l_rejected_hits_by_cand)),
        "candidates_before_cos": int(len(after_l_cands)),
        "candidates_after_cos": int(len(after_cos_cands)),
        "candidates_rejected_by_cos_le_min": int(len(after_l_cands - after_cos_cands)),
        "candidates_affected_by_cos_le_min": int(len(cos_rejected_hits_by_cand)),
    }


# ════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════
 
def plot_angular_response(cos_centers, resp_data, resp_err, fig_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    for c, color in CATEGORY_COLORS.items():
        if c not in resp_data:
            continue
        r = resp_data[c]; e = resp_err[c]
        m = np.isfinite(r)
        if m.sum() == 0:
            continue
        ax.errorbar(cos_centers[m], r[m], yerr=e[m], fmt="o-", color=color,
                    markersize=3, elinewidth=0.6, capsize=0, label=c)
    ax.axhline(1.0, ls=":", color="k")
    ax.set_xlabel(r"$\cos\theta_{\mathrm{PMT}}$ (from reconstructed vertex)")
    ax.set_ylabel(r"Angular response  $\epsilon(\theta)$  (Data/MC, norm. at $\cos\theta=1$)")
    ax.set_title("PMT angular response per mPMT type (vertex-based method)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "angular_response_vertex.png"), dpi=150)
    plt.close()
 
 
def plot_nhits_histograms(cos_centers, data_by_cat, mc_by_cat, fig_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    cats4 = ["TRI In-situ", "TRI Ex-situ", "WUT In-situ", "WUT Ex-situ"]
    for ax, c in zip(axes.ravel(), cats4):
        d = data_by_cat.get(c); m = mc_by_cat.get(c)
        if d is None or m is None:
            ax.set_visible(False); continue
        dd = d / (d.sum() if d.sum() > 0 else 1)
        mm = m / (m.sum() if m.sum() > 0 else 1)
        ax.step(cos_centers, dd, where="mid", color="blue", label="Data (norm.)")
        ax.step(cos_centers, mm, where="mid", color="black", label="MC (norm.)")
        ax.set_title(c); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_xlabel(r"$\cos\theta_{\mathrm{PMT}}$")
        ax.set_ylabel("NHits (area-normalised)")
    plt.suptitle("NHits vs cos(theta) per mPMT type — Data vs MC", y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "nhits_vs_costheta.png"), dpi=150)
    plt.close()
 
 
def plot_nhits_histograms_raw(cos_centers, data_by_cat, mc_by_cat,
                              data_sumw2_by_cat, mc_sumw2_by_cat, fig_dir,
                              use_weights=False):
    """
    Pre-normalisation NHits-vs-cos(theta), i.e. the raw bin contents (counts, or
    summed weights if use_weights) with their per-bin error sqrt(sum w^2). This
    exposes the actual statistics behind each bin, so low-cos bins with few hits
    (which drive the noise in the ratio) are visible. Data and MC are shown with
    independent y-axes per panel because their absolute normalisations differ.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    cats4 = ["TRI In-situ", "TRI Ex-situ", "WUT In-situ", "WUT Ex-situ"]
    ylab = ("Summed weights" if use_weights else "NHits (raw counts)")
    for ax, c in zip(axes.ravel(), cats4):
        d = data_by_cat.get(c); m = mc_by_cat.get(c)
        if d is None or m is None:
            ax.set_visible(False); continue
        de = np.sqrt(data_sumw2_by_cat[c])
        me = np.sqrt(mc_sumw2_by_cat[c])
        ax.errorbar(cos_centers, d, yerr=de, fmt="o-", color="blue",
                    markersize=3, elinewidth=0.6, capsize=0, label="Data")
        # MC scaled to data area so both shapes are comparable on one axis,
        # while the annotation keeps the true MC integral.
        scale = (d.sum() / m.sum()) if m.sum() > 0 else 1.0
        ax.errorbar(cos_centers, m * scale, yerr=me * scale, fmt="s--",
                    color="black", markersize=3, elinewidth=0.6, capsize=0,
                    label=f"MC (×{scale:.2g} to data area)")
        ax.set_title(f"{c}   (N_data={d.sum():.0f}, N_MC={m.sum():.0f})")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_xlabel(r"$\cos\theta_{\mathrm{PMT}}$")
        ax.set_ylabel(ylab)
    plt.suptitle("NHits vs cos(theta) per mPMT type — raw statistics "
                 "(pre-normalisation)", y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "nhits_vs_costheta_raw.png"), dpi=150)
    plt.close()
 
 
def plot_weight_check(df_sig_hits, df_mc_hits, cos_edges, cos_min, fig_dir,
                      n_show=6):
    """
    Self-check requested in the slides: per cos(theta) bin, compare the
    distribution of the weight proxy  exp(-L/lambda)/L^2  between data and MC.
    If these distributions agree bin by bin, ignoring the per-hit weight (i.e.
    just counting hits) is justified. We overlay a handful of representative
    bins.

    Both L values are in mm (data and MC vertices come from the same
    multilaterator and the same JSON PMT positions), so the weights are directly
    comparable; the histograms are still area-normalised because only the SHAPE
    within a cos(theta) bin matters for the cancellation argument.
    """
    nbins = len(cos_edges) - 1
    # choose bins with decent stats, spread across the cos range
    show_bins = np.linspace(nbins // 4, nbins - 1, n_show).astype(int)
    show_bins = sorted(set(show_bins))
 
    d_cos = df_sig_hits["cos_theta"].values
    d_w   = df_sig_hits["weight_proxy"].values
    m_cos = df_mc_hits["cos_theta"].values
    m_w   = df_mc_hits["weight_proxy"].values
 
    d_bin = np.digitize(d_cos, cos_edges) - 1
    m_bin = np.digitize(m_cos, cos_edges) - 1
 
    ncols = 3
    nrows = int(np.ceil(len(show_bins) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
 
    for ax, b in zip(axes, show_bins):
        dvals = d_w[(d_bin == b) & (d_cos > cos_min)]
        mvals = m_w[(m_bin == b) & (m_cos > cos_min)]
        if len(dvals) == 0 and len(mvals) == 0:
            ax.set_visible(False); continue
        # log-spaced bins on the weight proxy (it spans orders of magnitude)
        allv = np.concatenate([v for v in (dvals, mvals) if len(v) > 0])
        allv = allv[allv > 0]
        if len(allv) == 0:
            ax.set_visible(False); continue
        lo, hi = np.percentile(allv, [1, 99])
        if lo <= 0:
            lo = allv.min()
        wbins = np.logspace(np.log10(lo), np.log10(hi), 30)
        if len(dvals):
            ax.hist(dvals, bins=wbins, density=True, histtype="step",
                    color="blue", label="Data")
        if len(mvals):
            ax.hist(mvals, bins=wbins, density=True, histtype="step",
                    color="black", label="MC")
        ax.set_xscale("log")
        c0, c1 = cos_edges[b], cos_edges[b + 1]
        ax.set_title(f"cos θ ∈ [{c0:.2f}, {c1:.2f}]")
        ax.set_xlabel(r"$e^{-L/\lambda}/L^2$ (a.u., shape only)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
 
    for ax in axes[len(show_bins):]:
        ax.set_visible(False)
    plt.suptitle("Self-check: weight-proxy distribution per cos(theta) bin "
                 "(Data vs MC, shape)", y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "weight_proxy_check.png"), dpi=150)
    plt.close()
 
 
# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
 
def main():
    args = parse_args()
 
    # Both data and MC vertices come from the same multilaterator (mm) and the
    # same JSON PMT positions (mm), so L is in mm on both sides and we use
    # args.atten_length_mm directly. No cm/mm conversion needed any more.
 
    if args.use_source_pos and args.source_pos_mm is None:
        sys.exit("ERROR: --use-source-pos requires --source-pos-mm "
                 "(the Ni ball position in WCTE/mm, same frame as the geometry "
                 "JSON). This is NOT --source-pos-cm, which is the WCSim frame "
                 "used only for the trigger ToF.")
    if not args.use_source_pos:
        missing = [n for n, v in (("--vertex-csv", args.vertex_csv),
                                  ("--mc-vertex-csv", args.mc_vertex_csv))
                   if v is None]
        if missing:
            sys.exit("ERROR: vertex-based mode (the default) requires "
                     + " and ".join(missing) + ". Pass them, or use "
                     "--use-source-pos for the point-source cross-check.")
    if args.data_candidates_csv is None:
        missing = [n for n, v in (("--sig-parquet", args.sig_parquet),
                                  ("--bkg-parquet", args.bkg_parquet))
                   if v is None]
        if missing:
            sys.exit("ERROR: parquet data mode requires "
                     + " and ".join(missing) + ". Pass --data-candidates-csv "
                     "to use the low-memory data path.")
 
    fig_dir  = os.path.join(args.output_dir, "figures")
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
 
    # Binning in cos(theta). Only the forward hemisphere is used: cos < 0 means
    # the light arrived on the back of the photocathode, which is either a
    # reflection or a vertex-reconstruction failure, not a measurement of the
    # angular response.
    cos_edges   = np.linspace(args.cos_min, 1.0, args.n_cos_bins + 1)
    cos_centers = 0.5 * (cos_edges[:-1] + cos_edges[1:])
 
    # ---- geometry & categories ----
    print("Loading geometry, normals and mPMT categories...")
    pmt_pos, pmt_normal = load_pmt_geometry(args.geo_json)
    category = build_category_map(args.mpmt_info)
 
    # Transform between the multilaterator frame (WCSim cm) and the frame of the
    # PMT positions used here (WCTE mm). Fitted from the channels the two
    # geometry files have in common; see pipeline_utils for the details.
    frame_offset = np.zeros(3)
    if not args.use_source_pos and args.vertex_frame == "wcsim-cm":
        frame_offset, _ = wcsim_cm_to_wcte_mm_offset(args.geo_file, args.geo_json)

    if args.use_source_pos:
        print(f"Using FIXED source position {args.source_pos_mm} mm (WCTE) for "
              f"all data hits (point-source cross-check; no vertex CSV needed)")
        vertex_df = None
    else:
        print("Loading multilaterator vertices...")
        vertex_df = load_vertices(args.vertex_csv, args, frame_offset, "data")
 
    print(f"Preparing data hits (tRMS<{args.trms_cut}, "
          f"{args.min_hits}<=nhits<={args.max_nhits})...")
    if args.data_candidates_csv is not None:
        df_sig_hits, bkg_counts, n_events_common, n_reco = (
            prepare_data_hits_from_candidates(
                args.data_candidates_csv, vertex_df, pmt_pos, pmt_normal, args)
        )
    else:
        print("Loading data candidate parquets...")
        df_sig = pd.read_parquet(args.sig_parquet)
        df_bkg = pd.read_parquet(args.bkg_parquet)
        for df in (df_sig, df_bkg):
            if "trms" not in df.columns:
                df["trms"] = df.groupby("candidate_id")[
                    "hit_pmt_calibrated_times"].transform("std")
            if "nhits" not in df.columns:
                df["nhits"] = df.groupby("candidate_id")[
                    "hit_pmt_calibrated_times"].transform("count")
        df_sig_hits, bkg_counts, n_events_common, n_reco = prepare_data_hits(
            df_sig, df_bkg, vertex_df, pmt_pos, pmt_normal, args)
    print(f"  Common events: {n_events_common}, "
          f"signal candidates used: {n_reco}")
 
    # ---- MC ----
    print("\nProcessing MC...")
    tube_mapping = load_wcsim_tube_mapping(args.geo_file)
    df_mc_hits, n_cands_mc = prepare_mc_hits(args, pmt_pos, pmt_normal,
                                             tube_mapping, frame_offset)
 
    # ---- min-L cut (BOTH sides, identically) ----
    # At small vertex->PMT distance the 1/L^2 point-source weight diverges and
    # the vertex-reconstruction error dominates cos(theta); discarding short-L
    # hits is the collaboration's suggested fix for the run-to-run discrepancy.
    df_sig_hits_before_L = df_sig_hits
    df_mc_hits_before_L = df_mc_hits
    effective_min_L_mm = float(args.min_L_mm) if args.min_L_mm and args.min_L_mm > 0 else 0.0
    if args.min_L_mm and args.min_L_mm > 0:
        n_d0, n_m0 = len(df_sig_hits), len(df_mc_hits)
        df_sig_hits = df_sig_hits[df_sig_hits["L"] >= args.min_L_mm].copy()
        df_mc_hits  = df_mc_hits[df_mc_hits["L"]  >= args.min_L_mm].copy()
        print(f"\nApplying L >= {args.min_L_mm:.0f} mm cut:")
        print(f"  Data hits: {n_d0} -> {len(df_sig_hits)} "
              f"({100*len(df_sig_hits)/max(n_d0,1):.1f}% kept)")
        print(f"  MC   hits: {n_m0} -> {len(df_mc_hits)} "
              f"({100*len(df_mc_hits)/max(n_m0,1):.1f}% kept)")

    df_sig_hits_after_cos = df_sig_hits[
        df_sig_hits["cos_theta"] > args.cos_min].copy()
    df_mc_hits_after_cos = df_mc_hits[
        df_mc_hits["cos_theta"] > args.cos_min].copy()

    cut_flow = [
        cut_flow_stats("data", df_sig_hits_before_L, df_sig_hits,
                       df_sig_hits_after_cos, effective_min_L_mm, args.cos_min),
        cut_flow_stats("mc", df_mc_hits_before_L, df_mc_hits,
                       df_mc_hits_after_cos, effective_min_L_mm, args.cos_min),
    ]
    cut_flow_df = pd.DataFrame(cut_flow)

    print("\nCandidate accounting for L cut:")
    for row in cut_flow:
        print(f"  {row['sample'].upper():4s} L candidates with any hit: "
              f"{row['candidates_before_L']} -> {row['candidates_after_L']} "
              f"({row['candidates_rejected_by_L_lt_min']} fully rejected; "
              f"{row['candidates_affected_by_L_lt_min']} affected)")

    print(f"\nApplying cos(theta) > {args.cos_min:.3f} histogram cut:")
    for row in cut_flow:
        print(f"  {row['sample'].upper():4s} hits: "
              f"{row['hits_before_cos']} -> {row['hits_after_cos']} "
              f"({row['hits_rejected_by_cos_le_min']} rejected)")
        print(f"  {row['sample'].upper():4s} candidates with any hit: "
              f"{row['candidates_before_cos']} -> {row['candidates_after_cos']} "
              f"({row['candidates_rejected_by_cos_le_min']} fully rejected; "
              f"{row['candidates_affected_by_cos_le_min']} affected)")
 
    # ---- histograms ----
    mode = "weighted (exp(-L/lambda)/L^2)" if args.use_weights else "unweighted (counts)"
    print(f"\nFilling NHits vs cos(theta) histograms per mPMT type [{mode}]...")
    sig_hist, sig_sumw2 = fill_hist_by_category(
        df_sig_hits, category, cos_edges, args.cos_min, args.use_weights)
    pure_hist, pure_sumw2 = subtract_background_hist(
        sig_hist, sig_sumw2, df_sig_hits, bkg_counts,
        category, cos_edges, args.cos_min, args.use_weights)
    mc_hist, mc_sumw2 = fill_hist_by_category(
        df_mc_hits, category, cos_edges, args.cos_min, args.use_weights)
 
    # ---- angular response ----
    # One independent data/MC ratio per mPMT category, each renormalised so that
    # its highest-cos(theta) bin equals 1. The normalisation makes the result a
    # relative angular response: it cancels the per-category quantum-efficiency
    # offset (that is what run_qe.py measures) and leaves only the shape in
    # angle. It also means the last bin carries no information and that a noisy
    # last bin rescales the whole curve, which is worth checking on the raw
    # histogram figure before quoting numbers.
    print("Computing angular response (Data/MC, normalised at cos theta = 1)...")
    resp_data, resp_err = {}, {}
    for c in CATEGORY_COLORS:
        r, e = angular_response(pure_hist[c], mc_hist[c],
                                pure_sumw2[c], mc_sumw2[c])
        resp_data[c] = r
        resp_err[c]  = e
 
    # ---- plots ----
    print("Plotting...")
    plot_nhits_histograms(cos_centers, pure_hist, mc_hist, fig_dir)
    plot_nhits_histograms_raw(cos_centers, pure_hist, mc_hist,
                              pure_sumw2, mc_sumw2, fig_dir, args.use_weights)
    plot_angular_response(cos_centers, resp_data, resp_err, fig_dir)
    plot_weight_check(df_sig_hits, df_mc_hits, cos_edges, args.cos_min, fig_dir)
 
    # ---- save ----
    out = {"cos_center": cos_centers}
    for c in CATEGORY_COLORS:
        key = c.replace(" ", "_")
        out[f"data_{key}"]       = pure_hist[c]
        out[f"data_sumw2_{key}"] = pure_sumw2[c]
        out[f"mc_{key}"]         = mc_hist[c]
        out[f"mc_sumw2_{key}"]   = mc_sumw2[c]
        out[f"response_{key}"]   = resp_data[c]
        out[f"response_err_{key}"] = resp_err[c]
    pd.DataFrame(out).to_csv(
        os.path.join(data_dir, "angular_response_vertex.csv"), index=False)
    cut_flow_csv = os.path.join(data_dir, "angular_cut_flow_summary.csv")
    cut_flow_json = os.path.join(data_dir, "angular_cut_flow_summary.json")
    cut_flow_df.to_csv(cut_flow_csv, index=False)
    with open(cut_flow_json, "w") as f:
        json.dump(cut_flow, f, indent=2)
 
    print("\n--- Summary ---")
    emode = (f"FIXED source {args.source_pos_mm} mm (point-source)"
             if args.use_source_pos else "reconstructed vertex per candidate")
    print(f"  Emission point:                   {emode}")
    if not args.use_source_pos:
        print(f"  Vertex frame:                     {args.vertex_frame}"
              + (" -> converted to WCTE mm" if args.vertex_frame == "wcsim-cm"
                 else " -> used unconverted"))
    print(f"  Mode:                             {mode}")
    min_l_txt = (f"{args.min_L_mm:.0f} mm"
                 if args.min_L_mm and args.min_L_mm > 0 else "(disabled)")
    print(f"  min-L cut:                        {min_l_txt}")
    print(f"  Common events (SIG/BKG):          {n_events_common}")
    print(f"  Signal candidates w/ vertex:      {n_reco}")
    print(f"  MC candidates after cuts:         {n_cands_mc}")
    for c in ["TRI In-situ", "TRI Ex-situ", "WUT In-situ", "WUT Ex-situ"]:
        r = resp_data[c]; m = np.isfinite(r)
        if m.sum() > 0:
            print(f"  {c:14s}: <response> = {np.nanmean(r[m]):.3f} "
                  f"over {m.sum()} cos-bins")
    print(f"\nSaved figures to {fig_dir}")
    print(f"Saved table   to {os.path.join(data_dir, 'angular_response_vertex.csv')}")
    print(f"Saved cuts    to {cut_flow_csv}")
    print(f"Saved cuts    to {cut_flow_json}")
    print("Done!")
 
 
if __name__ == "__main__":
    main()
