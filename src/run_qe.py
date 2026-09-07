#!/usr/bin/env python3
"""
NiCf Analysis Pipeline — QE Studies
=====================================
Computes the relative quantum efficiency per PMT by comparing
data (background-subtracted signal) to Monte Carlo.

MC is processed through the same pipeline as data:
  1. Read true hits
  2. Apply ToF correction to source
  3. Run greedy nHits trigger
  4. Apply tRMS cut
  5. Compare hits per PMT: data vs MC

Usage
-----
    python run_qe.py \
        --sig-parquet ./output/data/df_sig_R1767.parquet \
        --bkg-parquet ./output/data/df_bkg_R1766.parquet \
        --mc-npz /path/to/wcsim_nicf.npz \
        --geo-file /path/to/geofile_NuPRISMBeamTest_16cShort_mPMT.txt \
        --source-pos-cm 0 80 0 \
        --trms-cut 2.0 \
        --window 20 \
        --thresh-min 2 \
        --output-dir ./output
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from functions import (
    load_wcsim_tube_mapping,
    read_mc_truehits,
    apply_tof_correction_mc,
    run_mc_trigger,
    mc_cands_to_pmt_id,
)

sys.path.append(os.path.abspath(
    os.environ.get("WCTE_SOFTWARE_DIR",
                    "/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk")
))

N_PMTS = 2014


def parse_args():
    p = argparse.ArgumentParser(description="NiCf QE studies")
    p.add_argument("--sig-parquet",   type=str,   default=None,
                   help="Signal parquet (monolithic mode). Omit if using "
                        "--data-hits-npz.")
    p.add_argument("--bkg-parquet",   type=str,   default=None,
                   help="Background parquet (monolithic mode).")
    p.add_argument("--data-hits-npz", type=str,   default=None,
                   help="Light per-PMT data .npz produced by run_data_hits.py "
                        "(chunked, low-memory mode for many part-files). "
                        "If given, --sig-parquet/--bkg-parquet are ignored.")
    p.add_argument("--mc-npz",        type=str,   required=True,
                   help="Path to WCSim MC .npz file. May be a single file, "
                        "a glob pattern with quotes (e.g. 'sim.part*.npz'), "
                        "or a directory containing the chunk .npz files.")
    p.add_argument("--geo-file",      type=str,   required=True,
                   help="WCSim geofile for tube mapping")
    p.add_argument("--source-pos-cm", type=float, nargs=3, default=[0, 80, 0],
                   help="Source position in MC coords [cm]")
    p.add_argument("--n-water",       type=float, default=1.33)
    p.add_argument("--trms-cut",      type=float, default=2.0)
    p.add_argument("--window",        type=float, default=20)
    p.add_argument("--thresh-min",    type=int,   default=2)
    p.add_argument("--max-nhits",     type=int,   default=50)
    p.add_argument("--output-dir",    type=str,   default="./output")
    return p.parse_args()


def hits_per_pmt(pmt_ids, n_pmts=N_PMTS):
    result = np.zeros(n_pmts)
    ids, counts = np.unique(pmt_ids[pmt_ids < n_pmts], return_counts=True)
    result[ids.astype(int)] = counts
    return result


def resolve_mc_files(mc_npz):
    """
    Resolve the --mc-npz argument into a sorted list of .npz files.

    Accepts:
      - a single .npz file            -> [that file]
      - a glob pattern (sim.part*.npz)-> sorted matches
      - a directory                   -> all *.npz inside, sorted

    Sorting is lexicographic, which keeps part000, part001, ... in order
    given the zero-padded naming written by event_dump_minimal.py.
    """
    if os.path.isdir(mc_npz):
        files = sorted(glob.glob(os.path.join(mc_npz, "*.npz")))
    elif any(ch in mc_npz for ch in "*?[]"):
        files = sorted(glob.glob(mc_npz))
    else:
        files = [mc_npz]

    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files matched --mc-npz='{mc_npz}'")
    return files


def compute_relative_qe(data_hits, mc_hits, n_cands_data, n_cands_mc):
    data_rate = np.zeros(N_PMTS)
    mc_rate   = np.zeros(N_PMTS)
    err_data  = np.zeros(N_PMTS)
    err_mc    = np.zeros(N_PMTS)

    md = data_hits > 0; mm = mc_hits > 0
    data_rate[md] = data_hits[md] / n_cands_data
    mc_rate[mm]   = mc_hits[mm]   / n_cands_mc
    err_data[md]  = np.sqrt(data_hits[md]) / n_cands_data
    err_mc[mm]    = np.sqrt(mc_hits[mm])   / n_cands_mc

    rqe     = np.zeros(N_PMTS)
    err_rqe = np.zeros(N_PMTS)
    mask    = (mc_rate > 0) & (data_rate > 0)
    rqe[mask] = data_rate[mask] / mc_rate[mask]
    err_rqe[mask] = np.sqrt(
        (err_data[mask] / mc_rate[mask])**2 +
        (data_rate[mask] * err_mc[mask] / mc_rate[mask]**2)**2
    )
    return rqe, err_rqe

def plot_background_subtraction(sig_hits, bkg_hits, pure_signal,
                                n_events_common, fig_dir):
    """
    Hits per PMT: SIG+BKG, BKG, and the background-subtracted pure signal.
    Both SIG and BKG use the same common events.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: SIG and BKG overlaid
    ax1.step(range(N_PMTS), sig_hits, where="mid", color="black",
             linewidth=1.0, label="SIG+BKG")
    ax1.step(range(N_PMTS), bkg_hits, where="mid", color="red",
             linewidth=1.0, label="BKG")
    ax1.set_ylabel("Hits per PMT")
    ax1.set_title(f"Hits per PMT — same {n_events_common} events for SIG and BKG")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Bottom: pure signal after subtraction
    ax2.step(range(N_PMTS), pure_signal, where="mid", color="blue",
             linewidth=1.0, label="Pure signal (SIG − BKG)")
    ax2.set_xlabel("PMT ID"); ax2.set_ylabel("Hits per PMT")
    ax2.set_title("Background-subtracted signal")
    ax2.legend(); ax2.grid(alpha=0.3)

    for ax in [ax1, ax2]:
        ax.axvspan(0,      21*19,  color="red",         alpha=0.07)
        ax.axvspan(21*19,  53*19,  color="forestgreen", alpha=0.07)
        ax.axvspan(53*19,  85*19,  color="yellow",      alpha=0.07)
        ax.axvspan(85*19, 105*19,  color="blue",        alpha=0.07)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "background_subtraction.png"), dpi=150)
    plt.close()

def main():
    args = parse_args()

    fig_dir  = os.path.join(args.output_dir, "figures")
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # ─── 1+2. Data side: pure signal hits per PMT and candidate counts ───
    # Two input modes:
    #   (a) --data-hits-npz : light arrays precomputed by run_data_hits.py
    #       (chunked, low-memory; use this for many part-files).
    #   (b) --sig-parquet/--bkg-parquet : monolithic DataFrames (original mode).
    if args.data_hits_npz is not None:
        print(f"Loading precomputed data hits from {args.data_hits_npz} ...")
        d = np.load(args.data_hits_npz)
        sig_hits        = d["sig_hits"]
        bkg_hits        = d["bkg_hits"]
        pure_signal     = d["pure_signal"]
        n_cands_sig     = int(d["n_cands_sig"])
        n_cands_bkg     = int(d["n_cands_bkg"])
        n_cands_pure    = int(d["n_cands_pure"])
        n_events_common = int(d["n_events_common"])

        # Sanity: the npz was made with some cuts; warn if they differ from CLI.
        if "trms_cut" in d and float(d["trms_cut"]) != args.trms_cut:
            print(f"  WARNING: npz was made with tRMS<{float(d['trms_cut'])} "
                  f"but --trms-cut={args.trms_cut}. The npz cuts are what "
                  f"applies; re-run run_data_hits.py to change them.")
        if "max_nhits" in d and int(d["max_nhits"]) != args.max_nhits:
            print(f"  WARNING: npz was made with nhits<={int(d['max_nhits'])} "
                  f"but --max-nhits={args.max_nhits}. The npz cuts are what "
                  f"applies.")

        print(f"  Common events SIG/BKG: {n_events_common}")
        print(f"  SIG candidates: {n_cands_sig}, BKG candidates: {n_cands_bkg}")
        print(f"  Pure signal candidates: {n_cands_pure}")

        plot_background_subtraction(sig_hits, bkg_hits, pure_signal,
                                    n_events_common, fig_dir)
    else:
        if args.sig_parquet is None or args.bkg_parquet is None:
            raise SystemExit("Provide either --data-hits-npz or both "
                             "--sig-parquet and --bkg-parquet.")

        # ─── 1. Load data DataFrames ───
        print("Loading data DataFrames...")
        df_sig = pd.read_parquet(args.sig_parquet)
        df_bkg = pd.read_parquet(args.bkg_parquet)

        if "trms" not in df_sig.columns:
            df_sig["trms"] = df_sig.groupby("candidate_id")[
                "hit_pmt_calibrated_times"].transform("std")
        if "trms" not in df_bkg.columns:
            df_bkg["trms"] = df_bkg.groupby("candidate_id")[
                "hit_pmt_calibrated_times"].transform("std")

        print(f"  SIG: {df_sig['candidate_id'].nunique()} candidates")
        print(f"  BKG: {df_bkg['candidate_id'].nunique()} candidates")

        # ─── 2. Data: select common events, apply cuts, subtract BKG ───
        print(f"\nApplying tRMS < {args.trms_cut} ns and "
              f"nhits <= {args.max_nhits} cuts to data...")
        df_sig_cut = df_sig[(df_sig["trms"] < args.trms_cut) &
                            (df_sig["nhits"] <= args.max_nhits)]
        df_bkg_cut = df_bkg[(df_bkg["trms"] < args.trms_cut) &
                            (df_bkg["nhits"] <= args.max_nhits)]

        common_events = np.intersect1d(
            df_sig_cut["event_id"].unique(),
            df_bkg_cut["event_id"].unique()
        )
        n_events_common = len(common_events)
        print(f"  Common events SIG/BKG: {n_events_common}")

        df_sig_matched = df_sig_cut[df_sig_cut["event_id"].isin(common_events)]
        df_bkg_matched = df_bkg_cut[df_bkg_cut["event_id"].isin(common_events)]

        sig_hits = hits_per_pmt(df_sig_matched["pmt_id"].values)
        bkg_hits = hits_per_pmt(df_bkg_matched["pmt_id"].values)
        pure_signal = sig_hits - bkg_hits
        pure_signal[pure_signal < 0] = 0

        plot_background_subtraction(sig_hits, bkg_hits, pure_signal,
                                    n_events_common, fig_dir)

        n_cands_sig = df_sig_matched["candidate_id"].nunique()
        n_cands_bkg = df_bkg_matched["candidate_id"].nunique()
        n_cands_pure = n_cands_sig - n_cands_bkg
        print(f"  SIG candidates: {n_cands_sig}, BKG candidates: {n_cands_bkg}")
        print(f"  Pure signal candidates: {n_cands_pure}")

    # ─── 3. Load and process MC through same pipeline (chunk-aware) ───
    mc_files = resolve_mc_files(args.mc_npz)
    print(f"\nProcessing MC from {len(mc_files)} file(s)...")
    from WCSimFilePackages.npz_to_df import truehits_info_to_df

    # Tube mapping is geometry, identical for every chunk: load once.
    tube_mapping = load_wcsim_tube_mapping(args.geo_file)

    # Accumulators: hits per PMT and candidate count, summed across chunks.
    mc_hits           = np.zeros(N_PMTS)
    n_cands_mc        = 0
    n_events_mc_total = 0
    n_cands_mc_raw    = 0

    for i, mc_file in enumerate(mc_files):
        tag = f"[{i+1}/{len(mc_files)}] {os.path.basename(mc_file)}"
        print(f"\n  {tag}")

        print("    Reading true hits...")
        df_mc_hits = read_mc_truehits(mc_file, truehits_info_to_df)
        n_ev = df_mc_hits["event_id"].nunique()
        n_events_mc_total += n_ev
        print(f"    {len(df_mc_hits)} hits, {n_ev} events")

        print(f"    Applying ToF correction to source {args.source_pos_cm}...")
        df_mc_hits = apply_tof_correction_mc(
            df_mc_hits, args.source_pos_cm, args.n_water
        )

        print(f"    Running trigger (w={args.window}, thresh_min={args.thresh_min})...")
        df_mc_cands = run_mc_trigger(
            df_mc_hits, w=args.window, thresh_min=args.thresh_min,
            time_col="true_hit_time_tof_corrected"
        )
        n_cands_mc_raw += len(df_mc_cands)

        df_mc_cands_cut = df_mc_cands[
            (df_mc_cands["trms"] < args.trms_cut) &
            (df_mc_cands["nhits"] <= args.max_nhits)
        ]
        n_cands_mc += len(df_mc_cands_cut)

        # Map this chunk's candidates to pmt_ids and accumulate hits per PMT.
        df_mc_hits_mapped = mc_cands_to_pmt_id(df_mc_cands_cut, tube_mapping)
        mc_hits += hits_per_pmt(df_mc_hits_mapped["pmt_id"].values)

        # Free chunk-level objects before the next iteration.
        del df_mc_hits, df_mc_cands, df_mc_cands_cut, df_mc_hits_mapped

    print(f"\n  MC totals across {len(mc_files)} file(s): "
          f"{n_events_mc_total} events, {n_cands_mc_raw} candidates found")
    print(f"  After tRMS < {args.trms_cut} ns and nhits <= {args.max_nhits}: "
          f"{n_cands_mc} MC candidates")

    # ─── 4. Compute and plot QE ───
    print("\nComputing relative QE (normalised by candidates)...")
    rqe, err_rqe = compute_relative_qe(
        pure_signal, mc_hits, n_cands_pure, n_cands_mc
    )

    # --- Load mPMT type dictionary ---
    import pickle
    # mpmt_info_path = os.path.join(os.path.dirname(args.geo_file), "other_mpmt_info.dict")
    mpmt_info_path = "/mnt/lustre/scratch/nlsas/home/usc/ie/dcr/hk/raw_data/other_mpmt_info.dict"
    
    pmt_color = np.full(N_PMTS, "gray", dtype=object)
    pmt_label_map = np.full(N_PMTS, "Other", dtype=object)
    
    try:
        with open(mpmt_info_path, "rb") as f:
            mpmt_info = pickle.load(f)
        
        for pmt_id in range(N_PMTS):
            slot = pmt_id // 19
            if slot not in mpmt_info:
                continue
            mtype = mpmt_info[slot].get("mpmt_type", "")
            msite = mpmt_info[slot].get("mpmt_site", "")
            
            if msite == "TRI" and mtype == "In-situ":
                pmt_color[pmt_id] = "blue"
                pmt_label_map[pmt_id] = "TRI In-situ"
            elif msite == "TRI" and mtype == "Ex-situ":
                pmt_color[pmt_id] = "green"
                pmt_label_map[pmt_id] = "TRI Ex-situ"
            elif msite == "WUT" and mtype == "In-situ":
                pmt_color[pmt_id] = "red"
                pmt_label_map[pmt_id] = "WUT In-situ"
            elif msite == "WUT" and mtype == "Ex-situ":
                pmt_color[pmt_id] = "orange"
                pmt_label_map[pmt_id] = "WUT Ex-situ"
        
        has_mpmt_info = True
    except FileNotFoundError:
        print(f"  Warning: {mpmt_info_path} not found, using uniform colors")
        has_mpmt_info = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 2]})

    data_rate = pure_signal / max(n_cands_pure, 1)
    mc_rate   = mc_hits / max(n_cands_mc, 1)

    ax1.step(range(N_PMTS), data_rate, where="mid", color="blue",
             label=f"Data pure SIG ({n_cands_pure} cands)")
    ax1.step(range(N_PMTS), mc_rate, where="mid", color="black",
             label=f"MC triggered+ToF+cuts ({n_cands_mc} cands)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Hits per PMT (normalised)")
    ax1.legend(); ax1.grid(alpha=0.3)

    for ax in [ax1, ax2]:
        ax.axvspan(0,      21*19,  color="red",         alpha=0.1)
        ax.axvspan(21*19,  53*19,  color="forestgreen", alpha=0.1)
        ax.axvspan(53*19,  85*19,  color="yellow",      alpha=0.1)
        ax.axvspan(85*19, 105*19,  color="blue",        alpha=0.1)

    # --- Plot errorbar colored by mPMT type ---
    if has_mpmt_info:
        group_defs = {
            "TRI In-situ": "blue",
            "TRI Ex-situ": "green",
            "WUT In-situ": "red",
            "WUT Ex-situ": "orange",
        }
        for label, color in group_defs.items():
            mask = pmt_label_map == label
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            ax2.errorbar(idx, rqe[mask], yerr=err_rqe[mask],
                         fmt="o", markersize=2, markerfacecolor=color,
                         markeredgecolor=color, ecolor=color,
                         elinewidth=0.5, capsize=0, alpha=0.7, label=label)
    else:
        ax2.errorbar(range(N_PMTS), rqe, yerr=err_rqe,
                     fmt="o", markersize=2, markerfacecolor="green",
                     markeredgecolor="green", ecolor="black",
                     elinewidth=0.5, capsize=0, label="Relative QE (Data / MC)")

    ax2.axhline(1.0, linestyle=":", color="k")
    ax2.set_ylabel("Relative QE (Data / MC)"); ax2.set_xlabel("PMT ID")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    plt.suptitle(f"Relative QE — tRMS < {args.trms_cut} ns, "
                 f"nhits ≤ {args.max_nhits}, "
                 f"w={args.window} ns, thresh>{args.thresh_min}", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "relative_qe.png"), dpi=150)
    plt.close()

    # ─── 5. Projection: RQE distribution by mPMT type ───
    fig_proj, ax_proj = plt.subplots(figsize=(8, 6))

    if has_mpmt_info:
        for label, color in group_defs.items():
            mask = (pmt_label_map == label) & (rqe > 0)
            if mask.sum() == 0:
                continue
            ax_proj.hist(rqe[mask], bins=50, range=(0, 2), histtype="step",
                         color=color, linewidth=1.5, 
                         label=f"{label} ({mask.sum()} PMTs, "
                               f"$\\mu$={rqe[mask].mean():.2f})")
    else:
        valid = rqe > 0
        ax_proj.hist(rqe[valid], bins=50, range=(0, 2), histtype="step",
                     color="green", linewidth=1.5,
                     label=f"All ({valid.sum()} PMTs, $\\mu$={rqe[valid].mean():.2f})")

    ax_proj.axvline(1.0, linestyle=":", color="k", label="RQE = 1")
    ax_proj.set_xlabel("Relative QE (Data / MC)")
    ax_proj.set_ylabel("PMTs")
    ax_proj.set_title(f"Relative QE distribution by mPMT type — tRMS < {args.trms_cut} ns — nhits ≤ {args.max_nhits}")
    ax_proj.legend(fontsize=8)
    ax_proj.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "relative_qe_projection.png"), dpi=150)
    plt.close()
    
    # ─── 5. Projection: RQE distribution by mPMT type ───
    fig_proj, ax_proj = plt.subplots(figsize=(8, 6))

    if has_mpmt_info:
        for label, color in group_defs.items():
            mask = (pmt_label_map == label) & (rqe > 0)
            if mask.sum() == 0:
                continue
            ax_proj.hist(rqe[mask], bins=50, range=(0, 2), histtype="step",
                         color=color, linewidth=1.5, 
                         label=f"{label} ({mask.sum()} PMTs, "
                               f"$\\mu$={rqe[mask].mean():.2f})")
    else:
        valid = rqe > 0
        ax_proj.hist(rqe[valid], bins=50, range=(0, 2), histtype="step",
                     color="green", linewidth=1.5,
                     label=f"All ({valid.sum()} PMTs, $\\mu$={rqe[valid].mean():.2f})")

    ax_proj.axvline(1.0, linestyle=":", color="k", label="RQE = 1")
    ax_proj.set_xlabel("Relative QE (Data / MC)")
    ax_proj.set_ylabel("PMTs")
    ax_proj.set_title(f"Relative QE distribution by mPMT type — tRMS < {args.trms_cut} ns — nhits ≤ {args.max_nhits}")
    ax_proj.legend(fontsize=8)
    ax_proj.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "relative_qe_projection.png"), dpi=150)
    plt.close()

    # Save
    qe_df = pd.DataFrame({
        "pmt_id": np.arange(N_PMTS),
        "relative_qe": rqe,
        "error_qe": err_rqe,
        "data_hits": pure_signal,
        "mc_hits": mc_hits,
    })
    qe_path = os.path.join(data_dir, "relative_qe.parquet")
    qe_df.to_parquet(qe_path, index=False)

    valid = rqe > 0
    print(f"\n--- QE Summary ---")
    print(f"  Common events (SIG/BKG subtraction): {n_events_common}")
    print(f"  Data candidates: SIG={n_cands_sig}, BKG={n_cands_bkg}, Pure={n_cands_pure}")
    print(f"  MC candidates (after cut):           {n_cands_mc}")
    print(f"  PMTs with valid QE:     {valid.sum()} / {N_PMTS}")
    print(f"  Mean relative QE:       {rqe[valid].mean():.3f}")
    print(f"  Std relative QE:        {rqe[valid].std():.3f}")
    print(f"\nSaved to {qe_path}")
    print("Done!")


if __name__ == "__main__":
    main()
