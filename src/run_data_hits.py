#!/usr/bin/env python3
"""
NiCf Analysis Pipeline — Data Hits-per-PMT (chunked, low memory)
=================================================================
Produces the light-weight per-PMT arrays that run_qe.py needs from data,
WITHOUT ever holding the full hit-level DataFrame of all part-files in RAM.

Motivation
----------
The monolithic df_sig / df_bkg parquets do not scale: with ~25 part-files the
hit-level DataFrames have hundreds of millions of rows and the process runs out
of memory (and the parquet write/read becomes fragile). But run_qe.py only ever
consumes, from the data side:

    sig_hits, bkg_hits  (hits per PMT, after cuts, on common SIG/BKG events)
    n_cands_sig, n_cands_bkg, n_cands_pure
    n_events_common

All of these are ADDITIVE across disjoint blocks of part-files. So we process
the part-files in blocks of --parts-per-chunk, apply the SAME cuts and the SAME
common-event SIG-BKG matching that run_qe.py does, accumulate the light arrays,
and free each block before the next.

Because part-files index different readout windows, events from different blocks
are disjoint, so summing per-block results is identical to processing everything
at once. Within each block SIG and BKG are read with the same part indices, so
their event_id values stay aligned for the common-event subtraction — exactly as
in the current pipeline.

The output is a small .npz that run_qe.py can consume instead of the giant
parquets (see --data-hits-npz wiring described at the bottom of this file).

Usage
-----
    python run_data_hits.py \
        --sig-run 1767 --bkg-run 1766 \
        --n-parts 25 --parts-per-chunk 5 \
        --data-dir /path/to/raw_data/production_v0 \
        --geo-json /path/to/wcte_v11_20250513.json \
        --source-pos 0 1525 0 \
        --trms-cut 2.0 --max-nhits 50 \
        --window 20 --thresh-min 2 \
        --output-dir ./1767
"""

import argparse
import os
import sys
import gc

import numpy as np

from functions import (
    load_pmt_positions, build_tof_map, read_run,
    nHits_greedy, build_candidates_dataframe, add_candidate_observables,
)

sys.path.append(os.path.abspath(
    os.environ.get("WCTE_SOFTWARE_DIR",
                   "/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk")))


from WCTE_BRB_Data_Analysis.wcte.brbtools import sort_run_files, get_part_files

N_PMTS = 2014


def parse_args():
    p = argparse.ArgumentParser(
        description="Chunked data hits-per-PMT for QE (low memory)")
    p.add_argument("--sig-run",    type=int,   required=True)
    p.add_argument("--bkg-run",    type=int,   required=True)
    p.add_argument("--n-parts",    type=int,   default=25,
                   help="Total number of part-files to process")
    p.add_argument("--parts-per-chunk", type=int, default=5,
                   help="How many part-files to hold in RAM at once. "
                        "Lower = less memory, more passes.")
    p.add_argument("--data-dir",   type=str,   required=True)
    p.add_argument("--geo-json",   type=str,   required=True)
    p.add_argument("--source-pos", type=float, nargs=3, default=[0, 1525, 0],
                   help="Source position [x, y, z] in mm (WCTE coords)")
    p.add_argument("--n-water",    type=float, default=1.33)
    p.add_argument("--window",     type=float, default=20)
    p.add_argument("--thresh-min", type=int,   default=2)
    p.add_argument("--trms-cut",   type=float, default=2.0)
    p.add_argument("--max-nhits",  type=int,   default=50)
    p.add_argument("--no-tof",     action="store_true", default=False)
    p.add_argument("--output-dir", type=str,   default="./output")
    p.add_argument("--part-start", type=int, default=0)
    p.add_argument("--part-stop", type=int, default=None)
    p.add_argument("--partial-tag", type=str, default=None)
    return p.parse_args()


def hits_per_pmt(pmt_ids, n_pmts=N_PMTS):
    result = np.zeros(n_pmts)
    ids, counts = np.unique(pmt_ids[pmt_ids < n_pmts], return_counts=True)
    result[ids.astype(int)] = counts
    return result


EVENTS_PER_PART_MAX = 1_000_000  # fixed multiplier; no part has this many windows


def process_one_part(run_number, run_files, part_index, tof_map,
                     window, thresh_min, label):
    """
    Read ONE part-file, run the trigger, build the candidate DataFrame with
    observables. The event_id is made globally unique and chunking-independent
    by offsetting the local window index with part_index * EVENTS_PER_PART_MAX.

    Processing one part at a time (rather than a concatenated block) is what
    lets us assign each physical readout window the SAME event_id regardless of
    how parts are grouped into chunks, and keeps SIG and BKG aligned: window N
    of part P in SIG gets the same id as window N of part P in BKG.
    """
    data = read_run(run_number, run_files, [part_index], 1, tof_map=tof_map)
    idx, times, _ = nHits_greedy(data['hit_times'], window, thresh_min)
    df = build_candidates_dataframe(idx, times, data, label)
    df["event_id"] = df["event_id"] + part_index * EVENTS_PER_PART_MAX
    # candidate_id is numbered from 0 inside each call, so offset it too to
    # keep it unique after concatenating parts/blocks (nunique() must be exact).
    df["candidate_id"] = df["candidate_id"] + part_index * EVENTS_PER_PART_MAX
    df = add_candidate_observables(df)
    return df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data_dir_out = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir_out, exist_ok=True)

    # ─── ToF map ───
    print("Loading PMT geometry and building ToF map...")
    pmt_positions = load_pmt_positions(args.geo_json)
    tof_map = None if args.no_tof else build_tof_map(
        pmt_positions, args.source_pos, args.n_water)
    print(f"  {len(pmt_positions)} PMTs, source at {args.source_pos}, "
          f"ToF {'OFF' if args.no_tof else 'ON'}")

    # ─── Resolve part-file lists for SIG and BKG ───
    sig_files = sort_run_files(
        f"{args.data_dir}/{args.sig_run}/WCTE_offline_R{args.sig_run}S*P*.root")
    bkg_files = sort_run_files(
        f"{args.data_dir}/{args.bkg_run}/WCTE_offline_R{args.bkg_run}S*P*.root")
    part_stop = args.part_stop if args.part_stop is not None else args.n_parts

    sig_parts = get_part_files(sig_files)[args.part_start:part_stop]
    bkg_parts = get_part_files(bkg_files)[args.part_start:part_stop]

    # SIG and BKG must be matched part-by-part for the common-event
    # subtraction to be valid, so we iterate over the shorter list.
    n_blocks_parts = min(len(sig_parts), len(bkg_parts))
    if len(sig_parts) != len(bkg_parts):
        print(f"  WARNING: SIG has {len(sig_parts)} parts, BKG has "
              f"{len(bkg_parts)}. Using first {n_blocks_parts} of each.")
    sig_parts = sig_parts[:n_blocks_parts]
    bkg_parts = bkg_parts[:n_blocks_parts]

    ppc = args.parts_per_chunk
    n_chunks = int(np.ceil(n_blocks_parts / ppc))
    print(f"  Processing {n_blocks_parts} part-files in {n_chunks} "
          f"block(s) of up to {ppc}")

    # ─── Accumulators (light: arrays of length N_PMTS + scalars) ───
    sig_hits      = np.zeros(N_PMTS)   # after cuts, common events
    bkg_hits      = np.zeros(N_PMTS)   # after cuts, common events
    n_cands_sig   = 0
    n_cands_bkg   = 0
    n_events_common = 0

    # Charge accumulators for the per-PMT calibration check (sum and count
    # per PMT let us recover the mean without storing every hit).
    q_sum   = np.zeros(N_PMTS)
    q_count = np.zeros(N_PMTS)

    for c in range(n_chunks):
        block_sig = sig_parts[c*ppc:(c+1)*ppc]
        block_bkg = bkg_parts[c*ppc:(c+1)*ppc]
        print(f"\n=== Block {c+1}/{n_chunks}: parts {block_sig} ===")

        for sig_part, bkg_part in zip(block_sig, block_bkg):
            print(f"  SIG part {sig_part}...")

            df_sig = process_one_part(
                args.sig_run, sig_files, sig_part, tof_map,
                args.window, args.thresh_min, "sig"
            )

            # Charge sums BEFORE cuts.
            gq = df_sig.groupby("pmt_id")["hit_pmt_charges"]
            for pid, s in gq.sum().items():
                if 0 <= pid < N_PMTS:
                    q_sum[int(pid)] += s
            for pid, n in gq.count().items():
                if 0 <= pid < N_PMTS:
                    q_count[int(pid)] += n

            df_sig_cut = df_sig.loc[
                (df_sig["trms"] < args.trms_cut)
                & (df_sig["nhits"] <= args.max_nhits),
                ["event_id", "candidate_id", "pmt_id"],
            ].copy()

            del df_sig, gq
            gc.collect()

            print(f"  BKG part {bkg_part}...")

            df_bkg = process_one_part(
                args.bkg_run, bkg_files, bkg_part, tof_map,
                args.window, args.thresh_min, "bkg"
            )

            df_bkg_cut = df_bkg.loc[
                (df_bkg["trms"] < args.trms_cut)
                & (df_bkg["nhits"] <= args.max_nhits),
                ["event_id", "candidate_id", "pmt_id"],
            ].copy()

            del df_bkg
            gc.collect()

            common = np.intersect1d(
                df_sig_cut["event_id"].unique(),
                df_bkg_cut["event_id"].unique(),
            )
            n_events_common += len(common)

            df_sig_m = df_sig_cut[df_sig_cut["event_id"].isin(common)]
            df_bkg_m = df_bkg_cut[df_bkg_cut["event_id"].isin(common)]

            sig_hits += hits_per_pmt(df_sig_m["pmt_id"].values)
            bkg_hits += hits_per_pmt(df_bkg_m["pmt_id"].values)
            n_cands_sig += df_sig_m["candidate_id"].nunique()
            n_cands_bkg += df_bkg_m["candidate_id"].nunique()

            print(
                f"  part common events: {len(common)}, "
                f"SIG cands: {df_sig_m['candidate_id'].nunique()}, "
                f"BKG cands: {df_bkg_m['candidate_id'].nunique()}"
            )

            del df_sig_cut, df_bkg_cut, df_sig_m, df_bkg_m, common
            gc.collect()

    # ─── Pure signal ───
    pure_signal = sig_hits - bkg_hits
    pure_signal[pure_signal < 0] = 0
    n_cands_pure = n_cands_sig - n_cands_bkg

    # Per-PMT mean raw charge (for the calibration plot in run_qe, if wanted)
    mean_charge = np.divide(q_sum, q_count,
                            out=np.zeros_like(q_sum), where=q_count > 0)

    # ─── Save light npz ───
    # out_path = os.path.join(data_dir_out, f"data_hits_R{args.sig_run}.npz")
    
    tag = f"_{args.partial_tag}" if args.partial_tag else ""
    out_path = os.path.join(data_dir_out, f"data_hits_R{args.sig_run}{tag}.npz")

    np.savez(
        out_path,
        sig_hits=sig_hits,
        bkg_hits=bkg_hits,
        pure_signal=pure_signal,
        n_cands_sig=n_cands_sig,
        n_cands_bkg=n_cands_bkg,
        n_cands_pure=n_cands_pure,
        n_events_common=n_events_common,
        mean_charge_per_pmt=mean_charge,
        trms_cut=args.trms_cut,
        max_nhits=args.max_nhits,
        source_pos=np.array(args.source_pos),
        q_sum_per_pmt=q_sum,
        q_count_per_pmt=q_count
    )

    print(f"\n--- Data hits summary ---")
    print(f"  Common events (SIG/BKG): {n_events_common}")
    print(f"  Candidates: SIG={n_cands_sig}, BKG={n_cands_bkg}, "
          f"Pure={n_cands_pure}")
    print(f"  PMTs with signal: {(pure_signal > 0).sum()} / {N_PMTS}")
    print(f"\nSaved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()