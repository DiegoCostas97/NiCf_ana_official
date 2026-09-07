#!/usr/bin/env python3
import argparse
import os
import sys
import gc

import numpy as np
import pandas as pd

from functions import (
    load_pmt_positions, build_tof_map, read_run,
    nHits_greedy, build_candidates_dataframe, add_candidate_observables,
)

sys.path.append(os.path.abspath(
    os.environ.get("WCTE_SOFTWARE_DIR",
                   "/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk")
))
from WCTE_BRB_Data_Analysis.wcte.brbtools import sort_run_files, get_part_files

EVENTS_PER_PART_MAX = 1_000_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create candidate CSV for multilateration reconstruction. "
                    "Can read an existing signal parquet or stream ROOT part-files."
    )

    # Legacy mode.
    parser.add_argument("--sig-parquet", default=None,
                        help="Input signal parquet file (legacy mode).")

    # Chunked ROOT mode.
    parser.add_argument("--sig-run", type=int, default=None,
                        help="Signal run number for chunked ROOT mode.")
    parser.add_argument("--n-parts", type=int, default=25,
                        help="Total number of part-files to process.")
    parser.add_argument("--parts-per-chunk", type=int, default=1,
                        help="How many ROOT part-files to process before flushing rows.")
    parser.add_argument("--part-start", type=int, default=0)
    parser.add_argument("--part-stop", type=int, default=None)
    parser.add_argument("--data-dir", default=None,
                        help="Root data directory containing run folders.")
    parser.add_argument("--geo-json", default=None,
                        help="WCTE geometry JSON for ToF correction.")
    parser.add_argument("--source-pos", type=float, nargs=3, default=[0, 1525, 0],
                        help="Source position [x,y,z] in WCTE mm.")
    parser.add_argument("--n-water", type=float, default=1.33)
    parser.add_argument("--window", type=float, default=20)
    parser.add_argument("--thresh-min", type=int, default=2)
    parser.add_argument("--no-tof", action="store_true", default=False)

    parser.add_argument("--out-csv", required=True,
                        help="Output CSV file for multilateration.")
    parser.add_argument("--trms-cut", type=float, default=2.0,
                        help="Maximum tRMS accepted.")
    parser.add_argument("--min-hits", type=int, default=6,
                        help="Minimum number of hits required per candidate.")
    parser.add_argument("--max-nhits", type=int, default=None,
                        help="Optional maximum hits per candidate.")

    return parser.parse_args()


def clusters_from_dataframe(df_sig, trms_cut, min_hits, max_nhits):
    if "trms" not in df_sig.columns or "nhits" not in df_sig.columns:
        df_sig = add_candidate_observables(df_sig)

    if "hit_pmt_calibrated_times_raw" not in df_sig.columns:
        # If the input was intentionally read with no ToF correction, the
        # calibrated time is already the raw time used by the multilaterator.
        df_sig = df_sig.copy()
        df_sig["hit_pmt_calibrated_times_raw"] = df_sig["hit_pmt_calibrated_times"]

    sel = df_sig["trms"] < trms_cut
    if max_nhits is not None:
        sel &= df_sig["nhits"] <= max_nhits
    df_cut = df_sig[sel]

    df_clusters = df_cut.groupby("candidate_id").agg(
        event_id=("event_id", "first"),
        hit_times_ns=("hit_pmt_calibrated_times_raw", list),
        hit_slot_ids=("hit_mpmt_slot_ids", list),
        hit_channel_ids=("hit_pmt_position_ids", list),
        nhits=("hit_pmt_calibrated_times_raw", "count"),
        trms=("trms", "first"),
    ).reset_index()

    df_clusters = df_clusters[df_clusters["nhits"] >= min_hits]
    return df_clusters


def write_clusters(df_clusters, out_csv, append):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df_clusters.to_csv(out_csv, index=False, mode="a" if append else "w",
                       header=not append)


def run_legacy_parquet_mode(args):
    print(f"Loading signal parquet: {args.sig_parquet}")
    df_sig = pd.read_parquet(args.sig_parquet)
    df_clusters = clusters_from_dataframe(
        df_sig, args.trms_cut, args.min_hits, args.max_nhits)
    write_clusters(df_clusters, args.out_csv, append=False)
    print(f"Prepared {len(df_clusters)} candidates for reconstruction")


def process_one_part(run_number, run_files, part_index, tof_map,
                     window, thresh_min):
    data = read_run(run_number, run_files, [part_index], 1, tof_map=tof_map)
    idx, times, _ = nHits_greedy(data["hit_times"], window, thresh_min)
    df = build_candidates_dataframe(idx, times, data, "sig")

    offset = part_index * EVENTS_PER_PART_MAX
    df["event_id"] = df["event_id"] + offset
    df["candidate_id"] = df["candidate_id"] + offset
    df = add_candidate_observables(df)

    if tof_map is None:
        df["hit_pmt_calibrated_times_raw"] = df["hit_pmt_calibrated_times"]
    else:
        df["hit_pmt_calibrated_times_raw"] = (
            df["hit_pmt_calibrated_times"] + df["pmt_id"].map(tof_map)
        )
        df["trms_raw"] = df.groupby("candidate_id")[
            "hit_pmt_calibrated_times_raw"].transform("std")

    return df


def run_chunked_root_mode(args):
    missing = []
    for name, value in (("--sig-run", args.sig_run),
                        ("--data-dir", args.data_dir),
                        ("--geo-json", args.geo_json)):
        if value is None:
            missing.append(name)
    if missing:
        raise SystemExit("Chunked ROOT mode requires " + ", ".join(missing))

    print("Loading PMT geometry and building ToF map...")
    pmt_positions = load_pmt_positions(args.geo_json)
    tof_map = None if args.no_tof else build_tof_map(
        pmt_positions, args.source_pos, args.n_water)
    print(f"  {len(pmt_positions)} PMTs, source at {args.source_pos}, "
          f"ToF {'OFF' if args.no_tof else 'ON'}")

    sig_files = sort_run_files(
        f"{args.data_dir}/{args.sig_run}/WCTE_offline_R{args.sig_run}S*P*.root")
    all_parts = get_part_files(sig_files)
    part_stop = args.part_stop if args.part_stop is not None else args.n_parts
    parts = all_parts[args.part_start:part_stop]

    if not parts:
        raise SystemExit("No signal parts selected")

    ppc = max(int(args.parts_per_chunk), 1)
    total_written = 0
    append = False

    print(f"Processing {len(parts)} signal part-file(s) in blocks of up to {ppc}")
    for block_start in range(0, len(parts), ppc):
        block = parts[block_start:block_start + ppc]
        print(f"\n=== Candidate block parts {block} ===")
        frames = []
        for part in block:
            print(f"  SIG part {part}...")
            df_part = process_one_part(
                args.sig_run, sig_files, part, tof_map,
                args.window, args.thresh_min)
            frames.append(df_part)

        df_block = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        df_clusters = clusters_from_dataframe(
            df_block, args.trms_cut, args.min_hits, args.max_nhits)
        write_clusters(df_clusters, args.out_csv, append=append)
        append = True
        total_written += len(df_clusters)
        print(f"  Wrote {len(df_clusters)} candidates from this block")

        del frames, df_block, df_clusters
        gc.collect()

    print(f"Prepared {total_written} candidates for reconstruction")


def main():
    args = parse_args()
    if args.sig_parquet:
        run_legacy_parquet_mode(args)
    else:
        run_chunked_root_mode(args)
    print(f"Output: {args.out_csv}")


if __name__ == "__main__":
    main()
