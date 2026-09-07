#!/usr/bin/env python3
"""
NiCf Analysis Pipeline — MC clusters for multilateration
=========================================================
Produces a `candidates_for_reco.csv` for the MONTE CARLO that has the EXACT
same format as the data one, so it can be fed to the very same reconstruction
script (multilat_vertex_reconstruction.py) used for data.

Why this script exists
----------------------
For data you do:
    df_sig  ->  groupby candidate_id  ->  (hit_times_ns, hit_slot_ids,
               hit_channel_ids, nhits, trms)  ->  candidates_for_reco.csv
    then:  python multilat_vertex_reconstruction.py --csv candidates_for_reco.csv ...

For the angular-response method to be unbiased, the MC vertices must be
reconstructed with the SAME multilaterator and the SAME geometry as data
(so the reconstruction bias cancels in the data/MC ratio). This script builds
the MC equivalent of candidates_for_reco.csv.

Key correctness points
-----------------------
1. TIMES: the multilaterator expects RAW (non-ToF-corrected) times, because it
   computes the time-of-flight internally from the vertex it fits. In data you
   pass `hit_pmt_calibrated_times_raw`. The MC analogue is the RAW true hit time
   `true_hit_time` (NOT the ToF-corrected one). This script triggers on the
   ToF-corrected time (to reproduce exactly the candidate selection of
   run_qe.py / run_angular_vertex.py) but then writes the RAW `true_hit_time`
   for each hit into the CSV.

2. PER-HIT alignment: run_mc_trigger() stores only UNIQUE tube numbers and a
   sorted time list, which loses the time<->PMT correspondence and the hit
   multiplicity. So we DO NOT use cand["all_pmts"]; instead we re-extract the
   hits of each candidate directly from df_mc_hits by matching the candidate's
   stored times back to the event hits (same trick as the reflection study),
   which preserves the per-hit (time, tube, channel) triplets.

3. GEOMETRY MAPPING: WCSim tube numbers are mapped to (mPMT slot, PMT position)
   via the WCSim geofile, exactly as functions.mc_cands_to_pmt_id does
   (tube_key = tube_no + 1; position returned 0-indexed). The data multilat uses
   hit_slot_ids = mpmt_slot and hit_channel_ids = pmt_position (0..18), so the
   MC must use the SAME scheme: hit_slot_ids = slot, hit_channel_ids = pos.

Output
------
A CSV with columns:
    candidate_id, event_id, hit_times_ns, hit_slot_ids, hit_channel_ids,
    nhits, trms
identical in structure to the data candidates_for_reco.csv. Lists are written
with Python repr so that multilat_vertex_reconstruction.py reads them back with
its `converters={... : eval}`.

Usage
-----
    python build_mc_clusters_for_reco.py \
        --mc-npz   .../1Mneutrons_..._pos1769_...newTuning.npz \
        --geo-file .../geofile_NuPRISMBeamTest_16cShort_mPMT.txt \
        --source-pos-cm 0.0 47.6 117.58 \
        --out-csv  .../1769/data/mc_candidates_for_reco.csv \
        --trms-cut 6.0 --min-hits 6 --window 20 --thresh-min 2

Then reconstruct exactly like data:
    python multilat_vertex_reconstruction.py \
        --csv    .../1769/data/mc_candidates_for_reco.csv \
        --outdir .../1769/data --verbose

That yields .../1769/data/mc_candidates_for_reco_multilat_chi2.csv, which you
pass to run_angular_vertex.py as the MC vertex CSV.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from functions import (
    load_wcsim_tube_mapping,
    read_mc_truehits,
    apply_tof_correction_mc,
    run_mc_trigger,
)

# WCTE software dir (for WCSimFilePackages), same convention as run_qe.py
sys.path.append(os.path.abspath(
    os.environ.get("WCTE_SOFTWARE_DIR",
                   "/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk")
))

CANDIDATES_PER_FILE_MAX = 100_000_000


def resolve_mc_files(mc_npz):
    if os.path.isdir(mc_npz):
        files = sorted(glob.glob(os.path.join(mc_npz, "*.npz")))
    elif any(ch in mc_npz for ch in "*?[]"):
        files = sorted(glob.glob(mc_npz))
    else:
        files = [mc_npz]
    if not files:
        raise FileNotFoundError(f"No .npz files matched --mc-npz='{mc_npz}'")
    return files


def parse_args():
    p = argparse.ArgumentParser(
        description="Build MC candidate clusters CSV for the multilaterator "
                    "(same format as the data candidates_for_reco.csv)")
    p.add_argument("--mc-npz",   required=True,
                   help="WCSim .npz for this source position. May be a single "
                        "file, a glob pattern, or a directory of chunk .npz files.")
    p.add_argument("--geo-file", required=True, help="WCSim geofile (tube -> slot,pos)")
    p.add_argument("--source-pos-cm", type=float, nargs=3, required=True,
                   help="Ni ball position in MC coords [cm], for the ToF-correction "
                        "used ONLY to reproduce the candidate selection")
    p.add_argument("--out-csv", required=True, help="Output CSV path")
    # selection / trigger — MUST match run_qe.py / run_angular_vertex.py
    p.add_argument("--n-water",    type=float, default=1.33)
    p.add_argument("--window",     type=float, default=20)
    p.add_argument("--thresh-min", type=int,   default=2)
    p.add_argument("--trms-cut",   type=float, default=6.0,
                   help="Loose tRMS cut for which candidates to reconstruct "
                        "(use the same loose cut you used for data, e.g. 6 ns)")
    p.add_argument("--max-nhits",  type=int,   default=None,
                   help="Optional upper nhits cut (leave None for reco stage)")
    p.add_argument("--min-hits",   type=int,   default=6,
                   help="Minimum hits per candidate (multilaterator needs >=6)")
    p.add_argument("--time-match-atol", type=float, default=1e-3,
                   help="Abs. tolerance [ns] to match candidate times to event hits")
    return p.parse_args()


def main():
    args = parse_args()
    from WCSimFilePackages.npz_to_df import truehits_info_to_df

    # ─── 5. Tube -> (slot, pos) mapping ───
    tube_mapping = load_wcsim_tube_mapping(args.geo_file)
    mc_files = resolve_mc_files(args.mc_npz)
    print(f"Processing MC from {len(mc_files)} file(s)")

    rows_all = []
    n_skip_match = 0
    n_skip_tube  = 0
    n_cands_raw_total = 0
    n_cands_cut_total = 0

    for file_idx, mc_file in enumerate(mc_files):
        cand_offset = file_idx * CANDIDATES_PER_FILE_MAX
        print(f"\n[{file_idx + 1}/{len(mc_files)}] {os.path.basename(mc_file)}")

        print("  Reading MC true hits...")
        df_mc_hits = read_mc_truehits(mc_file, truehits_info_to_df)
        print(f"    {len(df_mc_hits)} hits, {df_mc_hits['event_id'].nunique()} events")

        print(f"  ToF correction to source {args.source_pos_cm} cm (selection only)...")
        df_mc_hits = apply_tof_correction_mc(
            df_mc_hits, args.source_pos_cm, args.n_water)

        print(f"  Running trigger (w={args.window}, thresh_min={args.thresh_min})...")
        df_mc_cands = run_mc_trigger(
            df_mc_hits, w=args.window, thresh_min=args.thresh_min,
            time_col="true_hit_time_tof_corrected")
        n_cands_raw_total += len(df_mc_cands)

        sel = ((df_mc_cands["trms"] < args.trms_cut) &
               (df_mc_cands["nhits"] >= args.min_hits))
        if args.max_nhits is not None:
            sel &= (df_mc_cands["nhits"] <= args.max_nhits)
        df_mc_cands = df_mc_cands[sel].reset_index(drop=True)
        n_cands_cut_total += len(df_mc_cands)
        print(f"    {len(df_mc_cands)} candidates after tRMS<{args.trms_cut} "
              f"and nhits>={args.min_hits}")

        print("  Re-extracting per-hit clusters (RAW times) and mapping tubes...")
        grouped = {ev: d for ev, d in df_mc_hits.groupby("event_id")}

        rows = []
        for _, cand in tqdm(df_mc_cands.iterrows(), total=len(df_mc_cands)):
            ev = cand["event_id"]
            ev_hits = grouped.get(ev)
            if ev_hits is None:
                continue

            t_tofcorr = ev_hits["true_hit_time_tof_corrected"].values.astype(np.float64)
            t_raw     = ev_hits["true_hit_time"].values.astype(np.float64)
            tubes     = ev_hits["true_hit_pmt"].values

            cand_times = np.asarray(cand["all_times"], dtype=np.float64)

            hit_times_ns   = []
            hit_slot_ids   = []
            hit_channel_id = []

            for t in cand_times:
                j = int(np.argmin(np.abs(t_tofcorr - t)))
                if abs(t_tofcorr[j] - t) > args.time_match_atol:
                    n_skip_match += 1
                    continue
                tube_no = int(tubes[j])
                mp = tube_mapping.get(tube_no + 1)
                if mp is None:
                    n_skip_tube += 1
                    continue
                slot, pos = mp
                hit_times_ns.append(float(t_raw[j]))
                hit_slot_ids.append(int(slot))
                hit_channel_id.append(int(pos))

            if len(hit_times_ns) < args.min_hits:
                continue

            rows.append({
                "candidate_id":    int(cand["candidate_id"]) + cand_offset,
                "event_id":        int(ev),
                "hit_times_ns":    repr(hit_times_ns),
                "hit_slot_ids":    repr(hit_slot_ids),
                "hit_channel_ids": repr(hit_channel_id),
                "nhits":           len(hit_times_ns),
                "trms":            float(cand["trms"]),
            })

        rows_all.extend(rows)
        del df_mc_hits, df_mc_cands, grouped

    df_clusters = pd.DataFrame(rows_all)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df_clusters.to_csv(args.out_csv, index=False)

    print("\n--- Summary ---")
    print(f"  MC files processed:            {len(mc_files)}")
    print(f"  MC candidates before cuts:     {n_cands_raw_total}")
    print(f"  MC candidates after cuts:      {n_cands_cut_total}")
    print(f"  MC candidate clusters written: {len(df_clusters)}")
    print(f"  Hits skipped (time match):     {n_skip_match}")
    print(f"  Hits skipped (tube unmapped):  {n_skip_tube}")
    print(f"  Output: {args.out_csv}")
    print("\nNext step (identical to data):")
    print(f"  python multilat_vertex_reconstruction.py \\")
    print(f"      --csv {args.out_csv} \\")
    print(f"      --outdir {os.path.dirname(os.path.abspath(args.out_csv))} --verbose")
    print("Done!")


if __name__ == "__main__":
    main()
