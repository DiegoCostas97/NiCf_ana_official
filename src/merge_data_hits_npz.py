#!/usr/bin/env python3
import argparse
import glob
import os

import numpy as np


ARRAY_KEYS = [
    "sig_hits",
    "bkg_hits",
    "pure_signal",
    "mean_charge_per_pmt",
]

SUM_KEYS = [
    "sig_hits",
    "bkg_hits",
    "n_cands_sig",
    "n_cands_bkg",
    "n_cands_pure",
    "n_events_common",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, help="Glob for partial .npz files")
    p.add_argument("--output", required=True, help="Merged output .npz")
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(glob.glob(args.inputs))
    if not files:
        raise SystemExit(f"No files matched: {args.inputs}")

    first = np.load(files[0])
    sig_hits = np.zeros_like(first["sig_hits"], dtype=float)
    bkg_hits = np.zeros_like(first["bkg_hits"], dtype=float)
    q_sum = np.zeros_like(first["sig_hits"], dtype=float)
    q_count = np.zeros_like(first["sig_hits"], dtype=float)

    n_cands_sig = 0
    n_cands_bkg = 0
    n_events_common = 0

    trms_cut = first["trms_cut"] if "trms_cut" in first else np.nan
    max_nhits = first["max_nhits"] if "max_nhits" in first else -1
    source_pos = first["source_pos"] if "source_pos" in first else np.zeros(3)

    for path in files:
        d = np.load(path)
        sig_hits += d["sig_hits"]
        bkg_hits += d["bkg_hits"]
        n_cands_sig += int(d["n_cands_sig"])
        n_cands_bkg += int(d["n_cands_bkg"])
        n_events_common += int(d["n_events_common"])

        if "q_sum_per_pmt" in d and "q_count_per_pmt" in d:
            q_sum += d["q_sum_per_pmt"]
            q_count += d["q_count_per_pmt"]
        elif "mean_charge_per_pmt" in d:
            # Fallback: cannot recover exact weighted mean without counts.
            pass

    pure_signal = sig_hits - bkg_hits
    pure_signal[pure_signal < 0] = 0
    n_cands_pure = n_cands_sig - n_cands_bkg
    mean_charge = np.divide(
        q_sum,
        q_count,
        out=np.zeros_like(q_sum),
        where=q_count > 0,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        sig_hits=sig_hits,
        bkg_hits=bkg_hits,
        pure_signal=pure_signal,
        n_cands_sig=n_cands_sig,
        n_cands_bkg=n_cands_bkg,
        n_cands_pure=n_cands_pure,
        n_events_common=n_events_common,
        mean_charge_per_pmt=mean_charge,
        q_sum_per_pmt=q_sum,
        q_count_per_pmt=q_count,
        trms_cut=trms_cut,
        max_nhits=max_nhits,
        source_pos=source_pos,
        merged_files=np.array(files),
    )

    print(f"Merged {len(files)} files -> {args.output}")
    print(f"Common events: {n_events_common}")
    print(f"Candidates: SIG={n_cands_sig}, BKG={n_cands_bkg}, pure={n_cands_pure}")


if __name__ == "__main__":
    main()