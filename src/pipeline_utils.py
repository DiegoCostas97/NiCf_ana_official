#!/usr/bin/env python3
"""
NiCf Analysis Pipeline — shared low-level helpers
=================================================
Small utilities that were previously duplicated (verbatim) in run_qe.py,
build_mc_clusters_for_reco.py and run_angular_vertex.py. Keeping a single copy
guarantees that the three stages resolve MC files in the SAME order and match
candidate hits with the SAME algorithm, which matters because:

  * MC candidate IDs are made globally unique with an offset that depends on the
    position of each .npz chunk in the sorted file list (see
    CANDIDATES_PER_FILE_MAX below). If two stages enumerated the chunks in a
    different order, the MC vertex CSV would be silently mis-associated with the
    MC hits.
  * The per-hit re-extraction (matching a candidate's stored times back to the
    event hit table) has to be identical in the script that writes the MC
    clusters and in the script that later histograms them.

This module is deliberately dependency-light: numpy only.
"""

import glob
import os

import numpy as np

# Total number of readout channels in WCTE: 106 mPMT slots x 19 PMTs.
# The pmt_id convention used everywhere in this pipeline is
#     pmt_id = mpmt_slot * 19 + pmt_position        (both 0-indexed)
N_PMTS = 2014

# Offset applied to MC candidate IDs so that candidates coming from different
# .npz chunks of the same production never collide:
#     global_candidate_id = local_candidate_id + chunk_index * CANDIDATES_PER_FILE_MAX
# The value must be larger than the number of candidates any single chunk can
# produce (a 50 M-neutron chunk yields O(10^5-10^6) candidates, so 10^8 is safe)
# and it must be IDENTICAL in build_mc_clusters_for_reco.py and
# run_angular_vertex.py, otherwise the MC vertices cannot be matched back.
CANDIDATES_PER_FILE_MAX = 100_000_000


def resolve_npz_files(spec):
    """
    Turn a user-supplied MC path specification into a sorted list of .npz files.

    Accepted forms
    --------------
    single file      '/path/sim.npz'            -> ['/path/sim.npz']
    glob pattern     '/path/sim.part*.npz'      -> sorted matches
    directory        '/path/npz'                -> sorted '/path/npz/*.npz'

    The sort is lexicographic, which keeps zero-padded chunk names
    (part000, part001, ...) in production order. This ordering defines the
    per-chunk candidate-ID offsets, so it must never be randomised.
    """
    if os.path.isdir(spec):
        files = sorted(glob.glob(os.path.join(spec, "*.npz")))
    elif any(ch in spec for ch in "*?[]"):
        files = sorted(glob.glob(spec))
    else:
        files = [spec]

    if not files:
        raise FileNotFoundError(f"No .npz files matched '{spec}'")
    return files


def wcsim_cm_to_wcte_mm_offset(geo_file, geo_json, verbose=True):
    """
    Derive the transform between the two detector frames used in this analysis.

    The two geometry descriptions are NOT the same frame:

      * the WCSim geofile (`geofile_NuPRISMBeamTest_16cShort_mPMT.txt`) gives PMT
        positions in **cm**, with the origin at the centre of the tank;
      * the WCTE geometry JSON (`wcte_v11_*.json`) gives them in **mm**, with an
        origin about 42 cm below the tank centre.

    Matching the two channel by channel (pmt_id = slot * 19 + position) gives

        r_WCTE[mm] = 10 * r_WCSim[cm] + offset,     offset ~ (0, 425, 0) mm

    which is consistent with the tank centre sitting at y = 424 mm in the WCTE
    frame. This matters because the multilaterator reconstructs vertices from
    the WCSim geofile and therefore returns them in **WCSim cm**, while the
    angular-response code compares them against the JSON positions in WCTE mm.

    The offset is fitted here rather than hardcoded so that it follows any
    change of geometry file. The per-channel residual is also reported: the two
    descriptions differ by a few cm on individual channels (different mPMT dome
    modelling), which is an irreducible systematic on the vertex-to-PMT distance.

    Returns
    -------
    offset  : (3,) array [mm]
    resid   : (3,) array, per-axis RMS of the residual after the shift [mm]
    """
    import json as _json

    geo = np.loadtxt(geo_file, skiprows=5, usecols=(0, 1, 2, 3, 4, 5))
    slots = geo[:, 1].astype(int)
    poss = geo[:, 2].astype(int) - 1          # geofile positions are 1-indexed
    pmt_ids = slots * 19 + poss
    wcsim_mm = geo[:, 3:6] * 10.0             # cm -> mm

    with open(geo_json, "r") as fh:
        j = _json.load(fh)
    wcte = {}
    for mpmt_idx, mpmt in j["mpmts"].items():
        for pmt_idx, pmt in mpmt["pmts"].items():
            wcte[int(mpmt_idx) * 19 + int(pmt_idx)] = pmt["placement"]["location"]

    keep = np.array([p in wcte for p in pmt_ids])
    if keep.sum() == 0:
        raise RuntimeError("No channel is present in both geometry files; "
                           "the frame transform cannot be derived.")
    ref = np.array([wcte[p] for p in pmt_ids[keep]], dtype=np.float64)
    delta = ref - wcsim_mm[keep]
    offset = delta.mean(axis=0)
    resid = delta.std(axis=0)

    if verbose:
        print(f"  WCSim(cm) -> WCTE(mm) transform from {keep.sum()} common "
              f"channels: r_mm = 10*r_cm + "
              f"({offset[0]:.1f}, {offset[1]:.1f}, {offset[2]:.1f}) mm "
              f"[per-channel RMS {resid[0]:.0f}/{resid[1]:.0f}/{resid[2]:.0f} mm]")
    return offset, resid


def match_times_to_hits(hit_times, cand_times, atol=1e-3):
    """
    Map every time stored in a candidate back to its row in the event hit table.

    Why this is needed
    ------------------
    The MC trigger (functions.run_mc_trigger) stores, per candidate, the sorted
    list of hit times and the list of UNIQUE tube numbers. The unique-ing loses
    both the hit multiplicity and the time <-> tube correspondence, so the only
    way to recover the per-hit (time, tube) pairs is to look each stored time up
    in the event's hit table. Times are float32 taken straight from the same
    array, so an exact-to-1e-3 ns match identifies the row unambiguously.

    Implementation
    --------------
    A naive `argmin(|hit_times - t|)` per candidate time is O(n_hits) per time
    and dominates the runtime on large MC productions. Sorting the event's hit
    times once and binary-searching each candidate time is O((n+m) log n) and
    returns exactly the same nearest neighbour.

    Parameters
    ----------
    hit_times  : (n,) array of event hit times (any order).
    cand_times : (m,) array of times stored in the candidate.
    atol       : maximum |dt| accepted as a match [ns].

    Returns
    -------
    idx  : (m,) int array. Index into `hit_times` of the nearest hit for each
           candidate time, or -1 when no hit is within `atol`.
    """
    hit_times = np.asarray(hit_times, dtype=np.float64)
    cand_times = np.asarray(cand_times, dtype=np.float64).ravel()

    n = hit_times.size
    if n == 0 or cand_times.size == 0:
        return np.full(cand_times.size, -1, dtype=np.int64)

    order = np.argsort(hit_times, kind="stable")
    sorted_times = hit_times[order]

    # Insertion points, then compare with the neighbour on each side and keep
    # whichever is closer (standard nearest-neighbour-on-sorted-array pattern).
    pos = np.searchsorted(sorted_times, cand_times)
    left = np.clip(pos - 1, 0, n - 1)
    right = np.clip(pos, 0, n - 1)

    d_left = np.abs(sorted_times[left] - cand_times)
    d_right = np.abs(sorted_times[right] - cand_times)
    take_left = d_left <= d_right

    best_sorted = np.where(take_left, left, right)
    best_dist = np.where(take_left, d_left, d_right)

    idx = order[best_sorted]
    idx[best_dist > atol] = -1
    return idx.astype(np.int64)
