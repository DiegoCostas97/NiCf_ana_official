"""
NiCf Analysis Pipeline — Core Functions
========================================
Shared library imported by every stage: reading WCTE ROOT data and WCSim MC,
applying the time-of-flight correction, running the greedy nHits trigger,
building candidate DataFrames, and computing per-candidate observables, the
charge calibration and the purity.

Channel convention
------------------
Throughout the pipeline a readout channel is identified by

    pmt_id = mpmt_slot * 19 + pmt_position          (both 0-indexed)

with 106 slots and 19 PMTs each, i.e. 2014 channels. The WCSim geofile uses a
different numbering (tube_no, and a 1-indexed PMT position), which
load_wcsim_tube_mapping() converts.

Coordinate frames
-----------------
Two frames appear and are never mixed inside this file:

  * WCTE, millimetres — the frame of wcte_v11_*.json. Used for the DATA ToF
    correction (load_pmt_positions / build_tof_map).
  * WCSim, centimetres — the frame of the MC hit positions in the .npz. Used for
    the MC ToF correction (apply_tof_correction_mc).

They differ by a shift of about 425 mm along the vertical axis; see
pipeline_utils.wcsim_cm_to_wcte_mm_offset(). Any code that combines the two
(the vertex-based angular response) must convert explicitly.
"""

import json
import numpy as np
import awkward as ak
import pandas as pd
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
# 1. ToF Correction
# ═══════════════════════════════════════════════════════════════════

def load_pmt_positions(json_path):
    """
    Load PMT positions from the WCTE geometry JSON.
    Returns dict: pmt_id → np.array([x, y, z]) in mm.
    """
    with open(json_path, "r") as f:
        geo_data = json.load(f)

    mpmt_data = geo_data.get("mpmts", {})
    pmt_positions = {}

    for mpmt_idx in sorted(mpmt_data.keys(), key=int):
        mpmt = mpmt_data[mpmt_idx]
        pmts = mpmt.get("pmts", {})
        for pmt_idx in sorted(pmts.keys(), key=int):
            pmt = pmts[pmt_idx]
            location = pmt["placement"]["location"]
            pmt_id = int(mpmt_idx) * 19 + int(pmt_idx)
            pmt_positions[pmt_id] = np.array(location, dtype=np.float64)

    return pmt_positions


def build_tof_map(pmt_positions, source_position, n_water=1.33):
    """
    Precompute ToF [ns] from a source position to every PMT.

    Parameters
    ----------
    pmt_positions    : dict {pmt_id: np.array([x,y,z])} in mm
    source_position  : np.array([x,y,z]) in mm
    n_water          : refractive index of water

    Returns
    -------
    tof_map : dict {pmt_id: tof_ns}
    """
    c_water_mm_ns = (3e8 / n_water) * 1e-6  # mm/ns
    source = np.array(source_position, dtype=np.float64)

    tof_map = {}
    for pmt_id, pos in pmt_positions.items():
        dist = np.linalg.norm(pos - source)
        tof_map[pmt_id] = dist / c_water_mm_ns

    return tof_map


def read_mpmt_offsets(json_path):
    """
    LEGACY, unused by the current pipeline.

    Reads a precomputed mPMT ToF offset map keyed by 'slot*100 + pos'. It was
    the original way of getting the source-to-PMT flight times, before
    build_tof_map() started computing them directly from the geometry JSON.
    Kept only so that old notebooks still import cleanly.

    Returns dict: (slot, pos) → tof_ns.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    mpmt_map = {}
    for key, val in raw.items():
        k = int(key)
        slot = k // 100
        pos  = k % 100
        mpmt_map[(slot, pos)] = val

    return mpmt_map


# ═══════════════════════════════════════════════════════════════════
# 2. Data Reading
# ═══════════════════════════════════════════════════════════════════

def read_run(run_number, run_files, part_files, n_parts,
             tof_map=None, max_slot=106, max_pos=19):
    """
    Read a run from ROOT files and return awkward arrays of
    hit-level variables, optionally with ToF correction applied.

    Parameters
    ----------
    run_number : int
    run_files  : dict {part_idx: filepath}
    part_files : list of part indices
    n_parts    : int, how many part files to read
    tof_map    : dict {pmt_id: tof_ns} or None (no correction)

    Returns
    -------
    dict with keys:
        'hit_times', 'hit_charges', 'hit_slot_ids',
        'hit_position_ids', 'hit_card_ids', 'hit_channel_ids'
    All are awkward arrays of shape [n_events, var].
    """
    import uproot

    acc = {
        'hit_times':        [],
        'hit_charges':      [],
        'hit_slot_ids':     [],
        'hit_position_ids': [],
        'hit_card_ids':     [],
        'hit_channel_ids':  [],
    }

    # Build a dense (slot, position) lookup table for the ToF, so the correction
    # can be applied to the flattened awkward array in one vectorised gather.
    # Channels absent from tof_map keep a correction of 0 ns; with the standard
    # geometry JSON all 2014 channels are present, so this never happens.
    if tof_map is not None:
        lookup = np.zeros((max_slot, max_pos))
        for pmt_id, tof in tof_map.items():
            slot = pmt_id // 19
            pos  = pmt_id % 19
            if slot < max_slot and pos < max_pos:
                lookup[slot, pos] = tof

    for part in part_files[:n_parts]:
        print(f"  Reading WCTE_offline_R{run_number}S0P{part}")
        tree = uproot.open(run_files[part] + ":WCTEReadoutWindows")

        card_ids     = ak.values_astype(tree["hit_mpmt_card_ids"].array(),        np.int16)
        channel_ids  = ak.values_astype(tree["hit_pmt_channel_ids"].array(),      np.int8)
        times_calib  = ak.values_astype(tree["hit_pmt_calibrated_times"].array(), np.float64)
        charges      = ak.values_astype(tree["hit_pmt_charges"].array(),          np.float64)
        slot_ids     = ak.values_astype(tree["hit_mpmt_slot_ids"].array(),        np.int16)
        position_ids = ak.values_astype(tree["hit_pmt_position_ids"].array(),     np.int16)
        # has_tc       = ak.values_astype(tree["hit_pmt_has_time_constant"].array(), bool)

        # Quality mask. `charges < 1e4` removes saturated/overflow ADC values
        # and `card_ids < 120` drops the non-PMT readout cards. The commented
        # `has_tc` term would additionally require a valid time constant for the
        # channel; it is left out because it is not filled in all productions.
        mask = (charges < 1e4) & (card_ids < 120) #& (has_tc != 0)
        times_calib  = times_calib [mask]
        charges      = charges     [mask]
        card_ids     = card_ids    [mask]
        channel_ids  = channel_ids [mask]
        slot_ids     = slot_ids    [mask]
        position_ids = position_ids[mask]

        # Apply ToF correction
        if tof_map is not None:
            flat_slots = ak.ravel(slot_ids)
            flat_pos   = ak.ravel(position_ids)
            flat_corr  = lookup[flat_slots, flat_pos]
            corrections = ak.unflatten(flat_corr, ak.num(card_ids))
            times_calib = times_calib - corrections

        # Time-order the hits inside each readout window. The trigger assumes
        # sorted times, and after the ToF correction the original acquisition
        # order is no longer monotonic.
        order = ak.argsort(times_calib)
        acc['hit_times']       .append(times_calib [order])
        acc['hit_charges']     .append(charges     [order])
        acc['hit_slot_ids']    .append(slot_ids    [order])
        acc['hit_position_ids'].append(position_ids[order])
        acc['hit_card_ids']    .append(card_ids    [order])
        acc['hit_channel_ids'] .append(channel_ids [order])

    return {k: ak.concatenate(v) for k, v in acc.items()}


# ═══════════════════════════════════════════════════════════════════
# 3. Greedy nHits Trigger
# ═══════════════════════════════════════════════════════════════════

def _process_event(ht, w, thresh_min, thresh_max, jump):
    """
    Greedy nHits trigger on a single readout window.

    `right[i]` is, for every hit i, the index one past the last hit within
    [t_i, t_i + w); `counts[i]` is therefore the occupancy of the window that
    opens on hit i. A candidate is seeded at the first hit whose window holds at
    least `thresh_min` hits, and is then extended greedily: as long as new hits
    fall within w of the last hit collected, the cluster keeps growing. That is
    what lets a candidate be longer than w and is why the tRMS of a real capture
    is not bounded by w/sqrt(12).

    `jump` imposes a dead time after a candidate; with jump = 0 the next
    candidate may start at the first hit after the previous cluster.

    Returns three parallel lists: hit indices, hit times and the seed time of
    each candidate. Indices refer to the TIME-SORTED array.
    """
    if len(ht) == 0:
        return [], [], []

    ht = np.sort(ht)
    n  = len(ht)

    ends   = ht + w
    right  = np.searchsorted(ht, ends, side="left")
    counts = right - np.arange(n)

    candidate_indices = []
    candidate_times   = []
    trigger_seeds     = []

    last_cluster_end = -np.inf
    i = 0

    while i < n:
        if ht[i] < last_cluster_end + jump:
            i += 1; continue
        if counts[i] < thresh_min:
            i += 1; continue
        if thresh_max is not None and counts[i] > thresh_max:
            i += 1; continue

        seed             = ht[i]
        window_start_idx = i
        collected_end    = right[i]

        while collected_end < n:
            t_last   = ht[collected_end - 1]
            next_end = np.searchsorted(ht, t_last + w, side="left")
            if next_end > collected_end:
                collected_end = next_end
            else:
                break

        indices_roi   = np.arange(window_start_idx, collected_end)
        hit_times_roi = ht[indices_roi]

        candidate_indices.append(indices_roi)
        candidate_times.append(hit_times_roi)
        trigger_seeds.append(seed)

        last_cluster_end = hit_times_roi[-1]
        i = collected_end

    return candidate_indices, candidate_times, trigger_seeds


def nHits_greedy(hit_times, w, thresh_min, thresh_max=None,
                 jump=0, progress_bar=True):
    """
    Run greedy nHits trigger on all events.

    Parameters
    ----------
    hit_times  : awkward array [n_events, var] of hit times per event
    w          : sliding window width [ns]
    thresh_min : minimum hits to trigger
    thresh_max : maximum hits in seed window (None = no limit)
    jump       : deadtime [ns] between candidates

    Returns
    -------
    all_indices : dict {ev: [array_of_indices, ...]}
    all_times   : dict {ev: [array_of_times, ...]}
    all_seeds   : dict {ev: [seed_time, ...]}
    """
    nevents = len(hit_times)

    all_indices = {}
    all_times   = {}
    all_seeds   = {}

    for ev in tqdm(range(nevents), total=nevents, disable=not progress_bar):
        ht = ak.to_numpy(hit_times[ev])

        if len(ht) == 0:
            all_indices[ev] = []
            all_times[ev]   = []
            all_seeds[ev]   = []
            continue

        idxs, times, seeds = _process_event(ht, w, thresh_min, thresh_max, jump)
        all_indices[ev] = idxs
        all_times[ev]   = times
        all_seeds[ev]   = seeds

    return all_indices, all_times, all_seeds


# ═══════════════════════════════════════════════════════════════════
# 4. DataFrame Building
# ═══════════════════════════════════════════════════════════════════

def build_candidates_dataframe(trigger_indices, trigger_times,
                               run_data, label="sig"):
    """
    Build a hit-level DataFrame from trigger output.

    Parameters
    ----------
    trigger_indices : dict {ev: [array_of_indices, ...]}
    trigger_times   : dict {ev: [array_of_times, ...]}
    run_data        : dict from read_run()
    label           : 'sig' or 'bkg'

    Returns
    -------
    pd.DataFrame with columns:
        event_id, candidate_id, hit_pmt_calibrated_times,
        hit_pmt_charges, hit_mpmt_slot_ids, hit_pmt_position_ids, pmt_id
    """
    rows = []
    global_cand_id = 0

    for ev, candidates_idx in tqdm(trigger_indices.items(),
                                    total=len(trigger_indices)):
        ev_times = ak.to_numpy(run_data['hit_times'][ev])
        ev_charges = ak.to_numpy(run_data['hit_charges'][ev])
        ev_slots = ak.to_numpy(run_data['hit_slot_ids'][ev])
        ev_pos = ak.to_numpy(run_data['hit_position_ids'][ev])

        for hit_indices in candidates_idx:
            for idx in hit_indices:
                rows.append({
                    "event_id":                 ev,
                    "candidate_id":             global_cand_id,
                    "hit_pmt_calibrated_times": ev_times[idx],
                    "hit_pmt_charges":          ev_charges[idx],
                    "hit_mpmt_slot_ids":        ev_slots[idx],
                    "hit_pmt_position_ids":     ev_pos[idx],
                })
            global_cand_id += 1

    df = pd.DataFrame(rows)
    df["pmt_id"] = df["hit_mpmt_slot_ids"] * 19 + df["hit_pmt_position_ids"]
    df["run"] = label
    return df


# ═══════════════════════════════════════════════════════════════════
# 5. Candidate-Level Observables
# ═══════════════════════════════════════════════════════════════════

def add_candidate_observables(df):
    """
    Add per-candidate observables propagated to every hit row:
        trms, nhits, tc, duration
    """
    # transform() broadcasts the per-candidate value back onto every hit row of
    # that candidate, so the observables can be used as a row-wise mask.
    # Note std() is the sample standard deviation (ddof=1), so a candidate with
    # a single hit gets NaN and is dropped by any tRMS cut.
    grp = df.groupby("candidate_id")

    df["trms"] = grp["hit_pmt_calibrated_times"].transform("std")
    df["nhits"] = grp["hit_pmt_calibrated_times"].transform("count")
    df["tc"] = grp["hit_pmt_calibrated_times"].transform(
        lambda t: (t - t.min()).mean()
    )
    df["duration"] = grp["hit_pmt_calibrated_times"].transform(
        lambda t: t.max() - t.min()
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# 6. MC Data Reading & Trigger
# ═══════════════════════════════════════════════════════════════════

def load_wcsim_tube_mapping(geo_path):
    """
    Load the WCSim tube_no → (mPMT slot, PMT position) mapping from the geofile.
    The position is converted to the 0-indexed convention used everywhere else.

    The geofile describes 1843 channels while the WCTE geometry JSON describes
    2014: nine mPMTs present in the real detector are absent from this WCSim
    geometry. Their channels have no MC counterpart, which is why the relative
    QE is defined for about 1550 channels rather than all 2014.
    """
    data = np.loadtxt(geo_path, skiprows=5, usecols=(0, 1, 2), dtype=int)
    return {row[0]: (row[1], row[2] - 1) for row in data}


def read_mc_truehits(npz_path, truehits_to_df_func):
    """
    Read MC true hits using the provided converter function.

    Parameters
    ----------
    npz_path          : str, path to WCSim .npz file
    truehits_to_df_func : callable, e.g. truehits_info_to_df

    Returns
    -------
    df_trueHits : pd.DataFrame with columns event_id, true_hit_parent,
                  true_hit_pmt, true_hit_time, hit_x, hit_y, hit_z, ...
    """
    return truehits_to_df_func(npz_path).dropna()


def apply_tof_correction_mc(df_hits, source_pos, n_water=1.33):
    """
    Apply the ToF correction to MC true hits, using the PMT hit positions
    stored in the .npz (hit_x, hit_y, hit_z) and the source position.

    Everything here is in the WCSim frame and in centimetres, which is why the
    source position must be given as --source-pos-cm and NOT as the WCTE mm
    position used for the data.

    Adds column 'true_hit_time_tof_corrected' to df_hits (in-place).

    Parameters
    ----------
    df_hits    : DataFrame with hit_x, hit_y, hit_z, true_hit_time
    source_pos : [x, y, z] in cm (MC coordinates)
    n_water    : refractive index
    """
    c_water = 3e8 / n_water  # m/s
    c_water_cm_ns = c_water * 1e-7  # cm/ns

    sx, sy, sz = source_pos
    dx = df_hits["hit_x"].values.astype(np.float64) - sx
    dy = df_hits["hit_y"].values.astype(np.float64) - sy
    dz = df_hits["hit_z"].values.astype(np.float64) - sz
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    tof = dist / c_water_cm_ns
    df_hits["true_hit_time_tof_corrected"] = df_hits["true_hit_time"].values - tof
    return df_hits


def run_mc_trigger(df_hits, w=20, thresh_min=2, thresh_max=None, jump=0,
                   time_col="true_hit_time_tof_corrected",
                   greedy=True, progress_bar=True):
    """
    Run the nHits trigger on MC true hits DataFrame.
 
    Parameters
    ----------
    df_hits  : DataFrame with event_id, true_hit_pmt, true_hit_parent, and time_col
    w, thresh_min, thresh_max, jump : trigger parameters
    time_col : which time column to use for triggering
    greedy   : if True (default), extend each cluster forward window-by-window
               while hits keep falling within w of the last hit. If False, use
               a FIXED window of width w (no forward extension): the cluster is
               exactly the hits in [seed, seed + w). Fixed windows bound the
               cluster width to w, so tRMS cannot exceed ~w/sqrt(12).
 
    Returns
    -------
    df_mc_cands : candidate-level DataFrame with nhits, trms, all_pmts, all_times, etc.
    """
    all_candidates = []
    global_cand_id = 0
 
    grouped = df_hits.groupby("event_id")
    for event_id, event_df in tqdm(grouped, total=len(grouped),
                                    disable=not progress_bar):
        ht   = event_df[time_col].values.astype(np.float64)
        hpmt = event_df["true_hit_pmt"].values
        hpar = event_df["true_hit_parent"].values
 
        if len(ht) == 0:
            continue
 
        # Sort
        sort_idx = np.argsort(ht)
        ht_s   = ht[sort_idx]
        hpmt_s = hpmt[sort_idx]
        hpar_s = hpar[sort_idx]
        n      = len(ht_s)
 
        ends   = ht_s + w
        right  = np.searchsorted(ht_s, ends, side="left")
        counts = right - np.arange(n)
 
        last_cluster_end = -np.inf
        i = 0
 
        while i < n:
            if ht_s[i] < last_cluster_end + jump:
                i += 1; continue
            if counts[i] < thresh_min:
                i += 1; continue
            if thresh_max is not None and counts[i] > thresh_max:
                i += 1; continue
 
            collected_end = right[i]
            if greedy:
                while collected_end < n:
                    t_last = ht_s[collected_end - 1]
                    next_end = np.searchsorted(ht_s, t_last + w, side="left")
                    if next_end > collected_end:
                        collected_end = next_end
                    else:
                        break
 
            times_roi = ht_s[i:collected_end]
            pmts_roi  = hpmt_s[i:collected_end]
 
            all_candidates.append({
                "event_id":     event_id,
                "candidate_id": global_cand_id,
                "nhits":        collected_end - i,
                "trms":         np.std(times_roi),
                "tc":           (times_roi - times_roi[0]).mean(),
                "duration":     times_roi[-1] - times_roi[0],
                "seed_time":    ht_s[i],
                "all_pmts":     list(np.unique(pmts_roi)),
                "all_times":    list(times_roi),
            })
 
            last_cluster_end = times_roi[-1]
            i = collected_end
            global_cand_id += 1
 
    return pd.DataFrame(all_candidates)


def mc_cands_to_pmt_id(df_mc_cands, tube_mapping):
    """
    Expand MC candidates into one row per (candidate, channel), converting
    WCSim tube numbers to pmt_id = slot*19 + pos.

    NOTE — one channel counted once per candidate.
    `all_pmts` holds the UNIQUE tube numbers of the candidate, so a tube that
    collected several photoelectrons inside the same cluster appears once. This
    matches the data, where the front-end merges photoelectrons arriving within
    a few ns into a single hit (the measured unique-PMT/hit ratio inside a
    candidate is 1.000). Counting every MC photoelectron instead would inflate
    the MC rate on the brightest channels and bias the relative QE.

    Beware that run_angular_vertex.py does NOT unique-ify: it rebuilds per-hit
    rows from `all_times`, so a doubly-hit tube contributes twice there. The two
    stages therefore treat MC multiplicity slightly differently; the difference
    is small but it is a known inconsistency.

    Returns a hit-level DataFrame with candidate_id and pmt_id.
    """
    rows = []
    for _, cand in df_mc_cands.iterrows():
        for tube_no in cand["all_pmts"]:
            # WCSim tube numbers are 0-based in the .npz and 1-based in the
            # geofile that produced `tube_mapping`.
            tube_key = int(tube_no) + 1
            if tube_key in tube_mapping:
                slot, pos = tube_mapping[tube_key]
                rows.append({
                    "candidate_id": cand["candidate_id"],
                    "pmt_id":       slot * 19 + pos,
                })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# 7. Charge Calibration
# ═══════════════════════════════════════════════════════════════════

def compute_charge_calibration(df, n_pmts=2014):
    """
    Compute per-PMT charge calibration factors.
    Factor = global_mean_charge / mean_charge_per_pmt.
    PMTs with no hits get factor = 1.0.

    `global_mean` is the unweighted mean of the per-PMT means, so every channel
    carries the same weight regardless of how many hits it collected. That is
    the intended behaviour here: the factors are meant to equalise channels, not
    to reproduce the overall charge scale.

    Returns
    -------
    factors : np.array of length n_pmts
    global_mean : float
    """
    mean_per_pmt = df.groupby("pmt_id")["hit_pmt_charges"].mean()

    global_mean = mean_per_pmt.mean()

    factors = np.ones(n_pmts)
    for pmt_id, mean_q in mean_per_pmt.items():
        if 0 <= pmt_id < n_pmts and mean_q > 0:
            factors[int(pmt_id)] = global_mean / mean_q

    return factors, global_mean


def apply_charge_calibration(df, factors):
    """
    Apply per-PMT charge calibration to a DataFrame.
    Adds column 'hit_pmt_charges_calibrated'.
    """
    # Vectorised gather: pad the factor array so that any out-of-range pmt_id
    # picks up a factor of 1.0 instead of raising.
    pid = df["pmt_id"].values.astype(np.int64)
    safe = np.ones(len(factors) + 1)
    safe[:len(factors)] = factors
    pid = np.where((pid >= 0) & (pid < len(factors)), pid, len(factors))
    df["hit_pmt_charges_calibrated"] = df["hit_pmt_charges"].values * safe[pid]
    return df


# ═══════════════════════════════════════════════════════════════════
# 8. Purity Calculation
# ═══════════════════════════════════════════════════════════════════

def compute_purity(sig_times, bkg_times, n_events_sig, n_events_bkg,
                   nhits_cut=None, trms_cut=None, tc_cut=None):
    """
    Compute purity, signal efficiency and background rejection.

    Parameters
    ----------
    sig_times, bkg_times : dict {ev: [array_of_times, ...]} from trigger
    n_events_sig, n_events_bkg : int, total readout windows per run
    nhits_cut : int or None — keep candidates with nhits > nhits_cut
    trms_cut  : float or None — keep candidates with tRMS < trms_cut
    tc_cut    : float or None — keep candidates with tc < tc_cut

    Returns
    -------
    dict with purity, sig_eff, bkg_rej, mean_sig, mean_bkg
    """
    def count_per_event(times_dict):
        counts = []
        for evCandidates in times_dict.values():
            n = 0
            for candidate in evCandidates:
                t = np.array(candidate)
                if nhits_cut is not None and len(t) <= nhits_cut:
                    continue
                if trms_cut is not None and np.std(t) >= trms_cut:
                    continue
                if tc_cut is not None and (t - t.min()).mean() >= tc_cut:
                    continue
                n += 1
            counts.append(n)
        return counts

    counts_sig = count_per_event(sig_times)
    counts_bkg = count_per_event(bkg_times)

    mean_sig = np.mean(counts_sig)
    mean_bkg = np.mean(counts_bkg)

    purity = (mean_sig - mean_bkg) / mean_sig * 100 if mean_sig > 0 else 0

    # Baseline (no cuts) for efficiency/rejection
    baseline_sig = np.mean([len(c) for c in sig_times.values()])
    baseline_bkg = np.mean([len(c) for c in bkg_times.values()])

    sig_eff = mean_sig / baseline_sig * 100 if baseline_sig > 0 else 0
    bkg_rej = (1 - mean_bkg / baseline_bkg) * 100 if baseline_bkg > 0 else 0

    return {
        "purity":   purity,
        "sig_eff":  sig_eff,
        "bkg_rej":  bkg_rej,
        "mean_sig": mean_sig,
        "mean_bkg": mean_bkg,
    }
