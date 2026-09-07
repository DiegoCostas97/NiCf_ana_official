import os
import sys
import numpy as np
import pandas as pd
import array
import ROOT
import cppyy


# =========================================================
# INITIALIZATION (DO THIS EXPLICITLY, NOT ON IMPORT)
# =========================================================

def init_bonsai_environment():
    """
    Load ROOT + BONSAI libraries and initialize geometry.
    Call this ONCE in your main script.
    """

    os.environ["ROOTSYS"]   = "/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/ROOT/ROOT_6.26.14/install"
    os.environ["BONSAIDIR"] = "/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/hk-BONSAI"
    os.environ["WCSIM_BUILD_DIR"] = "/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/WCSim/install/"
    sys.path.append(os.path.join(os.environ["ROOTSYS"], "lib"))
    sys.path.append(os.path.join(os.environ["BONSAIDIR"], "lib"))
    sys.path.append(os.path.join(os.environ["WCSIM_BUILD_DIR"], "lib"))

    geotree = None
    for event in simtree:
        geotree = event.wcsimrootgeom
        break

    bonsai = cppyy.gbl.WCSimBonsai()
    bonsai.Init(geotree)

    return bonsai


# =========================================================
# GEOMETRY
# =========================================================

def get_geo_mapping():
    """
    Load PMT geometry table.
    """

    path = "/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/WCSim/install/geofile_NuPRISMBeamTest_16cShort_mPMT.txt"

    geo = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=5,
        names=["id", "mpmtid", "spmtid", "x", "y", "z", "dx", "dy", "dz", "cyloc"]
    )

    return geo


def build_lookup_table(geo):
    """
    Precompute fast lookup dictionary for geometry.
    """

    return {
        (row.mpmtid, row.spmtid): (row.x, row.y, row.z, row.id)
        for row in geo.itertuples(index=False)
    }


def getxyz(lookup, mpmt_ids, pmt_ids):
    """
    Convert (mpmt, pmt) → (x,y,z,id)
    """

    mpmt_ids = np.asarray(mpmt_ids)
    pmt_ids = np.asarray(pmt_ids) + 1  # NOTE: original indexing convention

    results = [
        lookup.get((m, p), (-999.9, -999.9, -999.9, -999))
        for m, p in zip(mpmt_ids, pmt_ids)
    ]

    if len(results) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    x, y, z, cid = map(np.array, zip(*results))
    return x, y, z, cid


# =========================================================
# BONSAI WRAPPER (PURE FUNCTION)
# =========================================================

def run_BONSAI_candidate(
    bonsai,
    lookup,
    times,
    charges,
    mpmt,
    pmt,
    time_recenter=True
):
    """
    Run BONSAI reconstruction on a single event.

    IMPORTANT:
    - NO physics cuts here
    - NO charge filtering here
    - ONLY interface + conversion + execution
    """

    times = np.asarray(times, dtype=float)
    charges = np.asarray(charges, dtype=float)
    mpmt = np.asarray(mpmt, dtype=int)
    pmt = np.asarray(pmt, dtype=int)

    # -------------------------------------------------
    # BASIC SAFETY CHECKS
    # -------------------------------------------------
    if len(times) < 1 or len(charges) < 1:
        return {
            "success": False,
            "x": np.nan, "y": np.nan, "z": np.nan,
            "nhits": 0
        }

    # -------------------------------------------------
    # GEOMETRY
    # -------------------------------------------------
    x, y, z, cables = getxyz(lookup, mpmt, pmt)

    if len(cables) == 0:
        return {
            "success": False,
            "x": np.nan, "y": np.nan, "z": np.nan,
            "nhits": 0
        }

    # -------------------------------------------------
    # TIME HANDLING (OPTIONAL)
    # -------------------------------------------------
    if time_recenter:
        t0 = np.min(times)
        times = times - t0

    # -------------------------------------------------
    # BONSAI INPUT ARRAYS
    # -------------------------------------------------
    bsVertex = array.array('f', [0.0, 0.0, 0.0])
    bsResult = array.array('f', [0.0] * 6)
    bsGood = array.array('f', [0.0] * 3)

    bsNhit = array.array('i', [len(cables)])
    bsNsel = array.array('i', [0])

    bsT = array.array('f', times.astype(np.float32))
    bsQ = array.array('f', charges.astype(np.float32))
    bsCAB = array.array('i', map(int, cables))

    # -------------------------------------------------
    # RUN BONSAI
    # -------------------------------------------------
    try:
        nhits = bonsai.BonsaiFit(
            bsVertex,
            bsResult,
            bsGood,
            bsNsel,
            bsNhit,
            bsCAB,
            bsT,
            bsQ
        )

    except Exception as e:
        print("BONSAI FAILED:", e)
        return {
            "success": False,
            "x": np.nan, "y": np.nan, "z": np.nan,
            "nhits": len(times)
        }

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    return {
        "success": True,
        "nhits": int(nhits),
        "nhitso": int(len(times)),

        "x": float(bsVertex[0]),
        "y": float(bsVertex[1]),
        "z": float(bsVertex[2]),

        "result": np.array(bsResult, dtype=float),
        "good": np.array(bsGood, dtype=float)
    }
