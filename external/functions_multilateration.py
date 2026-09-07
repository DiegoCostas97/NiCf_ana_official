from collections import defaultdict
import numpy as np
from scipy.optimize import least_squares
import functions_bonsai

def run_multilateration_candidate(
    times, mpmt_ids, pmt_ids,
    *,
    sigma_t=1.0,
    n=1.33,
    c_cm_per_ns=29.9792458,   # cm/ns
    guess=(0., 0., 0., 0.),
    mins=(-300., -300., -300., -300.),
    maxs=(300., 300., 300., 300.),
    drop_invalid_geo=True,
    earliest_per_channel=True,
    early_window_ns=100.0,    #set None to disable extra windowing
    robust_loss="soft_l1",
    f_scale=2.0,
    **kwargs
):
    #convert to numpy arrays
    times = np.asarray(times, dtype=float)
    mpmt_ids = np.asarray(mpmt_ids, dtype=int)
    pmt_ids = np.asarray(pmt_ids, dtype=int)

    #geometry from bonsai
    x_pmt, y_pmt, z_pmt, _ = functions_bonsai.getxyz(functions_bonsai.geo, mpmt_ids, pmt_ids)

    #clean geometry
    if drop_invalid_geo:
        good = (x_pmt > -900) & (y_pmt > -900) & (z_pmt > -900) & np.isfinite(times)
        times, mpmt_ids, pmt_ids = times[good], mpmt_ids[good], pmt_ids[good]
        x_pmt, y_pmt, z_pmt = x_pmt[good], y_pmt[good], z_pmt[good]

    # keep earliest hit per (mPMT, PMT)
    if earliest_per_channel and len(times) > 0:
        #Optional: reject very late hits globally (helps a lot for delayed candidates)
        if early_window_ns is not None:
            tmin = float(np.min(times))
            wmask = times <= (tmin + float(early_window_ns))
            times, mpmt_ids, pmt_ids = times[wmask], mpmt_ids[wmask], pmt_ids[wmask]
            x_pmt, y_pmt, z_pmt = x_pmt[wmask], y_pmt[wmask], z_pmt[wmask]

        #find earliest time indez for each channel
        #Approach: sort by time, then take the first occurence of each (mPMT, PMT)
        order = np.argsort(times)
        times_s, mpmt_s, pmt_s = times[order], mpmt_ids[order], pmt_ids[order]
        x_s, y_s, z_s = x_pmt[order], y_pmt[order], z_pmt[order]

        seen = set()
        keep_idx = []
        for i, key in enumerate(zip(mpmt_s, pmt_s)):
            if key not in seen:
                seen.add(key)
                keep_idx.append(i)
        
        idx = np.array(keep_idx)
        times, x_pmt, y_pmt, z_pmt = times_s[idx], x_s[idx], y_s[idx], z_s[idx]

    #Mínimo de hits para un fit de 4 parámetros (x,y,z,t0)
    if len(times) < 6:
        return {"x": np.nan, "y": np.nan, "z": np.nan, "eps": np.nan,
                "success": False, "n_hits_used": int(len(times)), "result": None}

    pmt_locs = np.column_stack([x_pmt, y_pmt, z_pmt]).astype(float)
    vc = float(c_cm_per_ns) / float(n)
    sigma_ts = np.full(times.shape, float(sigma_t), dtype=float)

    #shift time origin
    t0_min = float(np.min(times))
    times_rel = times - t0_min

    # estimación inicial inteligente para el tiempo de emisión (eps) - better eps guess
    loc0 = np.array([0., 0., 0.])
    tof0 = np.linalg.norm(pmt_locs - loc0, axis=1) / vc
    eps_guess = float(np.median(times_rel - tof0))
    x0 = np.array([0., 0., 0., eps_guess], dtype=float)

    #función de residuos (pulls)
    def rho(pars):
        loc, eps = pars[0:3], pars[3]
        tofs = np.linalg.norm(pmt_locs - loc, axis=1) / vc
        return (times_rel - eps - tofs) / sigma_ts

    # Jacobiana (acelera la convergencia)
    def jac(pars):
        loc = pars[0:3]
        light_vecs = pmt_locs - loc
        dists = np.linalg.norm(light_vecs, axis=1)
        dists = np.where(dists == 0, 1e-12, dists)
        jac_xyz = light_vecs / dists.reshape(-1, 1) / vc / sigma_ts.reshape(-1, 1)
        jac_eps = -1.0 / sigma_ts
        return np.column_stack([jac_xyz, jac_eps])

    #fit
    result = least_squares(
        rho, x0, jac,
        bounds=(np.array(mins), np.array(maxs)),
        loss=robust_loss,
        f_scale=float(f_scale),
        **kwargs
    )

    if not result.success:
        return {"x": np.nan, "y": np.nan, "z": np.nan, "eps": np.nan,
                "success": False, "n_hits_used": int(len(times)), "result": result}

    #statistics 
    pulls = rho(result.x)                  #1D array of pulls
    chi2 = float(np.sum(pulls**2))
    ndof = int(len(pulls) - 4)
    chi2_ndof = chi2 / ndof if ndof > 0 else np.inf

    x, y, z, eps_rel = result.x
    
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "eps": float(eps_rel + t0_min), #abs time
        "success": True,
        "n_hits_used": int(len(pulls)),
        "chi2": chi2,
        "ndof": ndof,
        "chi2_ndof": chi2_ndof,
        "pulls": pulls, # PMT per PMT
        "result": result
    }