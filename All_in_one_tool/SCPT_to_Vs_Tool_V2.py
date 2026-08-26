# Import packages for use:
import numpy as np
import pandas as pd
import ngl_tools.smt as smt
import pwlf
from itertools import product

pd.set_option('display.max_columns', None)

def interleave(v1, v2):
    vout = np.empty(2 * len(v1), dtype=float)
    vout[0::2] = v1
    vout[1::2] = v2
    return vout 

def gamma_t (depth, fs, qt):

    # Replace invalid values in CPT_fs and CPT_qt with NaN
    fs1 = fs.copy()
    qt1 = qt.copy()
    fs1[fs1 <= 0] = np.nan
    qt1[qt1 <= 0] = np.nan

    # Compute gamma only for valid values
    valid_mask = ~np.isnan(fs1) & ~np.isnan(qt1)

    gamma = np.full_like(depth, np.nan)  # Initialize with NaN

    # Calculate gamma using the formula
    gamma[valid_mask] = 9.81 * (0.27 * np.log10(fs1[valid_mask]/qt1[valid_mask] * 100) + 0.36 * np.log10(qt1[valid_mask] / 101.3) + 1.236)

     # Interpolate NaN values
    gamma = np.interp(depth, depth[valid_mask], gamma[valid_mask])

    return gamma


def get_CPT (CPT_depth, CPT_qt, CPT_fs, dGWT):

    CPT_fs = CPT_fs * 1000 #Convert to kPa
    CPT_qt = CPT_qt * 1000 #Convert to kPa

    pa = 101.3 #KPa,
    gamma = gamma_t(CPT_depth, CPT_fs, CPT_qt)

    CPT_fs[CPT_fs < 0] = 0.01
    CPT_qt[CPT_qt < 0] = 0.01

    # integrate gamma over CPT_depth to get sigmav, then compute sigmavp by subtracting the pore pressure u
    # updated 2026-02-24
    depth_interval = np.diff(CPT_depth, prepend=0)
    sigmav = np.cumsum(gamma * depth_interval)
           
    u = (CPT_depth - dGWT) * 9.81
    u[u<0] = 0
    sigmavp = sigmav - u

    # Compute fz
    fz = sigmavp/pa

    # Calculate Ic, Qtn and their inverse filtered values
    Ic, Qtn, Fr = smt.get_Ic_Qtn_Fr(CPT_qt, CPT_fs, sigmav, sigmavp)
    qt_inv, fs_inv, Ic_inv = smt.cpt_inverse_filter(CPT_qt, CPT_depth, fs=CPT_fs, sigmav=sigmav, sigmavp=sigmavp)
    Ic_inv, Qtn_inv, Fr_inv = smt.get_Ic_Qtn_Fr(qt_inv, fs_inv, sigmav, sigmavp)

    # Ic correction for Qtn
    Qtncs = Qtn_inv.copy()
    kc = np.full(Qtn_inv.shape, 1.0)
    kc = 1 + 5 / (1 + np.exp( - 14 * (Ic_inv - 2.1) ))
    Qtncs = Qtn_inv * kc

    return fz, Ic_inv, Qtncs, CPT_depth, Qtn_inv, Ic_inv, Qtn, Ic, kc

def get_TT (TT_depth, Meas_TT, source = 1.0):
    # Correction of Source Offset Distance
    tt_corrected = TT_depth / np.sqrt (TT_depth**2 + source**2) * Meas_TT

    ztop = TT_depth[:-1]
    zbot = TT_depth[1:]

    DTT = np.diff(tt_corrected)

    return ztop, zbot, DTT

def regression(tt_depth, Meas_TT, n, source = 1.0, search_radius = 2):

    # Correction of Source Offset Distance
    traveltime = tt_depth / np.sqrt (tt_depth**2 + source**2) * Meas_TT

    # Compute half depths from tt_depth
    half_depths = (tt_depth[:-1] + tt_depth[1:]) / 2

    # joint the half depths with tt_depth (without the first and last point) as candidate breakpoints
    candidate_breakpoints = np.sort(np.concatenate((tt_depth[1:-1], half_depths)))

    n_segments = n

    # One segment
    if n_segments == 1:

        piecewise = pwlf.PiecewiseLinFit(tt_depth, traveltime)
        breaks = piecewise.fit(1)
        fitted_values = piecewise.predict(tt_depth)
        fitted_values_for_plot = fitted_values
        new_depths = tt_depth

    else:
        pw = pwlf.PiecewiseLinFit(tt_depth, traveltime)
        optimized_breaks = pw.fit(n_segments)
        internal_breaks = (optimized_breaks[1:-1])

        candidate_lists = []

        for b in internal_breaks:

            # Find the index of the closest candidate breakpoint to the optimized break
            idx = np.argmin(np.abs(candidate_breakpoints - b))
            lb = max(0, idx - search_radius)
            ub = min(len(candidate_breakpoints), idx + search_radius + 1)

            nearby = (candidate_breakpoints[lb : ub])
            candidate_lists.append(nearby)

        # Only consider combinations from the candidate lists
        best_rss = np.inf
        best_breaks = None
        best_piecewise = None

        for combo in product(*candidate_lists):

            internal = np.unique(combo)

            if len(internal) != (n_segments - 1):
                continue

            all_breaks = np.array(sorted([tt_depth.min()] + list(internal) + [tt_depth.max()]))

            # minimum 2 pts/segment
            valid = True

            for i in range(len(all_breaks) - 1):
                count = np.sum((tt_depth >= all_breaks[i]) & (tt_depth <= all_breaks[i + 1]))
                if count < 2:
                    valid = False
                    break
            if not valid:
                continue

            try:
                piecewise_trial = (pwlf.PiecewiseLinFit(tt_depth, traveltime))
                rss = (piecewise_trial.fit_with_breaks(all_breaks))
                if rss < best_rss:
                    best_rss = rss
                    best_breaks = (all_breaks)
                    best_piecewise = (piecewise_trial)
            except:
                continue

        if best_breaks is None:
            raise ValueError("No valid breakpoint combination found")

        breaks = best_breaks
        piecewise = best_piecewise

        # Predictions
        fitted_values = piecewise.predict(tt_depth)
        new_depths = np.array(sorted(set(breaks.tolist() + tt_depth.tolist())))
        fitted_values_for_plot = (piecewise.predict(new_depths))

    # Calculatae RMSE
    residuals = (traveltime - fitted_values)
    rmse = np.sqrt(np.mean(residuals**2))

    # Calculate slope of each segment
    slopes = piecewise.slopes
    ztop = breaks[:-1].copy()
    ztop[0] = 0.01
    zbot = breaks[1:]

    return {
        'breaks': breaks,
        'candidate_breakpoints': candidate_breakpoints,
        'fitted_values': fitted_values,
        'fitted_values_for_plot': fitted_values_for_plot,
        'new_depths': new_depths,
        'rmse': rmse,
        'slopes': slopes,
        'ztop': ztop,
        'zbot': zbot
    }

def regression_full_manual(tt_depth, Meas_TT, breakpoints, source = 1.0):

    # Correction of Source Offset Distance
    traveltime = tt_depth / np.sqrt (tt_depth**2 + source**2) * Meas_TT

    # Validate breakpoints
    breakpoints = sorted(set(breakpoints))  # Ensure sorted and unique
    if breakpoints[0] <= tt_depth.min() or breakpoints[-1] >= tt_depth.max():
        raise ValueError("Breakpoints must be within the range of tt_depth (excluding endpoints)")

    all_breaks = [tt_depth.min()] + breakpoints + [tt_depth.max()]

    # Ensure each segment has at least two data points
    for i in range(len(all_breaks) - 1):
        count = np.sum((tt_depth >= all_breaks[i]) & (tt_depth <= all_breaks[i + 1]))
        if count < 2:
            raise ValueError(f"Segment between {all_breaks[i]} and {all_breaks[i + 1]} has less than 2 data points.")

    # Fit with custom breakpoints
    piecewise = pwlf.PiecewiseLinFit(tt_depth, traveltime)
    rss = piecewise.fit_with_breaks(all_breaks)

    # Generate fitted values
    new_depths = np.array(sorted(set(all_breaks + list(tt_depth))))
    fitted_values = piecewise.predict(tt_depth)
    fitted_values_for_plot = piecewise.predict(new_depths)

    # RMSE
    residuals = traveltime - fitted_values
    rmse = np.sqrt(np.mean(residuals**2))

    # Slopes
    slopes = piecewise.slopes
    ztop = all_breaks[:-1]
    # Replace the first ztop with 0.01
    ztop[0] = 0.01

    zbot = all_breaks[1:]

    return {
        'breaks': all_breaks,
        'fitted_values': fitted_values,
        'fitted_values_for_plot': fitted_values_for_plot,
        'new_depths': new_depths,
        'rmse': rmse,
        'slopes': slopes,
        'ztop': ztop,
        'zbot': zbot
    }


def optimize (tt_depth, Meas_TT):

    if len(tt_depth) < 7: 
        error = np.zeros(len(tt_depth)-1)
        for i in range (1, len(tt_depth)):
            results = regression(tt_depth, Meas_TT, i)
            error[i-1] = results['rmse']

    else: 
        error = np.zeros(6)
        for i in range (1, 7): 
            results = regression(tt_depth, Meas_TT, i)
            error[i-1] = results['rmse']

    # Compute cost function (Updated on 2024-06-07)
    t_avg = (tt_depth.max() - tt_depth.min()) / np.arange(1, len(error)+1)
    tref = 4

    wR = 1
    wT = 1

    # Normalized RMSE Cost Function
    JR = error / error[0]
    # Thickness-Dependent Cost Function 
    JT = 0.2 * (tref / t_avg) ** 3 
    # Combined Cost Function
    cost = wR * JR + wT * JT

    min_cost_index = np.argmin(cost)
    Optimum = min_cost_index + 1

    return Optimum
    

def optimized_regression(tt_depth, Meas_TT, source = 1.0):
    n = optimize(tt_depth, Meas_TT)
    results = regression(tt_depth, Meas_TT, n, source = source)

    velocity = (1/np.array(results['slopes']))*1000

    return velocity, results

def Master (tt_depth, Meas_TT, slope_break_method = 1, breakpoints = None, source = 1.0):

    tt_corrected = tt_depth / np.sqrt (tt_depth**2 + source**2) * Meas_TT

    # Full Automatic
    if slope_break_method == 0: 
        velocity, results = optimized_regression(tt_depth, Meas_TT, source = source)
    # Full Manual
    elif slope_break_method == 1:
        if breakpoints is None:
            raise ValueError("Breakpoints must be provided for manual regression.")
        results = regression_full_manual(tt_depth, Meas_TT, breakpoints, source = source)
        velocity = (1/np.array(results['slopes']))*1000

    top_depth = results['ztop']
    bottom_depth = results['zbot']

    return velocity, tt_corrected, results, top_depth, bottom_depth