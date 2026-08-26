# Import packages for use:
import pwlf
import numpy as np
import pandas as pd
from itertools import combinations


def regression (tt_depth, Meas_TT, n, source = 1.0):

    # Correction of Source Offset Distance
    traveltime = tt_depth / np.sqrt (tt_depth**2 + source**2) * Meas_TT

    # Compute half depths from tt_depth
    half_depths = (tt_depth[:-1] + tt_depth[1:]) / 2

    # joint the half depths with tt_depth (without the first and last point) as candidate breakpoints
    candidate_breakpoints = np.sort(np.concatenate((tt_depth[1:-1], half_depths)))

    n_segments = n

    if n_segments == 1:
        piecewise = pwlf.PiecewiseLinFit(tt_depth, traveltime)
        breaks = piecewise.fit(n_segments)
        fitted_values = piecewise.predict(tt_depth)
        fitted_values_for_plot = fitted_values
        new_depths = tt_depth
    elif n_segments - 1 > len(half_depths):
        raise ValueError("Not enough candidate breakpoints for the number of segments")
    else: 
        best_rss = np.inf
        piecewise = None
        # Generate sorted combinations of candidate breakpoints
        for bp_combo in combinations(sorted(set(candidate_breakpoints)), n_segments - 1):
            # print (f"Trying breakpoints: {bp_combo}")
            try:
                all_breaks = sorted([tt_depth.min()] + list(bp_combo) + [tt_depth.max()])

                # Ensure each segment has at least two points
                for i in range(len(all_breaks) - 1):
                    count = np.sum((tt_depth >= all_breaks[i]) & (tt_depth <= all_breaks[i + 1]))
                    if count < 2:
                        break  # Invalid segment, skip this combo
                else:
                    # Only fits if all segments passed the test
                    piecewise_trial = pwlf.PiecewiseLinFit(tt_depth, traveltime)
                    rss = piecewise_trial.fit_with_breaks(all_breaks)
                    if rss < best_rss:
                        best_rss = rss
                        piecewise = piecewise_trial
                        breaks = all_breaks
            except Exception as e:
                print(f"Error fitting with breaks {bp_combo}: {e}")

        if piecewise is None or 'breaks' not in locals():
            raise ValueError(f"No valid breakpoint combination found for the data with n={n_segments}")

        # Combine breaks and tt_depth, then sort them
        new_depths = np.array(sorted(set(breaks + list(tt_depth))))

        # Generate the fitted values
        fitted_values = piecewise.predict(tt_depth)
        fitted_values_for_plot = piecewise.predict(new_depths)
    
    # Calculate RMSE
    residuals = traveltime - fitted_values
    rmse = np.sqrt(np.mean(residuals**2))

    # Calculate slopes of each segment
    slopes = piecewise.slopes
    ztop = breaks[:-1]
    # Replace the first ztop with 0.01
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
            # start_time = time.time()
            results = regression(tt_depth, Meas_TT, i)
            error[i-1] = results['rmse']
            # end_time = time.time()
            # print(f"Number of segment {i}, RMSE: {error[i-1]:.4f}, Time taken: {end_time - start_time:.2f} seconds")

    else: 
        error = np.zeros(6)

        for i in range (1, 7): 
            # start_time = time.time()
            results = regression(tt_depth, Meas_TT, i)
            error[i-1] = results['rmse']
            # end_time = time.time()
            # print(f"Number of segment {i}, RMSE: {error[i-1]:.4f}, Time taken: {end_time - start_time:.2f} seconds")

    # Compute the % change of error between each segment
    error_change = np.zeros(len(error)-1)
    for i in range (1,len(error_change)):
        error_change[i-1] = abs((error[i] - error[i-1]))/error[0] * 100
        if error[i-1] == 0:
            error_change[i-1] = 0

    # Get the number of segment when change of error is less than 10% 
    for i in range (1,len(error_change)+1):
        if error[i] < 0.5:
            if error_change[i-1] < 10 or error[i] <= 0.1:
                break
    Optimum = i+1
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

def interleave(v1, v2):
    vout = np.empty(2 * len(v1), dtype=float)
    vout[0::2] = v1
    vout[1::2] = v2
    return vout 