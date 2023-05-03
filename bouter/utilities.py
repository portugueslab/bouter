import math
from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd
from numba import jit
from scipy import linalg, signal


def merge_bouts(bouts, min_dist):
    bouts = list(bouts)
    i = 1

    while i < len(bouts) - 1:
        current_dist = bouts[i + 1][0] - bouts[i][1]
        if current_dist < min_dist:
            bout_after_to_merge = bouts.pop(i + 1)
            bouts[i][1] = bout_after_to_merge[1]
        else:
            i += 1

    return np.array(bouts)


@jit(nopython=True)
def extract_segments_above_threshold(
    trace,
    threshold=0.1,
    min_length=20,
    min_between=25,
    break_segment_on_nan=True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract periods from a trace where it's value is above threshold.
    The segments also can have a minimal length.
    Used for extracting bouts from vigor or velocity

    :param trace: vigor or velocity
    :param threshold: minimal value to be considered moving
    :param min_length: minimal number of samples of continuous movement
    :param min_between: minimal number of samples between two crossings
    :param break_segment_on_nan: if a NaN is encountered, it breaks the segment
    :return:
        segments: start and end indices of the segments above threshold
        connected: for each segment, whether it is connected to the previous.
        Segments are considered connected if there were no NaN values
        in the trace.
    """

    segments = []
    in_segment = False
    start = 0
    connected = []
    continuity = False

    # we start at the first possible time to detect the threshold crossing
    # (because the pad_before period has to be always included)
    i = 1
    i_last_segment_ended = 0
    while i < trace.shape[0] - min_between:

        # 3 cases where the state can change
        # we encountered a NaN (breaks continuity)
        if np.isnan(trace[i]):
            continuity = False
            if in_segment and break_segment_on_nan:
                in_segment = False
        # segment has ended and a positive threshold crossing has been found
        elif (
            i > i_last_segment_ended
            and trace[i - 1] < threshold < trace[i]
            and not in_segment
        ):
            in_segment = True
            start = i
        # a negative threshold crossing has been found while
        # we are inside a segment:
        elif trace[i - 1] > threshold > trace[i] and in_segment:
            in_segment = False
            if i - start > min_length:
                segments.append((start, i))
                i_last_segment_ended = i + min_between
                if continuity:
                    connected.append(True)
                else:
                    connected.append(False)
                continuity = True

        # in all other cases the state cannot change

        i += 1

    return np.array(segments), np.array(connected)

@jit(nopython=True)
def get_score_trace(trace_pad, kernel, bias=-0.2839):
    """
    Used to calculate the correlation score between the squared velocity
    trace and the kernel for bout detection.
    :param trace_pad: squared velocity (optionally) padded
    :param kernel: kernel used for bout detection
    :return:
        array of correlation scores that matches with the trace by index.
    """
    
    # For numerical stability.
    eta = 0.0000000001
    
    corr = np.empty((len(trace_pad) - len(kernel)))
    for i in range(len(trace_pad) - len(kernel)):
        current_values = trace_pad[i:i+len(kernel)]
        relative_values = current_values / (np.max(current_values) + eta)
        
        # Simplified convolution, works faster and with Numba.
        conv = np.sum(relative_values * np.flip(kernel))
        
        corr[i] = 1 / (1 + np.exp(conv + bias))
        
    # Align the correlation score trace with the actual bouts.
    # Found by trial and error - so can be changed as needed.
    corr = corr[:-int(kernel.shape[0]/5)]
    corr = np.concatenate((np.zeros(int(kernel.shape[0]/5)), corr))
        
    return (corr - 1) * -1


def calc_bout_score(trace, kernel=None, bias=-0.2839, pad_len=None):
    """ 
    Wrapper for the get_score_trace function.
    :param trace: the trace to detect bouts in.
    :param kernel: kernel used for bout detection.
    :param bias: the bias to shift the output
    :param pad_len: the length of the padding on each side.
    :return:
        array of correlation scores that matches with the trace by index.
    """

    if kernel is None:
        # Kernel trained using a NN.
        kernel = np.array([
          -0.5633, -0.5888, -0.5097, -0.3899, -0.4896, -0.4326, -0.5017,
          -0.4555, -0.4709, -0.4580, -0.4563, -0.4510, -0.3165, -0.3896,
          -0.2522, -0.2982, -0.2503, -0.0670,  0.1450,  0.2824,  0.2420,
           0.3270,  0.2569,  0.3174,  0.3718,  0.2896,  0.3356,  0.3905,
           0.2844,  0.3674,  0.3158,  0.3549,  0.2847,  0.4782,  0.4236,
           0.4416,  0.4049,  0.3622,  0.3755,  0.2569,  0.2525,  0.2626,
           0.2875,  0.1809,  0.1314,  0.1213,  0.1023, -0.0113,  0.1013,
           0.0844, -0.0785, -0.0316, -0.0584, -0.1613, -0.1835, -0.1843,
          -0.1421, -0.1407, -0.0838, -0.1655, -0.2004, -0.0968, -0.1559,
          -0.1564, -0.1867, -0.1494, -0.1192, -0.2535, -0.1645, -0.1529,
          -0.1918, -0.1987, -0.2686, -0.2107, -0.2132, -0.2063, -0.2253,
          -0.1670, -0.2638, -0.2669, -0.1228, -0.1679, -0.2795, -0.2066,
          -0.1625, -0.1498, -0.1983, -0.2351, -0.2337, -0.2803, -0.3189,
          -0.2672, -0.2565, -0.3481, -0.3722, -0.3055, -0.3174, -0.4059,
          -0.3363, -0.4085])

    if pad_len is None:
        pad_len = int(kernel.shape[0]/2)
        
    trace_pad = np.pad(trace, pad_len)
    return get_score_trace(trace_pad, kernel)


def get_bout_times(trace,
                   min_peak_value=0.75,
                   max_baseline_value=0.05,
                   include_nan=False,
                   max_zero_length=(55, 55),
                   min_bout_distance=0,
                   **kwargs):
    """Finds bout peaks and their start and end from a convolved
       trace (only tested on freely-swimming experiments).
    :param trace: the squared velocity trace convolved with a bout detection kernel.
    :param min_peak_value: the minimum peak value for a bump to count as a bout. 
        In NN terms this is the value of the sigmoid for classification, so 0.5
        Would be the cut-off value. However, one can opt for a higher or lower value
        if they want to change the false positive or true negative rate (i.e. a higher
        value should classify less noise as bouts but also means that more bouts will
        not be detected).
    :param max_baseline_value: (ab)uses the fact that a convolution will result in a smooth
        signal. The sides of the peak more or less reflect the start and end times of the
        bout. This is the cut-off value that determines at which points the peak start/ends
        and thus the bout starts/ends.
    :param include_nan: include bouts containing NaN values.
    :param max_length_to_baseline: the maximum number of samples that it may take to reach the
        baseline (defined by max_baseline_value) from the peak.
    :param min_bout_distance: minimum distance between the end of a bout and the peak of the
        next bout.
    :return: tuple: (bout start times, bout peak times, bout end times)
    """

    last_bout_end = 0
    bouts = []
        
    i = 0
    while i < trace.shape[0]:
        # Check if the value is high enough to indicate a bout.
        if trace[i] > min_peak_value:
            # We found a bout!
            bout_start = i
            bout_end = i
            
            # Check how far back it goes.
            j = i + 1
            found_start = False
            while j > last_bout_end:
                if trace[j] < max_baseline_value:
                    bout_start = j
                    found_start = True
                    break
                
                j -= 1
                    
            if not found_start:
                bout_start = last_bout_end
            
            # Check how far the end is.
            j = i + 1
            found_end = False
            while j < trace.shape[0]:
                if trace[j] < max_baseline_value:
                    bout_end = j
                    found_end = True
                    break
                elif trace[j-1] < min_peak_value < trace[j]:
                    # We found the next bout!
                    # We stop this bout at the valley.
                    bout_end = i + np.argmin(trace[i:j])
                    found_end = True
                    break
                
                j += 1
                    
            if not found_end:
                bout_end = trace.shape[0] - 1
            
            # Update values.
            last_bout_end = bout_end
            # Filter out bouts containing NaNs.
            if include_nan or not np.isnan(trace[bout_start:bout_end]).any():
                # The convolution during noise is less steep,
                # it might not reach the baseline value within the typical
                # bout time window. So we can use the requirement of the 'bout'
                # stopping within a certain time range, as a way to filter noise.
                if max_zero_length is None \
                    or (max_zero_length[0] > i - bout_start \
                    or max_zero_length[1] > bout_end - i):
                    bouts.append((bout_start, i, bout_end))
            
            # Skip the remaining part of the bout and optionally some extra distance.
            i = bout_end + min_bout_distance
        
        i += 1
        
    return bouts


def log_dt(log_df, i_start=10, i_end=110):
    return np.mean(np.diff(log_df.t[i_start:i_end]))


@jit(nopython=True)
def revert_segment_filling(fixed_mat, revert_pts):
    """
    Revert the filling of a tail segments matrix. Provided a data matrix and
    array with the numbers of segments to be reverted at each timepoint,
    this function will reset the previously-filled values to NaNs.

    :param fixed_mat: Data matrix (timepoints x n_segments) with the tail tracking data.
    :param revert_pts: Array (timepoints) registering how many segments were filled for each timepoint.
    :return:
    """
    # As they can be saved as uint8:
    int_pts = revert_pts.astype(np.int8)

    for i in range(len(int_pts)):
        if int_pts[i] > 0:
            fixed_mat[i, -int_pts[i] :] = np.nan

    return fixed_mat


@jit(nopython=True)
def n_missing_segments(tail_angle_mat):
    n_t, n_segments = tail_angle_mat.shape
    n_missing = np.zeros(n_t, dtype=np.uint8)
    for i in range(n_t):
        for i_seg in range(n_segments):
            if np.isnan(tail_angle_mat[i, i_seg]):
                n_missing[i] = n_segments - i_seg
                break
    return n_missing


@jit(nopython=True)
def fill_out_segments(tail_angle_mat, continue_curvature=0, revert_pts=None):
    """Fills out NaN values in a tail-tracking data matrix.
    Filling can consist on propagating the angle of the last tracked segment (continue_curvature=0)
    or on simulating the tail curvature by linearly-extrapolating the curvature of the last
    continue_curvature tracked segments.

    :param tail_angle_mat: Data matrix (timepoints x n_segments) with the tail tracking data.
    :param continue_curvature: Number of previous segments used for extrapolating curvature of each NaN segment.
    :return:
    """
    n_t, n_segments = tail_angle_mat.shape

    # If needed, revert previous filling (this could be done more efficiently
    # modifying later loop instead)
    if revert_pts is not None:
        tail_angle_mat = revert_segment_filling(tail_angle_mat, revert_pts)

    # To keep track of segments missing for every time point:
    n_segments_missing = np.zeros(n_t, dtype=np.uint8)

    # Fill in segments
    for i_t in range(tail_angle_mat.shape[0]):
        # If last value is nan...
        if np.isnan(tail_angle_mat[i_t, -1]):
            # ...loop over segments from the beginning...
            for i_seg in range(n_segments):
                # ...the first nan value marks where tail was interrupted, so
                if np.isnan(tail_angle_mat[i_t, i_seg]):
                    # 1) write how many we miss in n_segments_missing:
                    if n_segments_missing[i_t] == 0:
                        n_segments_missing[i_t] = n_segments - i_seg

                    # 2) Extrapolate:
                    if (
                        continue_curvature > 0
                        and i_seg > continue_curvature + 1
                    ):
                        # a) if we have at least continue_curvature+1 points
                        # and we want to interpolate from continue_curvature
                        # samples:
                        previous_tail_curvature = np.diff(
                            tail_angle_mat[
                                i_t,
                                i_seg - continue_curvature : i_seg,
                            ]
                        )
                        deviation = np.mean(previous_tail_curvature)
                    else:
                        # b) if we don't want to interpolate, we propagate the
                        # last value without additions:
                        deviation = 0
                    # write new values to array inplace:
                    tail_angle_mat[i_t, i_seg] = (
                        tail_angle_mat[i_t, i_seg - 1] + deviation
                    )

    return tail_angle_mat, n_segments_missing


@jit(nopython=True)
def nan_isolated(thetas):
    thetas_out = thetas.copy()
    for j in range(thetas_out.shape[1]):
        for i in range(1, thetas_out.shape[0] - 1):
            if np.isnan(thetas_out[i - 1, j]) and np.isnan(
                thetas_out[i + 1, j]
            ):
                thetas_out[i, j] = np.nan

    return thetas_out


@jit(nopython=True)
def mean_smooth(array, wnd):
    output = array.copy()
    for i in range(wnd, array.shape[0] - wnd):
        output[i] = np.nanmean(output[i - wnd : i + wnd])
    return output


def predictive_tail_fill(
    thetas,
    smooth_wnd=1,
    max_taildiff=np.pi / 2,
    start_from=4,
    fit_timepts=5,
    fit_tailpts=4,
):
    thetas -= np.nanmedian(thetas[:, 0])

    n_pts = thetas.shape[0]
    n_seg = thetas.shape[1]

    # Set to nan points with a suspect derivative:
    diff = np.concatenate([np.zeros((1, n_seg)), np.diff(thetas, axis=0)])
    too_deviating = np.abs(diff) > max_taildiff
    thetas[too_deviating] = np.nan
    thetas = nan_isolated(thetas)

    # smooth initial tail segments:
    for i in range(start_from):
        thetas[:, i] = mean_smooth(thetas[:, i], smooth_wnd)

    # Fitting loop:
    for i in range(start_from, n_seg):
        # First smooth previous thetas:
        thetas[:, i - 1] = mean_smooth(thetas[:, i - 1], smooth_wnd)
        # Find indexes to fix:
        i_to_fix = np.argwhere(np.isnan(thetas[:, i]))[:, 0]
        if len(i_to_fix) > 0:
            i_to_fit = np.argwhere(~np.isnan(thetas[:, i]))[:, 0]

            # Reorganize thetas, to use previous fit_timepts timepoints
            # and previous fit_tailpts segments for the fit
            # we need a (timepts, fit_timepts * fit_tailpts) matrix:
            theta_predict = [
                thetas[np.arange(n_pts) - t, i - s]
                for t, s in product(range(fit_timepts), range(1, fit_tailpts))
            ]

            theta_predict_res = np.column_stack(
                theta_predict + [np.ones(n_pts)]
            )
            # Linear fit with least squares:
            C, _, _, _ = linalg.lstsq(
                theta_predict_res[i_to_fit, :], thetas[i_to_fit, i]
            )
            # Predict the nan values:
            thetas[i_to_fix, i] = theta_predict_res[i_to_fix, :] @ C

    # smooth last angle:
    thetas[:, i] = mean_smooth(thetas[:, i], smooth_wnd)

    return thetas


def calc_vel(dx, t):
    """Calculates velocities from deltas and times, skipping over duplicated
    times

    Parameters
    ----------
    dx the differences in the parameter
    t times at which the parameter was sampled

    Returns
    -------
    t_vel, vel

    """
    dt = np.diff(t)
    duplicate_t = dt == 0
    vel = dx[~duplicate_t] / dt[~duplicate_t]
    t_vel = t[1:][~duplicate_t]
    return t_vel, vel


def bandpass(timeseries, dt, f_min=12, f_max=62, n_taps=9, axis=0):
    """Bandpass filtering used for tail motion, filters
    out unphysical frequencies for the fish tail

    :param timeseries:
    :param dt:
    :param f_min:
    :param f_max:
    :param n_taps:
    :return:
    """
    cfilt = signal.firwin(n_taps, [f_min, f_max], pass_zero=False, fs=1 / dt)
    return signal.filtfilt(cfilt, 1, timeseries, axis=axis)


def crop(traces, events, pre_int=20, post_int=30, **kwargs):
    """Apply cropping functions defined below depending on the dimensionality
    of the input (one cell or multiple cells). If input is pandas Series
    or DataFrame, it strips out the values first.
    :param traces: 1 (n_timepoints) or 2D (n_timepoints X n_cells) array,
                   pd.Series or pd.DataFrame with cells as columns.
    :param events: events around which to crop
    :param pre_int: interval to crop before the event, in points
    :param post_int: interval to crop after the event, in points
    :param dwn: downsampling (default=1 i.e. no downsampling)
    :return: events x n_points numpy array
    :return:
    """
    pre_int, post_int = int(pre_int), int(post_int)
    kwargs.update(dict(pre_int=pre_int, post_int=post_int))
    if isinstance(traces, pd.DataFrame) or isinstance(traces, pd.Series):
        traces = traces.values
    if isinstance(events, pd.DataFrame) or isinstance(events, pd.Series):
        events = events.values.flatten().astype(np.int)
    if len(traces.shape) == 1:
        return _crop_trace(traces, events, **kwargs)
    elif len(traces.shape) == 2:
        return _crop_block(traces, events, **kwargs)
    else:
        raise TypeError("traces matrix must be at most 2D!")


@jit(nopython=True)
def _crop_trace(trace, events, pre_int=20, post_int=30, dwn=1):
    """Crop the trace around specified events in a window given by parameters.
    :param trace: trace to be cropped
    :param events: events around which to crop
    :param pre_int: interval to crop before the event, in points
    :param post_int: interval to crop after the event, in points
    :param dwn: downsampling (default=1 i.e. no downsampling)
    :return: events x n_points numpy array
    """

    # Avoid problems with spikes at the borders:
    valid_events = (events > pre_int) & (events < len(trace) - post_int)

    mat = np.zeros((int((pre_int + post_int) / dwn), events.shape[0]))

    for i, s in enumerate(events):
        if valid_events[i]:
            cropped = trace[s - pre_int : s + post_int : dwn].copy()
            mat[: len(cropped), i] = cropped

    return mat


@jit(nopython=True)
def _crop_block(traces_block, events, pre_int=20, post_int=30, dwn=1):
    """Crop a block of traces
    :param traces_block: trace to be cropped (n_timepoints X n_cells)
    :param events: events around which to crop
    :param pre_int: interval to crop before the event, in points
    :param post_int: interval to crop after the event, in points
    :param dwn: downsampling (default=1 i.e. no downsampling)
    :return: events x n_points numpy array
    """

    n_timepts = traces_block.shape[0]
    n_cells = traces_block.shape[1]
    # Avoid problems with spikes at the borders:
    valid_events = events[(events > pre_int) & (events < n_timepts - post_int)]

    mat = np.full(
        (int((pre_int + post_int) / dwn), events.shape[0], n_cells), np.nan
    )

    for i, s in enumerate(events):
        if valid_events[i]:
            cropped = traces_block[s - pre_int : s + post_int : dwn, :].copy()
            mat[: len(cropped), i, :] = cropped

    return mat


def resample(df_in, resample_sec=0.005, fromzero=True):
    """

    Parameters
    ----------
    df_in :
    resample_sec :


    Returns
    -------

    """
    # TODO: probably everything should be interpolated nicely instead of resampled...
    df = df_in.copy()
    # Cut out negative times points if required:
    if fromzero:
        df = df[df["t"] > 0]
    t_index = pd.to_timedelta(df["t"].values, unit="s")
    df.set_index(t_index, inplace=True)

    df = df.resample("{}ms".format(int(resample_sec * 1000))).mean()
    df.index = df.index.total_seconds()
    return df.interpolate().drop("t", axis=1)


def polynomial_tail_coefficients(segments, n_max_missing=7, degree=3):
    """Fits a polynomial to the bout shape

    :param n_max_missing:
    :param degree: the polynomial degree
    :return:
    """
    segments -= segments[:, :1]
    n_max_missing = min(segments.shape[1] - degree, n_max_missing)

    # the Stytra tail tracking introduces NaNs at breaking point
    # a situation number - NaN - number never occurs in tracking
    n_missing = n_missing_segments(segments)

    poly_coefs = np.zeros((segments.shape[0], degree + 1))
    line_points = np.linspace(0, 1, segments.shape[1])

    for i_missing in range(n_max_missing + 1):
        sel_time = n_missing == i_missing
        poly_coefs[sel_time, :] = np.polynomial.polynomial.polyfit(
            line_points[: segments.shape[1] - i_missing],
            segments[sel_time, : segments.shape[1] - i_missing].T,
            degree,
        ).T
    return poly_coefs


def polynomial_tailsum(poly_coefs):
    return np.polynomial.polynomial.polyval(1, poly_coefs.T, False)


def reliability(data_block):
    """Function to calculate reliability of cell responses.
    Reliability is defined as the average of the across-trials correlation.
    This measure seems to generally converge after a number of 7-8 repetitions, so it
    is advisable to have such a repetition number to use it.
    :param data_block: block of cell responses (t x n_trials x n_cells)
    :return:
    """
    n_cells = data_block.shape[2]
    reliability = np.zeros(n_cells)
    for i in range(n_cells):
        cell_data = data_block[:, :, i]
        corr = fast_corrcoef(cell_data.T)
        np.fill_diagonal(corr, np.nan)
        reliability[i] = np.nanmean(corr)

    return reliability


@jit(nopython=True)
def fast_pearson(x, y):
    """Calculate correlation between two data series, excluding nan values.
    :param x: first array
    :param y: second array
    :return: pearson correlation
    """
    selection = ~np.isnan(x) & ~np.isnan(y)
    if not selection.any():
        return np.nan
    x = x[selection]
    y = y[selection]
    n = len(x)
    s_xy = 0.0
    s_x = 0.0
    s_y = 0.0
    s_x2 = 0.0
    s_y2 = 0.0
    for i in range(n):
        s_xy += x[i] * y[i]
        s_x += x[i]
        s_y += y[i]
        s_x2 += x[i] ** 2
        s_y2 += y[i] ** 2
    denominator = math.sqrt((s_x2 - (s_x**2) / n) * (s_y2 - (s_y**2) / n))
    if denominator == 0:
        return 0
    return (s_xy - s_x * s_y / n) / denominator


@jit(nopython=True)
def fast_corrcoef(mat):
    n = mat.shape[0]
    corrmat = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            corrmat[i, j] = fast_pearson(mat[i, :], mat[j, :])
            corrmat[j, i] = corrmat[i, j]

    return corrmat


@jit(nopython=True)
def compute_tbf(tail_sum, dt, min_valid_tps=5):
    """Estimate the instantaneous tail-beat frequency from the half-period of the tail oscillation.

    :param tail_sum: array of tracked tail angles
    :param dt: dt of the behavior array
    :param min_valid_tps: minimum array of valid timepoints within a bout to use in the TBF calculation
    :return: array with the tail-beat frequency.
    """

    idxs = np.arange(tail_sum.shape[0])
    tbf_output = np.full(idxs.shape[0], 0)

    min_idxs = []
    max_idxs = []

    for i in range(1, tail_sum.shape[0] - 1):
        if tail_sum[i - 1] < tail_sum[i] > tail_sum[i + 1]:
            max_idxs.append(i)
        elif tail_sum[i - 1] > tail_sum[i] < tail_sum[i + 1]:
            min_idxs.append(i)

    if len(min_idxs) == 0 or len(max_idxs) == 0:
        pass

    else:
        extrema = np.sort(
            np.concatenate((np.array(min_idxs), np.array(max_idxs)))
        )
        valid_idxs = idxs[
            np.logical_and(idxs >= min(extrema), idxs < max(extrema))
        ]

        if len(valid_idxs) > min_valid_tps:
            time_diffs = (
                np.array(
                    [x - extrema[i - 1] for i, x in enumerate(extrema)][1:]
                )
                * dt
            )
            binned_tps = np.digitize(valid_idxs, extrema) - 1
            instant_time_diff = np.array([time_diffs[i] for i in binned_tps])
            tbf = (1 / instant_time_diff) / 2
            tbf_output[valid_idxs[0] : valid_idxs[-1] + 1] = tbf

        else:
            pass

    return tbf_output
