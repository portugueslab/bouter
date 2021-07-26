import numpy as np
from numba import jit


@jit(nopython=True)
def bout_stats(vigor, tail_sum, bouts, wnd_turn_pts, th_offset_window_pts):
    """
    Compute statistics of bouts from vigor trace, tail sum,
    and bouts indexes.
    """
    peak_vig = np.full(bouts.shape[0], np.nan)
    med_vig = np.full(bouts.shape[0], np.nan)
    bias = np.full(bouts.shape[0], np.nan)
    bias_tot = np.full(bouts.shape[0], np.nan)

    for i in range(bouts.shape[0]):
        s = bouts[i, 0]
        e = bouts[i, 1]

        peak_vig[i] = np.nanmax(vigor[s:e])
        med_vig[i] = np.nanmedian(vigor[s:e])

        theta_offset = np.nanmean(tail_sum[s - th_offset_window_pts: s])
        bias[i] = np.nanmean(tail_sum[s: s + wnd_turn_pts]) - theta_offset
        bias_tot[i] = np.nanmean(tail_sum[s:e]) - theta_offset

    return peak_vig, med_vig, bias, bias_tot


@jit(nopython=True)
def count_peaks_between(ts, start_indices, end_indices, min_peak_dist=5):
    pos_peaks = np.zeros(len(start_indices), dtype=np.uint8)
    neg_peaks = np.zeros(len(start_indices), dtype=np.uint8)
    for i_bout, (si, ei) in enumerate(zip(start_indices, end_indices)):
        i = si
        while i < ei - 1:
            if ts[i - 1] < ts[i] > ts[i + 1]:
                pos_peaks[i_bout] += 1
                i += min_peak_dist
            elif ts[i - 1] > ts[i] < ts[i + 1]:
                neg_peaks[i_bout] += 1
                i += min_peak_dist
            else:
                i += 1
    return pos_peaks, neg_peaks


@jit(nopython=True)
def compute_tbf(tail_sum, start_indices, end_indices, dt):
    min_idxs = []
    max_idxs = []

    tbf_list = []

    for start_i, end_i in zip(start_indices, end_indices):
        bout_tail_sum = tail_sum[start_i:end_i]

        for i in range(1, bout_tail_sum.shape[0] - 1):
            if bout_tail_sum[i - 1] < bout_tail_sum[i] > bout_tail_sum[i + 1]:
                max_idxs.append(i)
            elif bout_tail_sum[i - 1] > bout_tail_sum[i] < bout_tail_sum[i + 1]:
                min_idxs.append(i)

        extrema = np.sort(np.concatenate((np.array(min_idxs), np.array(max_idxs))))

        idxs = np.arange(bout_tail_sum.shape[0])
        valid_idxs = idxs[np.logical_and(idxs >= min(extrema), idxs < max(extrema))]

        time_diffs = np.array([x - extrema[i - 1] for i, x in enumerate(extrema)][1:]) * dt

        binned_tps = np.digitize(valid_idxs, extrema) - 1

        instant_time_diff = np.array([time_diffs[i] for i in binned_tps])

        bout_tbf = (1 / instant_time_diff) / 2

        bout_output = np.full(idxs.shape[0], np.nan)
        bout_output[valid_idxs[0]:valid_idxs[-1] + 1] = bout_tbf
        tbf_list.append(bout_output)

    return tbf_list
