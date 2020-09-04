import numpy as np
from numba import jit


@jit(nopython=True)
def bout_stats(vigor, tail_sum, bouts, wnd_turn_pts):
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
        bias[i] = np.nanmean(tail_sum[s : s + wnd_turn_pts])
        bias_tot[i] = np.nanmean(tail_sum[s:e])

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
