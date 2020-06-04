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
    ang_turn = np.full(bouts.shape[0], np.nan)
    ang_turn_tot = np.full(bouts.shape[0], np.nan)

    for i in range(bouts.shape[0]):
        s = bouts[i, 0]
        e = bouts[i, 1]

        peak_vig[i] = np.nanmax(vigor[s:e])
        med_vig[i] = np.nanmedian(vigor[s:e])
        ang_turn[i] = np.nansum(tail_sum[s : s + wnd_turn_pts])
        ang_turn_tot[i] = np.nansum(tail_sum[s:e])

    return peak_vig, med_vig, ang_turn, ang_turn_tot
