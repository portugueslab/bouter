from numba import jit
import numpy as np


@jit(nopython=True)
def fill_out_segments(tail_angle_mat, continue_curvature=0):
    """ Fills out segments of tail trace

    :param tail_angle_mat:
    :return:
    """
    n_t, n_segments = tail_angle_mat.shape
    n_segments_missing = np.zeros(n_t, dtype=np.uint8)
    for i_t in range(tail_angle_mat.shape[0]):
        if np.isnan(tail_angle_mat[i_t, -1]):
            for i_seg in range(n_segments):
                if np.isnan(tail_angle_mat[i_t, i_seg]):
                    if continue_curvature:
                        deviation = np.mean(
                            np.diff(
                                tail_angle_mat[
                                    i_t,
                                    i_seg - 1 : i_seg - 1 - continue_curvature,
                                ]
                            )
                        )
                    else:
                        deviation = 0
                    tail_angle_mat[i_t, i_seg] = (
                        tail_angle_mat[i_t, i_seg - 1] + deviation
                    )
                n_segments_missing[i_t] = n_segments - i_seg
    return tail_angle_mat, n_segments_missing
