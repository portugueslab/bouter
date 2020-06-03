from numba import jit
import numpy as np


@jit(nopython=True)
def fill_out_segments(tail_angle_mat):
    """ Fills out segments of tail trace

    :param tail_angle_mat:
    :return:
    """
    n_t, n_segments = tail_angle_mat.shape
    for i_t in range(tail_angle_mat.shape[0]):
        if np.isnan(tail_angle_mat[i_t, -1]):
            for i_seg in range(n_segments):
                if np.isnan(tail_angle_mat[i_t, i_seg]):
                    tail_angle_mat[i_t, i_seg] = tail_angle_mat[i_t, i_seg - 1]

    return tail_angle_mat
