import numpy as np
from numba import jit


def quantize_directions(angles, n_dirs=8):
    """Bin angles into wedge bins

    Parameters
    ----------
    angles: array of angles
    n_dirs: number of wedges to split circles into

    Returns
    -------
    bin indices of the angles

    """

    bin_centers = np.arange(n_dirs + 1) * 2 * np.pi / n_dirs
    bin_hw = np.pi / n_dirs
    bins = np.r_[bin_centers - bin_hw, bin_centers[-1] + bin_hw]
    return (
        bin_centers[:n_dirs],
        (np.digitize(np.mod(angles, np.pi * 2), bins) - 1) % n_dirs,
    )


def reduce_to_pi(ar):
    """Reduce angles to the -pi to pi range"""
    return np.mod(ar + np.pi, np.pi * 2) - np.pi


def angle_mean(angles, axis=1):
    """Correct calculation of a mean of an array of angles

    Parameters
    ----------
    angles :

    axis :
         (Default value = 1)

    Returns
    -------

    """
    return np.arctan2(
        np.sum(np.sin(angles), axis), np.sum(np.cos(angles), axis)
    )


def angle_dif(a, b):
    """

    Parameters
    ----------
    a :

    b :


    Returns
    -------
    type


    """
    return np.minimum(
        np.minimum(np.abs(a - b), np.abs(a - b + 2 * np.pi)),
        np.abs(a - b - 2 * np.pi),
    )


def cossin(theta):
    """Expands theta into sine and cosine"""
    return np.array((np.cos(theta), np.sin(theta)))


@jit(nopython=True)
def transform_affine_point(point, tm):
    return tm[:, :2] @ point + tm[:, 2]


@jit(nopython=True)
def transform_affine(points, tm):
    """Affine transform a point or an array of points with the matrix tm

    Parameters
    ----------
    points :

    tm :


    Returns
    -------

    """
    padded = np.ones((points.shape[0], points.shape[1] + 1))
    padded[:, :-1] = points

    return padded @ tm.T


def rot_mat(theta):
    """The rotation matrix for an angle theta """
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


@jit(nopython=True)
def smooth_tail_angles(tail_angles):
    """Smooths out the tau jumps in tail angles, so that the angle between
    tail segments is smoothly changing

    Parameters
    ----------
    tail_angles :
        return:

    Returns
    -------

    """

    tau = 2 * np.pi

    for i in range(1, tail_angles.shape[0]):
        previous = tail_angles[i - 1]
        dist = np.abs(previous - tail_angles[i])
        if np.abs(previous - (tail_angles[i] + tau)) < dist:
            tail_angles[i] += tau
        elif np.abs(previous - (tail_angles[i] - tau)) < dist:
            tail_angles[i] -= tau

    return tail_angles


@jit(nopython=True)
def smooth_tail_angles_series(tail_angles_series):
    """Smooths out the tau jumps in tail angles, so that the angle between
    tail segments is smoothly changing, applied on series

    Parameters
    ----------
    tail_angles :
        return:
    tail_angles_series :


    Returns
    -------

    """
    # TODO use guvecotorize to avoid having this function

    for i_t in range(tail_angles_series.shape[0]):
        smooth_tail_angles(tail_angles_series[i_t, :])

    return tail_angles_series
