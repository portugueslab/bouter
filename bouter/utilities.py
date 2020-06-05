import numpy as np
from numba import jit
from typing import Tuple


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
    # (because the pad_before perido has to be always included)
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


def log_dt(log_df, i_start=10, i_end=110):
    return np.mean(np.diff(log_df.t[i_start:i_end]))


@jit(nopython=True)
def revert_segment_filling(fixed_mat, revert_pts):
    """Revert filling of a tail segments matrix.
    :param fixed_mat:
    :param revert_pts:
    :return:
    """
    # As they can be saved as uint8:
    int_pts = revert_pts.astype(np.int8)

    for i in range(len(int_pts)):
        if int_pts[i] > 0:
            fixed_mat[i, -int_pts[i] :] = np.nan

    return fixed_mat


@jit(nopython=True)
def fill_out_segments(tail_angle_mat, continue_curvature=0, revert_pts=None):
    """Fills out segments of tail trace.

    :param tail_angle_mat
    :param continue_curvature
    :return:
    """
    n_t, n_segments = tail_angle_mat.shape

    # If needed, revert previous filling (this could be done more efficiently
    # modifying later loop instead)
    if revert_pts is not None:
        tail_angle_mat = revert_segment_filling(tail_angle_mat, revert_pts)

    # To keep track of segments missing for every time point:
    n_segments_missing = np.zeros(n_t, dtype=np.uint8)

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
                                i_t, i_seg - continue_curvature : i_seg,
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
