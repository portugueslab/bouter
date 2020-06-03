import numpy as np
from numba import jit
from typing import Tuple, List


@jit(nopython=True)
def extract_segments_above_threshold(
    trace,
    threshold=0.1,
    min_length=20,
    min_between=25,
    break_segment_on_nan=True,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Extract periods from a trace where it's value is above threshold. The segments also can have a minimal length.
    Used for extracting bouts from vigor or velocity

    :param trace: vigor or velocity
    :param threshold: minimal value to be considered moving
    :param min_length: minimal number of samples of continuous movement
    :param min_between: minimal number of samples between two crossings
    :param break_segment_on_nan: if a NaN is encountered, it breaks the segment
    :return:
        segments: start and end indices of the segments above threshold
        connected: for each segment, whether it is connected to the previous. Segments are considered connected
            if there were no NaN values in the trace
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
        # the segment has ended and a positive threshold crossing has been found
        elif (
            i > i_last_segment_ended
            and trace[i - 1] < threshold < trace[i]
            and not in_segment
        ):
            in_segment = True
            start = i
        # a negative threshold crossing has been found while we are inside a sgement
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
