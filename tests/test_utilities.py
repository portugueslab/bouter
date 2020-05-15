import numpy as np
from bouter.utilities import extract_segments_above_threshold


def test_segment_extraction():
    trace = np.zeros(100)
    trace[10:12] = 1
    trace[15:25] = 1
    trace[27] = np.nan
    trace[29:35] = 1
    trace[40:50] = 1
    segments, continuous = extract_segments_above_threshold(
        trace,
        threshold=0.5,
        min_length=5,
        pad_before=1,
        pad_after=1,
        break_segment_on_nan=False,
    )

    # first segment is ignored, because it is too short
    assert segments[0] == (14, 26)
    assert segments[1] == (28, 36)
    assert segments[2] == (39, 51)

    assert continuous == [False, False, True]
