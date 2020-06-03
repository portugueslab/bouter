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
        trace, threshold=0.5, min_length=5, min_between=1, break_segment_on_nan=False,
    )

    # first segment is ignored, because it is too short
    np.testing.assert_equal(segments[0, :], (15, 25))
    np.testing.assert_equal(segments[1, :], (29, 35))
    np.testing.assert_equal(segments[2, :], (40, 50))

    np.testing.assert_equal(continuous, np.array([False, False, True]))
