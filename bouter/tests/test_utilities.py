import numpy as np

from bouter.utilities import (
    extract_segments_above_threshold,
    fill_out_segments,
)


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
        min_between=1,
        break_segment_on_nan=False,
    )

    # first segment is ignored, because it is too short
    np.testing.assert_equal(segments[0, :], (15, 25))
    np.testing.assert_equal(segments[1, :], (29, 35))
    np.testing.assert_equal(segments[2, :], (40, 50))

    np.testing.assert_equal(continuous, np.array([False, False, True]))


def test_continue_curvature():
    n_segments = 6
    curvature = np.full((2, n_segments), np.nan)
    curvature[0, 0:4] = np.arange(4)
    curvature[1, 0:3] = np.arange(3)

    continued, n_segments_missing = fill_out_segments(curvature)

    np.testing.assert_equal(continued[0, 4:], 3)
    np.testing.assert_equal(continued[1, 3:], 2)

    curvature = np.full((4, n_segments), np.nan)
    curvature[0, 0:4] = np.arange(4)
    curvature[1, 0:3] = np.arange(3)
    curvature[2, 0:2] = np.arange(2)
    curvature[3, 0] = 1

    continued, n_segments_missing = fill_out_segments(
        curvature, continue_curvature=2
    )

    np.testing.assert_equal(continued[0, :], np.arange(n_segments))
    np.testing.assert_equal(n_segments_missing, [2, 3, 4, 5])
