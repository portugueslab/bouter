import numpy as np

from bouter.angles import quantize_directions


def test_quantize():
    dirs, bins = quantize_directions(
        np.array([0, -np.pi, np.pi, np.pi / 2 + 0.01])
    )
    np.testing.assert_array_equal(bins, [0, 4, 4, 2])
