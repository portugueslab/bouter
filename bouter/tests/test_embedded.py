import flammkuchen as fl
import numpy as np
from numpy.testing import assert_array_almost_equal

from bouter import embedded, utilities
from bouter.tests import ASSETS_PATH


def corrupt_tail_matrix(tail_matrix, prop_missing=0.2, std=3):
    """Add some nan values to tail segments."""
    np.random.seed(2341)

    corrupted_mat = tail_matrix.copy()
    n_pts = corrupted_mat.shape[0]

    for i in range(n_pts):
        if np.random.rand() < prop_missing:
            n_missing = abs(int(np.random.randn() * std)) + 1
            corrupted_mat[i, -n_missing:] = np.nan

    return corrupted_mat


def test_class_simple_properties(embedded_exp_path):
    # Create Experiment class
    exp = embedded.EmbeddedExperiment(embedded_exp_path)

    assert exp.n_tail_segments == 25
    assert exp.tail_columns == [f"theta_{i:02}" for i in range(25)]


def test_tail_fix_function_real_data(embedded_exp_path):
    exp = embedded.EmbeddedExperiment(embedded_exp_path)

    # Original dataframe:
    original_mat = exp.behavior_log.loc[:, exp.tail_columns].values.copy()

    # Nan random points:
    corrupted_mat = corrupt_tail_matrix(original_mat, prop_missing=0.3, std=1)

    fixed_mat, pts = utilities.fill_out_segments(
        corrupted_mat.copy(), continue_curvature=4
    )

    fixed = np.isnan(corrupted_mat)
    diff_mat = np.abs(original_mat - fixed_mat)

    assert np.median(diff_mat[fixed]) < 0.1
    assert np.median(diff_mat[~fixed]) == 0

    reverted_mat = utilities.revert_segment_filling(
        fixed_mat,
        revert_pts=pts,
    )

    assert_array_almost_equal(reverted_mat, corrupted_mat)


def test_tail_fixing_experiment(embedded_exp_path):
    exp = embedded.EmbeddedExperiment(embedded_exp_path)

    # Original dataframe:
    original_mat = exp.behavior_log.loc[:, exp.tail_columns].values.copy()

    # Nan random points and set to real data:
    corrupted_mat = corrupt_tail_matrix(original_mat, prop_missing=0.3, std=1)
    exp.behavior_log.loc[:, exp.tail_columns] = corrupted_mat

    # Fix the log inplace:
    exp.reconstruct_missing_segments(continue_curvature=4)

    fixed_pts = np.isnan(corrupted_mat)
    diff_mat = np.abs(
        original_mat - exp.behavior_log.loc[:, exp.tail_columns].values
    )

    assert np.median(diff_mat[fixed_pts]) < 0.1
    assert np.median(diff_mat[~fixed_pts]) == 0


def test_tail_fixing_revert(embedded_exp_path):
    exp = embedded.EmbeddedExperiment(embedded_exp_path)

    # Original dataframe:
    original_mat = exp.behavior_log.loc[:, exp.tail_columns].values.copy()

    # Nan random points and set to real data:
    corrupted_mat = corrupt_tail_matrix(original_mat, prop_missing=0.3, std=1)
    exp.behavior_log.loc[:, exp.tail_columns] = corrupted_mat

    # Fix the log inplace:
    exp.reconstruct_missing_segments(continue_curvature=4)

    # Revert:
    exp.reconstruct_missing_segments(continue_curvature=None)

    assert_array_almost_equal(
        corrupted_mat, exp.behavior_log.loc[:, exp.tail_columns].values
    )


def test_vigor_and_bouts(embedded_exp_path):
    # Create EmbeddedExperiment class and calculate vigor
    embedded_exp = embedded.EmbeddedExperiment(embedded_exp_path)
    calculated_vigor = embedded_exp.compute_vigor(vigor_duration_s=0.05)[
        "vigor"
    ]

    expected_vigor = fl.load(
        ASSETS_PATH / "embedded_dataset" / "expected_vigor.h5", "/vigor"
    )

    assert_array_almost_equal(calculated_vigor, expected_vigor, 5)

    bouts = embedded_exp.get_bouts(vigor_threshold=0.1)

    assert_array_almost_equal(
        bouts, np.array([[40, 236], [377, 584], [717, 887], [1015, 1174]])
    )

    bts_df = embedded_exp.get_bout_properties(directionality_duration=0.07)
    assert all(
        bts_df.columns
        == [
            "t_start",
            "duration",
            "peak_vig",
            "med_vig",
            "bias",
            "bias_total",
            "n_pos_peaks",
            "n_neg_peaks",
        ]
    )

    np.testing.assert_array_almost_equal(
        bts_df.values,
        np.array(
            [
                [
                    0.133313,
                    0.654065,
                    1.543424,
                    1.424966,
                    -0.08797419,
                    -0.0261515,
                    15.0,
                    15.0,
                ],
                [
                    1.257425,
                    0.689069,
                    1.850382,
                    1.495598,
                    0.9017999,
                    0.169295,
                    15.0,
                    14.0,
                ],
                [
                    2.390538,
                    0.567057,
                    2.045012,
                    1.478295,
                    0.90041942,
                    0.22395974,
                    13.0,
                    12.0,
                ],
                [
                    3.384638,
                    0.529053,
                    1.738690,
                    1.452299,
                    0.74527736,
                    0.13003394,
                    11.0,
                    13.0,
                ],
            ]
        ),
        1,
    )
