import flammkuchen as fl
import numpy as np
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from bouter import free
from bouter.tests import ASSETS_PATH


def corrupt_tail_matrix(tail_matrix, prop_missing=0.2, std=3):
    """Add some nan values to tail segments."""
    np.random.seed(1520)

    corrupted_mat = tail_matrix.copy()
    n_pts = corrupted_mat.shape[0]

    for i in range(n_pts):
        if np.random.rand() < prop_missing:
            n_missing = abs(int(np.random.randn() * std)) + 1
            corrupted_mat[i, -n_missing:] = np.nan

    return corrupted_mat


def test_n_fish_extraction(freely_swimming_exp_path):
    experiment = free.FreelySwimmingExperiment(freely_swimming_exp_path)
    assert experiment.n_fish == 3


def test_n_segment_extraction(freely_swimming_exp_path):
    experiment = free.FreelySwimmingExperiment(freely_swimming_exp_path)
    assert experiment.n_tail_segments == 9


def test_tail_fixing_experiment(freely_swimming_exp_path):
    experiment = free.FreelySwimmingExperiment(freely_swimming_exp_path)

    # Empty lists to store the generated corrupted matrices
    original_mats = []
    corrupted_mats = []

    for i_fish in range(experiment.n_fish):

        # Load original dataframe:
        original_fish_mat = experiment.behavior_log.loc[
            :, experiment.tail_columns[i_fish]
        ].values.copy()
        original_mats.append(original_fish_mat)

        # Nan random points and substitute experiment data, in place:
        corrupted_fish_mat = corrupt_tail_matrix(
            original_fish_mat, prop_missing=0.3, std=1
        )
        corrupted_mats.append(corrupted_fish_mat)
        experiment.behavior_log.loc[
            :, experiment.tail_columns[i_fish]
        ] = corrupted_fish_mat

    # Fix the experiment behavior_log, in place
    experiment.reconstruct_missing_segments(continue_curvature=4)

    # Assert success for each fish in the experiment
    for i_fish in range(experiment.n_fish):

        fixed_pts = np.isnan(corrupted_mats[i_fish])
        diff_mat = np.abs(
            original_mats[i_fish]
            - experiment.behavior_log.loc[
                :, experiment.tail_columns[i_fish]
            ].values
        )

        assert np.median(diff_mat[fixed_pts]) < 0.1
        assert np.median(diff_mat[~fixed_pts]) == 0


def test_tail_fixing_revert(freely_swimming_exp_path):
    experiment = free.FreelySwimmingExperiment(freely_swimming_exp_path)

    for i_fish in range(experiment.n_fish):

        # Load original dataframe:
        original_fish_mat = experiment.behavior_log.loc[
            :, experiment.tail_columns[i_fish]
        ].values.copy()

        # Nan random points and substitute experiment data, in place:
        corrupted_fish_mat = corrupt_tail_matrix(
            original_fish_mat, prop_missing=0.3, std=1
        )
        experiment.behavior_log.loc[
            :, experiment.tail_columns[i_fish]
        ] = corrupted_fish_mat

    corrupted_behavior_log = experiment.behavior_log

    # Fix the behavior_log, in place:
    experiment.reconstruct_missing_segments(continue_curvature=4)

    # Revert fixing:
    experiment.reconstruct_missing_segments(continue_curvature=None)

    assert_frame_equal(corrupted_behavior_log, experiment.behavior_log)


def test_compute_velocity(freely_swimming_exp_path):
    experiment = free.FreelySwimmingExperiment(freely_swimming_exp_path)
    velocities_df = experiment.compute_velocity()
    fish_vels = velocities_df[
        ["vel2_f{}".format(i_fish) for i_fish in range(experiment.n_fish)]
    ]

    # Load computed velocities
    loaded_vel2 = fl.load(
        ASSETS_PATH / "freely_swimming_dataset" / "test_extracted_bouts.h5",
        "/velocities",
    )

    # Compare DataFrame with velocities from the 3 experiment fish
    assert_frame_equal(fish_vels, loaded_vel2)


def test_bout_extraction(freely_swimming_exp_path):
    experiment = free.FreelySwimmingExperiment(freely_swimming_exp_path)
    bouts, cont = experiment.get_bouts()

    # Load expected bouts to be extracted. Only first fish is used for the assertion
    loaded_bouts = fl.load(
        ASSETS_PATH / "freely_swimming_dataset" / "test_extracted_bouts.h5",
        "/bouts",
    )
    loaded_cont = fl.load(
        ASSETS_PATH / "freely_swimming_dataset" / "test_extracted_bouts.h5",
        "/continuity",
    )

    # Compare dataframes for each of the detected bouts in the first fish
    for bout in range(len(bouts[0])):
        assert_frame_equal(bouts[0][bout], loaded_bouts[bout])

    # Compare also continuity array
    assert_array_equal(cont[0], loaded_cont)


def test_bout_summary(freely_swimming_exp_path):
    experiment = free.FreelySwimmingExperiment(freely_swimming_exp_path)

    # Summarize all the bouts detected in the experiment.
    bouts_summary = experiment.get_bout_properties()

    # Load summary of bouts from all fish in the experiment.
    loaded_bouts_summary = fl.load(
        ASSETS_PATH / "freely_swimming_dataset" / "test_extracted_bouts.h5",
        "/bouts_summary",
    )

    assert_frame_equal(bouts_summary, loaded_bouts_summary)
