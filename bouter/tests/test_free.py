import bouter
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from pathlib import Path
import flammkuchen as fl
from bouter.tests import ASSETS_PATH

from bouter import free


# Define dataset Path
dataset_path = ASSETS_PATH / "freely_swimming_dataset"

def test_n_fish_extraction():
    experiment = free.FreelySwimmingExperiment(dataset_path)
    assert experiment.n_fish == 3


def test_n_segment_extraction():
    experiment = free.FreelySwimmingExperiment(dataset_path)
    assert experiment.n_tail_segments == 9


def test_compute_velocity():
    experiment = free.FreelySwimmingExperiment(dataset_path)
    extended_behavior_log = experiment.compute_velocity()
    fish_vels = extended_behavior_log[["vel2_f{}".format(i_fish) for i_fish in range(experiment.n_fish)]]

    #Load computed velocities
    loaded_vel2 = fl.load(ASSETS_PATH / "freely_swimming_dataset" / "test_extracted_bouts.h5", "/velocities")

    #Compare DataFrame with velocities from the 3 experiment fish
    assert_frame_equal(fish_vels, loaded_vel2)


def test_bout_extraction():
    experiment = free.FreelySwimmingExperiment(dataset_path)
    bouts, cont = experiment.get_bouts()

    #Load expected bouts to be extracted. Only first fish is used for the assertion
    loaded_bouts = fl.load(ASSETS_PATH / "freely_swimming_dataset" / "test_extracted_bouts.h5", "/bouts")
    loaded_cont = fl.load(ASSETS_PATH / "freely_swimming_dataset" / "test_extracted_bouts.h5", "/continuity")

    #Compare dataframes for each of the detected bouts in the first fish
    for bout in range(len(bouts[0])):
        assert_frame_equal(bouts[0][bout], loaded_bouts[bout])

    #Compare also continuity array
    assert_array_equal(cont[0], loaded_cont)


def test_bout_summary():
    experiment = free.FreelySwimmingExperiment(dataset_path)

    #Summarize all the bouts detected in the experiment.
    bouts_summary = experiment.get_bout_properties()

    #Load summary of bouts from all fish in the experiment.
    loaded_bouts_summary = fl.load(ASSETS_PATH / "freely_swimming_dataset" / "test_extracted_bouts.h5", "/bouts_summary")

    assert_frame_equal(bouts_summary, loaded_bouts_summary)



