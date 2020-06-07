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


def test_bout_extraction():
    experiment = free.FreelySwimmingExperiment(dataset_path)
    bouts, cont = experiment.extract_bouts()

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
    bouts, _ = experiment.extract_bouts()

    #Summarize all the bouts detected in the experiment.
    bouts_summary = experiment.summarize_bouts(bouts)

    #Load summary of bouts from all fish in the experiment.
    loaded_bouts_summary = fl.load(ASSETS_PATH / "freely_swimming_dataset" / "test_extracted_bouts.h5", "/bouts_summary")

    assert_frame_equal(bouts_summary, loaded_bouts_summary)



