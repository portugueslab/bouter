import bouter
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from pathlib import Path
import flammkuchen as fl
import numpy as np

from bouter import free


ASSETS_PATH = Path(__file__).parent / "assets"

# Define dataset Path
dataset_path = ASSETS_PATH / "freely_swimming_dataset"

def test_n_fish_extraction():
    experiment = free.FreelySwimmingExperiment(dataset_path)
    assert experiment.get_n_fish() == 3


def test_n_segment_extraction():
    experiment = free.FreelySwimmingExperiment(dataset_path)
    assert experiment.get_n_segments() == 9


def test_bout_extraction():
    experiment = free.FreelySwimmingExperiment(dataset_path)
    bouts, cont = experiment.extract_bouts()

    #Load expected bouts to be extracted. ONly first fish is used for the assertion
    loaded_bouts = fl.load(ASSETS_PATH / "freely_swimming_dataset" / "test_extracted_bouts.h5", "/bouts")
    loaded_cont = fl.load(ASSETS_PATH / "freely_swimming_dataset" / "test_extracted_bouts.h5", "/continuity")

    #Compare dataframes for each of the detected bouts in the first fish
    for bout in range(len(bouts[0])):
        assert_frame_equal(bouts[0][bout], loaded_bouts[bout])

    #Compare also continuity array
    assert_array_equal(cont[0], loaded_cont)

