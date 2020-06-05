import bouter
from numpy.testing import assert_array_almost_equal
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

