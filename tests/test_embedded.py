import bouter
from numpy.testing import assert_array_almost_equal
from pathlib import Path
import flammkuchen as fl
import numpy as np

from bouter import embedded


ASSETS_PATH = Path(__file__).parent / "assets"


def test_class_instantiation():
    # Define Path
    dataset_path = ASSETS_PATH / "embedded_dataset"

    # Create Experiment class
    experiment = bouter.Experiment(dataset_path)

    assert type(experiment) == bouter.Experiment


def test_calculate_vigor():
    # Create EmbeddedExperiment class and calculate vigor
    dataset_path = ASSETS_PATH / "embedded_dataset"
    embedded_exp = embedded.EmbeddedExperiment(dataset_path)
    calculated_vigor = embedded_exp.vigor()

    expected_vigor = fl.load(ASSETS_PATH / "embedded_dataset" / "expected_vigor.h5", "/vigor")

    assert_array_almost_equal(calculated_vigor, expected_vigor, 5)
