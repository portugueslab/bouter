import bouter
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from pathlib import Path
import h5py

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string

def test_class_instantiation():
    #Define Path
    dataset_path = Path(__file__).parent / "test_dataset"

    #Create Experiment class
    experiment = bouter.Experiment(dataset_path)

    assert type(experiment) == bouter.Experiment

def test_calculate_vigor():
    #Create EmbeddedExperiment class and calculate vigor
    dataset_path = Path(__file__).parent / "test_dataset"
    embedded_exp = bouter.EmbeddedExperiment(dataset_path)
    calculated_vigor = embedded_exp.vigor()

    #Load expected vigor
    with h5py.File(dataset_path.parent / 'test_data.h5', 'r') as hf:
        expected_vigor = hf['vigor'][:]

    assert_array_almost_equal(calculated_vigor, expected_vigor, 5)
