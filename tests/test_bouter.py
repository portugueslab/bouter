import bouter
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

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
    experiment = bouter.Experiment("test_dataset/")

    assert type(experiment) == bouter.Experiment

def test_calculate_vigor():
    embedded_exp = bouter.EmbeddedExperiment("test_dataset/")
    calculated_vigor = embedded_exp.vigor()
    expected_vigor = np.loadtxt("vigor.txt")

    assert_array_almost_equal(calculated_vigor, expected_vigor, 5)
