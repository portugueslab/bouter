import tempfile
from pathlib import Path

from bouter import embedded, free
from bouter.nwb.conversion import experiment_to_nwb


def test_convert_freely_swimming(freely_swimming_exp_path):
    exp = free.FreelySwimmingExperiment(freely_swimming_exp_path)
    tempdir = tempfile.mkdtemp()
    experiment_to_nwb(exp, Path(tempdir) / "free.nwb")


def test_convert_embedded(embedded_exp_path):
    exp = embedded.EmbeddedExperiment(embedded_exp_path)
    tempdir = tempfile.mkdtemp()
    experiment_to_nwb(exp, Path(tempdir) / "embedded.nwb")
