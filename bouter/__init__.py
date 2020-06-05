__version__ = "0.1.1"

from pathlib import Path

from bouter.experiment import Experiment
from bouter.embedded import EmbeddedExperiment
from bouter.multisession_exp import MultiSessionExperiment

# Locate assets
ASSETS_PATH = Path(__file__).parent.parent / "tests" / "assets"


def load_folder(folder, exp_class=Experiment):
    """ Load all experiments in the folder and subfolders

    :param folder:
        folder containing the _metadata.json files (also searches subfolders)
    :param exp_class:
        the kind of experiment to make
    :return:
    """
    folder = Path(folder)
    return [exp_class(f) for f in sorted(folder.glob("**/*_metadata.json"))]
