__version__ = "0.1.1"

from pathlib import Path

from bouter.experiment import Experiment
from bouter.embedded import EmbeddedExperiment
from bouter.multisession_exp import MultiSessionExperiment
import numpy as np

from bouter import tests

# Locate assets
ASSETS_PATH = Path(tests.__file__).parent / "assets"


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


def get_scale_mm(exp_class: Experiment):
    """ Return camera pixel size in millimeters

    :param exp:
    :return:
    """
    cal_params = exp_class["stimulus"]["calibration_params"]
    if exp_class["general"]["animal"]["embedded"]:
        return cal_params["mm_px"]
    else:
        proj_mat = np.array(cal_params["cam_to_proj"])
        return (
            np.linalg.norm(np.array([1.0, 0.0]) @ proj_mat[:, :2]) * cal_params["mm_px"]
        )
