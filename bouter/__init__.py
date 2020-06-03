from bouter.core_exp import Experiment
from bouter.embedded_exp import EmbeddedExperiment
from bouter.multisession_exp import MultiSessionExperiment


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
