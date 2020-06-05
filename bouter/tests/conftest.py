import tempfile
import pytest
from distutils.dir_util import copy_tree
from pathlib import Path

from bouter.tests import ASSETS_PATH


@pytest.fixture()
def embedded_exp_path():
    tempdir = tempfile.mkdtemp()
    source_dataset_path = ASSETS_PATH / "embedded_dataset"
    copy_tree(source_dataset_path, tempdir)

    return Path(tempdir)
