import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import json

from bouter import decorators
from bouter import utilities
from bouter import descriptors


def root_sid_from_path(path, session_id=None):
    """Read root and session id from several combinations of inputs.
    :param path:
    :param session_id:
    :return:
    """

    # Prepare path:
    inpath = Path(path)

    # if we are passing a metadata file:
    if inpath.suffix == ".json":
        root = inpath.parent
        session_id = "_".join(inpath.name.split("_")[:-1])

    # if this is a full directory:
    else:
        root = inpath

        if session_id is None:
            pattern = "*" + descriptors.METADATA_SFX
            meta_files = sorted(list(root.glob(pattern)))

            # Load metadata:
            if len(meta_files) == 0:
                raise FileNotFoundError("No metadata file in specified path!")
            elif len(meta_files) > 1:
                raise FileNotFoundError(
                    "Multiple metadata files in specified path!"
                )
            else:
                session_id = str(meta_files[0].name).split("_")[0]

    return root, session_id


class Experiment(dict):
    """

    Parameters
    ----------
    path :


    Returns
    -------

    """

    log_mapping = dict(
        stimulus_param_log=["stimulus_log"],
        estimator_log=["estimator_log"],
        behavior_log=["behavior_log"],
    )

    def __init__(
        self, path, session_id=None, cache_active=True, default_cached=True
    ):

        # If true forces to use cached with whatever params it was computed:
        self.default_cached = default_cached
        self.cache_active = cache_active

        self.root, self.session_id = root_sid_from_path(
            path, session_id=session_id
        )
        print(self.root, self.session_id)

        metadata_file = self.root / (
            self.session_id + descriptors.METADATA_SFX
        )

        with open(str(metadata_file), "r") as f:
            source_metadata = json.load(f)

        # TODO semipermanent?
        # Temporary workaround for Stytra saving mess:
        try:
            source_metadata["behavior"] = source_metadata.pop("tracking")
        except KeyError:
            pass

        super().__init__(self, **source_metadata)

        # Make list with all the files referring to this experiment:

        # Private attributes for properties caching:
        self._behavior_dt = None
        self._stimulus_param_log = None
        self._behavior_log = None
        self._estimator_log = None

    @property
    def params_filename(self):
        return self.root / (self.session_id + descriptors.PARAMS_LOG_SFX)

    @property
    def processing_params(self):
        """As a property it automatically keeps the log in synch.
        """
        if self.cache_active:

            if self.params_filename.exists():
                with open(self.params_filename, "r") as f:
                    processing_params = json.load(f)
            else:
                processing_params = {}
                with open(self.params_filename, "w") as f:
                    json.dump({}, f)

            return processing_params
        else:
            raise ValueError(
                "You want to use cached params but the cache_active flag attribute is false!"
            )

    @processing_params.setter
    def processing_params(self, vals):
        if self.cache_active:
            with open(self.params_filename, "w") as f:
                json.dump(vals, f)
        else:
            raise ValueError(
                "You want to store cached params but the cache_active flag attribute is false!"
            )

    def update_processing_params(self, new_dict):
        params = self.processing_params
        params.update(new_dict)
        self.processing_params = params

    @property
    def file_list(self):
        """As a property it automatically updates the cached files.
        """
        return list(self.root.glob(f"{self.session_id}*"))

    @property
    def fish_id(self):
        return self["general"]["fish_id"]

    @property
    def full_name(self):
        return self.fish_id + "_" + self.session_id

    @property
    def behavior_dt(self):
        if self._behavior_dt is None:
            self._behavior_dt = utilities.log_dt(self.behavior_log)

        return self._behavior_dt

    @property
    def protocol_parameters(self):
        """ Goes around annoying problem of knowing experiment number
        and version as keywords for the stimulus parameters dictionary.
        """
        if self.protocol_version is None:
            return self["stimulus"]["protocol"][self.protocol_name]
        else:
            return self["stimulus"]["protocol"][self.protocol_name][
                self.protocol_version
            ]

    @property
    def protocol_name(self):
        return list(self["stimulus"]["protocol"].keys())[0]

    @property
    def protocol_version(self):
        containing_dict = self["stimulus"]["protocol"][self.protocol_name]
        version = list(containing_dict.keys())[0]

        # b/c of the funny nesting {protocol_name: {protocol_version: {}}},
        # and since protocols might not have a version:
        if isinstance(containing_dict[version], dict):
            return version
        else:
            return None

    @property
    def stim_start_times(self):
        """ Get start and end time of all stimuli in the log.
        :return: arrays with start and end times for all stimuli
        """
        return np.array([stim["t_start"] for stim in self["stimulus"]["log"]])

    @property
    def stim_end_times(self):
        """ Get start and end time of all stimuli in the log.
        :return: arrays with start and end times for all stimuli
        """
        return np.array([stim["t_stop"] for stim in self["stimulus"]["log"]])

    @property
    def stimulus_param_log(self):
        return self._get_log("stimulus_param_log")

    @property
    def estimator_log(self):
        return self._get_log("estimator_log")

    @property
    def behavior_log(self):
        return self._get_log("behavior_log")

    def _get_log(self, log_name):
        """ Given name of the log get it from attributes or load it ex novo
        :param log_name:  string with the type ot the log to load
        :return:  loaded log DataFrame
        """
        uname = "_" + log_name

        # Check whether this was already set:
        if getattr(self, uname) is None:

            # If not, loop over different possibilities for that filename
            for possible_name in self.log_mapping[log_name]:
                try:
                    # Load and set attribute
                    logname = next(
                        self.root.glob(
                            self.session_id + "_" + possible_name + ".*"
                        )
                    ).name
                    setattr(self, uname, self._load_log(logname))
                    break
                except StopIteration:
                    pass
            else:
                raise ValueError(log_name + " does not exist")

        return getattr(self, uname)

    def _load_log(self, data_name):
        """

        Parameters
        ----------
        data_name : filename to load


        Returns
        -------
        Loaded dataframe

        """

        file = self.root / data_name
        if file.suffix == ".csv":
            return pd.read_csv(str(file), delimiter=";").drop(
                "Unnamed: 0", axis=1
            )
        elif file.suffix == ".h5" or file.suffix == ".hdf5":
            return pd.read_hdf(file)
        elif file.suffix == ".feather":
            return pd.read_feather(file)
        elif file.suffix == ".json":
            return pd.read_json(file)
        else:
            raise ValueError(
                str(data_name)
                + " format is not supported, trying to load "
                + str(file)
            )

    def copy_to_dir(self, target_dir):
        """ Copy all the files pertaining to this experiment in a target
        directory. If it does not exist, make it.
        """
        target_dir = Path(target_dir)

        if target_dir != self.root:  # Make sure we don't overwrite
            if not target_dir.exists():
                target_dir.mkdir()

            for f in self.file_list:
                shutil.copyfile(f, target_dir / f.name)

    def clear_cache(self):
        for file in self.root.glob(decorators.CACHE_FILE_TEMPLATE.format("*")):
            shutil.rmtree(file)

    def get_bouts(self):
        raise

    @decorators.deprecated(
        "Use Experiment.stim_start_times and stim_end_times instead."
    )
    def stimulus_starts_ends(self):
        """ Get start and end time of all stimuli in the log.
        :return: arrays with start and end times for all stimuli
        """
        return self.stim_start_times, self.stim_end_times
