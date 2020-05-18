import numpy as np
import pandas as pd
from pathlib import Path
from shutil import copyfile
import json
from datetime import datetime


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

    def __init__(self, path, session_id=None):
        # Prepare path:
        inpath = Path(path)

        if inpath.suffix == ".json":  # if we are passing a metadata file:
            self.root = inpath.parent
            session_id = "_".join(inpath.name.split("_")[:-1])

        else:  # if this is a full directory:
            self.root = Path(path)

            if session_id is None:
                meta_files = sorted(list(self.root.glob("*metadata.json")))

                # Load metadata:
                if len(meta_files) == 0:
                    raise FileNotFoundError("No metadata file in specified path!")
                elif len(meta_files) > 1:
                    raise FileNotFoundError(
                        "Multiple metadata files in specified path!"
                    )
                else:
                    session_id = str(meta_files[0].name).split("_")[0]

        metadata_file = self.root / (session_id + "_metadata.json")
        with open(str(metadata_file), "r") as f:
            source_metadata = json.load(f)

        self.fish_id = source_metadata["general"]["fish_id"]
        self.session_id = session_id

        self.full_name = self.fish_id + "_" + self.session_id

        # TODO semipermanent?
        # Temporary workaround for Stytra saving mess:
        try:
            source_metadata["behavior"] = source_metadata.pop("tracking")
        except KeyError:
            pass

        # Make list with all the files referring to this experiment:
        self.file_list = list(self.root.glob(f"{self.session_id}*"))
        self._stimulus_param_log = None
        self._behavior_log = None
        self._estimator_log = None
        super().__init__(**source_metadata)

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
                        self.root.glob(self.session_id + "_" + possible_name + ".*")
                    ).name
                    setattr(self, uname, self._load_log(logname))
                    break
                except StopIteration:
                    pass
            else:
                raise ValueError(log_name + " does not exist")

        return getattr(self, uname)

    @property
    def protocol_parameters(self):
        """ Goes around annoying problem of knowing experiment number
        and version as keywords for the stimulus parameters dictionary.
        """
        return self["stimulus"]["protocol"][self.protocol_name][self.protocol_version]

    @property
    def protocol_name(self):
        return list(self["stimulus"]["protocol"].keys())[0]

    @property
    def protocol_version(self):
        return list(self["stimulus"]["protocol"][self.protocol_name].keys())[0]

    @property
    def stimulus_param_log(self):
        return self._get_log("stimulus_param_log")

    @property
    def estimator_log(self):
        return self._get_log("estimator_log")

    @property
    def behavior_log(self):
        return self._get_log("behavior_log")

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
            return pd.read_csv(str(file), delimiter=";").drop("Unnamed: 0", axis=1)
        elif file.suffix == ".h5" or file.suffix == ".hdf5":
            return pd.read_hdf(file)
        elif file.suffix == ".feather":
            return pd.read_feather(file)
        elif file.suffix == ".json":
            return pd.read_json(file)
        else:
            raise ValueError(
                str(data_name) + " format is not supported, trying to load " + str(file)
            )

    def stimulus_starts_ends(self):
        """ Get start and end time of all stimuli in the log
        :return: arrays with start and end times for all stimuli
        """
        starts = np.array([stim["t_start"] for stim in self["stimulus"]["log"]])
        ends = np.array([stim["t_stop"] for stim in self["stimulus"]["log"]])
        return starts, ends

    def copy_to_dir(self, target_dir):
        """ Copy all the files pertaining to this experiment in a target
        directory. If it does not exist, make it.
        """
        target_dir = Path(target_dir)

        if target_dir != self.root:  # Make sure we don't overwrite
            if not target_dir.exists():
                target_dir.mkdir()

            for f in self.file_list:
                copyfile(f, target_dir / f.name)

    def bouts(self):
        raise


class EmbeddedExperiment(Experiment):
    def vigor(self, vigor_duration=0.05):
        """ Get vigor, the proxy of embedded fish forward velocity,
        a standard deviation calculated on a rolling window of tail curvature

        :param vigor_duration: standar deviation window length in seconds
        :return:
        """
        if vigor in self.behavior_log.columns:
            return self.behavior_log["vigor"]

        dt = log_dt(self.behavior_log)
        vigor_win = int(vigor_duration / dt)
        self.behavior_log["vigor"] = (
            self.behavior_log["tail_sum"]
            .iterpolate()
            .rolling(vigor_win, center=True)
            .std()
        )
        return self.behavior_log["vigor"]


class MultiSessionExperiment(Experiment):
    """ Class to handle the scenario of multiple stytra sessions within the
    same experiment - typically, for plane-wise repetitions in 2p imaging.
    """
    def __init__(self, path):
        self.session_list = sorted(list(path.glob("*_metadata.json")))
        super().__init__(self.session_list[0])

        self.session_id_list = []
        for i_meta in range(len(self.session_list)):
            self.session_id_list += [str(self.session_list[i_meta].name).split("_")[0]]

        for log_name in ['behavior_log', 'stimulus_param_log', 'estimator_log']:
            for possible_name in self.log_mapping[log_name]:
                logfnames = list(self.root.glob("*_" + possible_name + ".*"))
                if len(logfnames) > 0:
                    self.log_mapping[log_name] = possible_name + logfnames[0].suffix

    def _get_log(self, log_name):
        """ Given name of the log get it from attributes or load it ex novo
        :param log_name:  string with the type ot the log to load
        :return:  loaded log DataFrame
        """
        uname = "_" + log_name

        # We will concatenate all logs in a single one using correct timestamps from first
        # session start time:
        timestarts = self.session_start_tstamps()
        timedeltas = [(s - timestarts[0]).total_seconds() for s in timestarts]

        # Check whether this was already set:
        if getattr(self, uname) is None:
            # If not, loop over different possibilities for that filename (from old stytra versions)
            # print(self.log_mapping[log_name])
            for possible_name in self.log_mapping[log_name]:
                # TODO not sure what this is handling:
                # if possible_name[:3] == "key":
                #    try:
                #        data = self
                #        path = possible_name[4:].split("/")
                #        for i in range(0, len(path)):
                #            data = data[path[i]]
                #        setattr(self, uname, pd.DataFrame(data))
                #        break
                #    except KeyError:
                #        pass
                # else:
                try:
                    # Load all dataframes:
                    logfnames = list(
                        self.root.glob("*_" + possible_name + ".*"))
                    all_logs = []
                    k_idx = 0
                    for logfname, dt in zip(logfnames, timedeltas):
                        df = self._load_log(logfname.name)

                        # Correct index for concatenation and compute time from exp start:
                        df.index += k_idx
                        df["t"] += dt
                        k_idx += len(df)

                        all_logs.append(df)

                    df = pd.concat(all_logs)
                    setattr(self, uname, df)
                    break
                except (ValueError, UnboundLocalError):
                    pass
            else:
                raise ValueError(log_name + " does not exist")

        return getattr(self, uname)

    def session_start_tstamps(self):
        """ Return timestamps for all the sessions in this folder.
        """
        start_tstamps = []
        for p in self.session_list:
            s = Experiment(p)["general"]["t_protocol_start"]
            start_tstamps.append(datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f"))
        return start_tstamps

    def get_session_log(self, log_name, session_idx):
        """ Function to load a log for a single session, specified by index-
        :param log_name: string specifying the type of log (e.g., "behavior_log")
        :param session_idx: int, index of session to load:
        :return:
        """
        log_name = self.log_mapping[log_name]
        session_log = self.root / str(self.session_id_list[session_idx] + "_" + log_name)
        return self._load_log(session_log)


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
