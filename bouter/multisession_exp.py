import json
from datetime import datetime

import pandas as pd

from bouter.embedded import EmbeddedExperiment


class MultiSessionExperiment(EmbeddedExperiment):
    """Class to handle the scenario of multiple stytra sessions within the
    same experiment - typically, for plane-wise repetitions in 2p imaging.
    """

    def __init__(self, path, **kwargs):
        self.session_list = sorted(list(path.glob("*_metadata.json")))

        session_id_list = []
        session_start = []
        for i_meta in range(len(self.session_list)):
            session_id_list += [
                str(self.session_list[i_meta].name).split("_")[0]
            ]
            metadata_file = self.session_list[i_meta]
            with open(str(metadata_file), "r") as f:
                source_metadata = json.load(f)
            session_start += [source_metadata["general"]["t_protocol_start"]]

        # sorting the session_id list
        self.session_id_list = [
            x for _, x in sorted(zip(session_start, session_id_list))
        ]
        self.session_list = [
            x for _, x in sorted(zip(session_start, self.session_list))
        ]

        super().__init__(self.session_list[0])

        self.experiments = [
            EmbeddedExperiment(pth, **kwargs) for pth in self.session_list
        ]

    def _get_log(self, log_name):
        """Given name of the log get it from attributes or load it ex novo
        :param log_name:  string with the type ot the log to load
        :return:  loaded log DataFrame
        """
        uname = "_" + log_name

        # We will concatenate all logs in a single one using correct timestamps
        # from first session start time:
        timestarts = self.session_start_tstamps()
        timedeltas = [(s - timestarts[0]).total_seconds() for s in timestarts]

        # Check whether this was already set:
        if getattr(self, uname) is None:
            # If not, loop over different possibilities for that filename
            # (from old stytra versions)
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
                        self.root.glob("*_" + possible_name + ".*")
                    )
                    all_logs = []
                    k_idx = 0
                    for logfname, dt in zip(logfnames, timedeltas):
                        df = self._load_log(logfname.name)

                        # Correct index for concatenation and compute time from
                        # exp start:
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
        """Return timestamps for all the sessions in this folder."""
        start_tstamps = []
        for exp in self.experiments:
            s = exp["general"]["t_protocol_start"]
            start_tstamps.append(datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f"))
        return start_tstamps

    def load_session_log(self, log_name, session_idx):
        """Function to load a log for a single session, specified by index-
        :param log_name: string specifying the type of log (e.g., "behavior_log"
        :param session_idx: int, index of session to load
        :return:
        """
        session_log = self.root / str(
            self.session_id_list[session_idx] + "_" + log_name + ".hdf5"
        )
        return self._load_log(session_log)
