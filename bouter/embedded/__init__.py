import flammkuchen as fl
import numpy as np

import bouter.embedded.utilities
from bouter import Experiment


class EmbeddedExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def vigor(self, vigor_duration=0.05):
        """ Get vigor, the proxy of embedded fish forward velocity,
        a standard deviation calculated on a rolling window of tail curvature.

        :param vigor_duration: standard deviation window length in seconds
        :return:
        """
        if "vigor" in self.behavior_log.columns:
            return self.behavior_log["vigor"]

        dt = utilities.log_dt(self.behavior_log)
        vigor_win = int(vigor_duration / dt)
        self.behavior_log["vigor"] = (
            self.behavior_log["tail_sum"]
            .interpolate()
            .rolling(vigor_win, center=True)
            .std()
        )
        return self.behavior_log["vigor"]

    def reconstruct_missing_segments(self):
        """ If the tail tip is not tracked throught the whole experiment
        reconstruct the tail sum from the segments that are

        """
        pass

    def bouts(self, vigor_threshold=0.1):
        bout_path = self.root / self.session_id + "_bouts.h5"
        if bout_path.is_file():
            return fl.load(bout_path)
        vigor = self.vigor()
        bout_starts_ends, cont = utilities.extract_segments_above_threshold(
            vigor, vigor_threshold
        )
        fl.save(bout_path, np.array(bout_starts_ends))

    def bout_summary(self):
        pass
