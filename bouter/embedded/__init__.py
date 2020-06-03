import numpy as np

from bouter import utilities
from bouter import Experiment
from bouter import decorators


class EmbeddedExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def n_tail_segments(self):
        return self["behavior"]["tail"]["n_segments"]

    @property
    def tail_points_matrix(self):
        """Return matrix with the tail points.
        Careful, the array is not copied.
        """
        columns = [f"theta_{i:02}" for i in range(self.n_tail_segments)]
        return self.behavior_log.loc[:, columns].values

    @decorators.method_caching
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

    def reconstruct_missing_segments(self, continue_curvature=0):
        """ If the tail tip is not tracked throught the whole experiment
        reconstruct the tail sum from the segments that are

        """
        utilities.fill_out_segments(
            self.tail_points_matrix, continue_curvature=continue_curvature
        )

    @decorators.method_caching
    def bouts(self, vigor_duration=0.05, vigor_threshold=0.1):
        """Extract bouts above threshold.
        :param vigor_threshold:
        :return:
        """
        vigor = self.vigor()
        return utilities.extract_segments_above_threshold(
            vigor.values, vigor_threshold
        )

    def bout_summary(self):
        pass
