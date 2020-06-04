import numpy as np
import pandas as pd

from bouter import utilities, decorators, bout_stats
from bouter import Experiment


class EmbeddedExperiment(Experiment):
    def __init__(self, *args, continue_curvature=None, **kwargs):
        super().__init__(*args, **kwargs)

        if continue_curvature is not None:
            self.tail_points_matrix, missing_n = utilities.fill_out_segments(
                self.tail_points_matrix.copy(),
                continue_curvature=continue_curvature,
            )

            self.behavior_log["missing_n"] = missing_n

    @property
    def n_tail_segments(self):
        return self["behavior"]["tail"]["n_segments"]

    @property
    def tail_points_matrix(self):
        """Return matrix with the tail points.
        Careful, the array is not copied!
        """
        columns = [f"theta_{i:02}" for i in range(self.n_tail_segments)]
        return self.behavior_log.loc[:, columns].values

    @tail_points_matrix.setter
    def tail_points_matrix(self, matrix):
        """Rewrite tail points in the tail dataframe
        """
        columns = [f"theta_{i:02}" for i in range(self.n_tail_segments)]
        self.behavior_log.loc[:, columns] = matrix

    @decorators.cache_results
    def vigor(self, vigor_duration_s=0.05):
        """ Get vigor, the proxy of embedded fish forward velocity,
        a standard deviation calculated on a rolling window of tail curvature.

        :param vigor_duration: standard deviation window length in seconds
        :return:
        """
        # TODO split in two methods so that it is set in the log df when
        # loaded from cache (alternatively never write there)
        if "vigor" in self.behavior_log.columns:
            return self.behavior_log["vigor"]

        vigor_win = int(vigor_duration_s / self.behavior_dt)
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

    @decorators.cache_results
    def bouts(self, vigor_threshold=0.1):
        """Extract bouts above threshold.
        :param vigor_threshold:
        :return:
        """
        vigor = self.vigor()
        bouts, _ = utilities.extract_segments_above_threshold(
            vigor.values, vigor_threshold
        )

        return bouts

    @decorators.cache_results
    def bout_properties(self, bout_init_window_s=0.07):
        """Create dataframe with summary of bouts properties.
        :param bout_init_window_s: Window defining initial part of
            the bout for the turning angle calculation, in seconds.
        :return:
        """
        bout_init_window_pts = int(bout_init_window_s / self.behavior_dt)
        tail_sum = self.behavior_log["tail_sum"].values
        vigor = self.vigor().values
        bouts = self.bouts()
        peak_vig, med_vig, ang_turn, ang_turn_tot = bout_stats.bout_stats(
            vigor, tail_sum, bouts, bout_init_window_pts
        )

        t_array = self.behavior_log["t"].values
        t_start, t_end = [t_array[bouts[:, i]] for i in range(2)]
        return pd.DataFrame(
            dict(
                t_start=t_start,
                duration=t_end - t_start,
                peak_vig=peak_vig,
                med_vig=med_vig,
                ang_turn=ang_turn,
                ang_turn_tot=ang_turn_tot,
            )
        )
