import numpy as np
import pandas as pd

from bouter import utilities, decorators, bout_stats
from bouter import Experiment


class EmbeddedExperiment(Experiment):
    @property
    def n_tail_segments(self):
        try:
            return self["behavior"]["tail"]["n_segments"]
        except KeyError:
            return self["tracking+tail_tracking"]["n_output_segments"]

    @property
    def tail_columns(self):
        """Return names of columns with tracking data from all tracked segments.
        Careful, the array is not copied!
        """
        return [f"theta_{i:02}" for i in range(self.n_tail_segments)]

    @decorators.cache_results(cache_filename="behavior_log")
    def reconstruct_missing_segments(self, continue_curvature=None):

        segments = self.behavior_log.loc[:, self.tail_columns].values.copy()

        if "missing_n" in self.behavior_log.columns:
            revert_pts = self.behavior_log["missing_n"].values
        else:
            revert_pts = None

        # Revert if possible if continue_curvature is None:
        if continue_curvature is None:
            if revert_pts is not None:
                fixed_segments = utilities.revert_segment_filling(
                    segments, revert_pts=revert_pts,
                )
                self.behavior_log.loc[:, self.tail_columns] = fixed_segments

        # Otherwise, use the parameter to do the filling:
        else:
            fixed_segments, missing_n = utilities.fill_out_segments(
                segments,
                continue_curvature=continue_curvature,
                revert_pts=revert_pts,
            )
            self.behavior_log.loc[:, self.tail_columns] = fixed_segments
            self.behavior_log["missing_n"] = missing_n

        return self.behavior_log

    @decorators.cache_results()
    def polynomial_tail_coefficients(self, n_max_missing=7, degree=3):
        """ Fits a polynomial to the bout shape

        :param n_max_missing:
        :param degree: the polynomial degree
        :return:
        """
        segments = self.behavior_log.loc[:, self.tail_columns].values
        segments -= segments[:, 0:1]
        n_max_missing = min(self.n_tail_segments - degree, n_max_missing)

        # the Stytra tail tracking introduces NaNs at breaking point
        # a situation number - NaN - number never occurs in tracking
        n_missing = utilities.n_missing_segments(segments)

        poly_coefs = np.zeros((segments.shape[0], degree + 1))
        line_points = np.linspace(0, 1, self.n_tail_segments)

        for i_missing in range(n_max_missing + 1):
            sel_time = n_missing == i_missing
            poly_coefs[sel_time, :] = np.polynomial.polynomial.polyfit(
                line_points[0 : self.n_tail_segments - i_missing],
                segments[sel_time, 0 : self.n_tail_segments - i_missing].T,
                degree,
            ).T
        return poly_coefs

    @decorators.cache_results()
    def polynomial_tailsum(self):
        return np.polynomial.polynomial.polyval(
            1, self.polynomial_tail_coefficients().T, False
        )

    @decorators.cache_results(cache_filename="behavior_log")
    def compute_vigor(
        self, vigor_duration_s=0.05, use_polynomial_tailsum=False
    ):
        """Compute vigor, the proxy of embedded fish forward velocity,
        a standard deviation calculated on a rolling window of tail curvature.
        Add it as a column to the dataframe log and return the full dataframe

        :param vigor_duration: standard deviation window length in seconds
        :return:
        """
        vigor_win = int(vigor_duration_s / self.behavior_dt)
        tailsum = (
            pd.Series(self.polynomial_tailsum())
            if use_polynomial_tailsum
            else self.behavior_log["tail_sum"]
        )
        self.behavior_log["vigor"] = (
            tailsum.interpolate().rolling(vigor_win, center=True).std()
        )
        return self.behavior_log

    @decorators.cache_results()
    def get_bouts(self, vigor_threshold=0.1):
        """Extract bouts above threshold.
        :param vigor_threshold:
        :return:
        """
        # Make sure there's a vigor column:
        self.compute_vigor()
        bouts, _ = utilities.extract_segments_above_threshold(
            self.behavior_log["vigor"].values, vigor_threshold
        )

        return bouts

    @decorators.cache_results()
    def get_bout_properties(
        self, directionality_duration=0.07, use_polynomial_tailsum=False,
    ):
        """Create dataframe with summary of bouts properties.
        :param directionality_duration: Window defining initial part of
            the bout for the turning angle calculation, in seconds.
        :param use_polynomial_tailsum: If the polynomial tail sum is to be used
            instead of the raw one created by Stytra
        :return: a dataframe giving properties for each bout
        """
        bout_init_window_pts = int(directionality_duration / self.behavior_dt)
        tail_sum = (
            self.polynomial_tailsum()
            if use_polynomial_tailsum
            else self.behavior_log["tail_sum"].values
        )
        vigor = self.compute_vigor(
            use_polynomial_tailsum=use_polynomial_tailsum
        ).values
        bouts = self.get_bouts()
        peak_vig, med_vig, bias, bias_tot = bout_stats.bout_stats(
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
                bias=bias,
                bias_tot=bias_tot,
            )
        )
