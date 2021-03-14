import pandas as pd

from bouter.bout_stats import bout_stats, count_peaks_between
from bouter.decorators import cache_results
from bouter.experiment import Experiment
from bouter.utilities import (
    bandpass,
    extract_segments_above_threshold,
    fill_out_segments,
    polynomial_tail_coefficients,
    polynomial_tailsum,
    revert_segment_filling,
)


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

    @cache_results(cache_filename="behavior_log")
    def reconstruct_missing_segments(self, continue_curvature=None):

        segments = self.behavior_log.loc[:, self.tail_columns].values.copy()

        if "missing_n" in self.behavior_log.columns:
            revert_pts = self.behavior_log["missing_n"].values
        else:
            revert_pts = None

        # Revert if possible if continue_curvature is None:
        if continue_curvature is None:
            if revert_pts is not None:
                fixed_segments = revert_segment_filling(
                    segments,
                    revert_pts=revert_pts,
                )
                self.behavior_log.loc[:, self.tail_columns] = fixed_segments

        # Otherwise, use the parameter to do the filling:
        else:
            fixed_segments, missing_n = fill_out_segments(
                segments,
                continue_curvature=continue_curvature,
                revert_pts=revert_pts,
            )
            self.behavior_log.loc[:, self.tail_columns] = fixed_segments
            self.behavior_log["missing_n"] = missing_n

        return self.behavior_log

    @cache_results()
    def polynomial_tail_coefficients(self, n_max_missing=7, degree=3):
        """Fits a polynomial to the bout shape

        :param n_max_missing:
        :param degree: the polynomial degree
        :return:
        """
        segments = self.behavior_log.loc[:, self.tail_columns].values
        poly_coefs = polynomial_tail_coefficients(
            segments, n_max_missing=n_max_missing, degree=degree
        )
        return poly_coefs

    @cache_results()
    def polynomial_tailsum(self):
        return polynomial_tailsum(self.polynomial_tail_coefficients())

    @cache_results(cache_filename="behavior_log")
    def compute_vigor(
        self, vigor_duration_s=0.05, use_polynomial_tailsum=False, **kwargs
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

    @cache_results()
    def get_bouts(self, vigor_threshold=0.1, **kwargs):
        """Extract bouts above threshold.
        :param vigor_threshold:
        :return:
        """
        # Make sure there's a vigor column:
        self.compute_vigor()
        bouts, _ = extract_segments_above_threshold(
            self.behavior_log["vigor"].values, vigor_threshold
        )

        return bouts

    @cache_results()
    def get_bout_properties(
        self,
        directionality_duration=0.07,
        use_polynomial_tailsum=False,
        **kwargs,
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
            use_polynomial_tailsum=use_polynomial_tailsum, **kwargs
        )["vigor"].values
        bouts = self.get_bouts(**kwargs)

        if bouts.shape[0] == 0:
            return pd.DataFrame(
                dict(
                    t_start=[],
                    duration=[],
                    peak_vig=[],
                    med_vig=[],
                    bias=[],
                    bias_total=[],
                    n_pos_peaks=[],
                    n_neg_peaks=[],
                )
            )

        peak_vig, med_vig, bias, bias_tot = bout_stats(
            vigor, tail_sum, bouts, bout_init_window_pts
        )
        n_pos_peaks, n_neg_peaks = count_peaks_between(
            bandpass(tail_sum, self.behavior_dt),
            bouts[:, 0],
            bouts[:, 1],
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
                bias_total=bias_tot,
                n_pos_peaks=n_pos_peaks,
                n_neg_peaks=n_neg_peaks,
            )
        )
