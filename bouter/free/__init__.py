import numpy as np
import pandas as pd

from bouter import Experiment, decorators, utilities


class FreelySwimmingExperiment(Experiment):
    @property
    def n_tail_segments(self):
        return self["tracking+fish_tracking"]["n_segments"] - 1

    @property
    def n_fish(self):
        return self["tracking+fish_tracking"]["n_fish_max"]

    @property
    def camera_px_in_mm(self):
        """Return camera pixel size in millimeters

        :param exp:
        :return:
        """
        cal_params = self["stimulus"]["calibration_params"]
        proj_mat = np.array(cal_params["cam_to_proj"])
        return (
            np.linalg.norm(np.array([1.0, 0.0]) @ proj_mat[:, :2])
            * cal_params["mm_px"]
        )

    @property
    def tail_columns(self):
        """Return a nested list of names of columns with tracking data from all tracked segments.
        One list for each fish tracked during the experiment.
        """
        return [
            [f"f{i}_theta_{j:02}" for j in range(self.n_tail_segments)]
            for i in range(self.n_fish)
        ]

    def _extract_bout(self, s, e, n_segments, i_fish=0, scale=1.0, dt=None):
        bout = self._rename_fish(
            self.behavior_log.iloc[s:e], i_fish, n_segments
        )
        # scale to physical coordinates
        if dt is None:
            dt = (bout.t.values[-1] - bout.t.values[0]) / bout.shape[0]

        # pixels are scaled to millimeters (columns x, vx, y and vy)
        bout.iloc[:, 1:5] *= scale
        # velocities are additionally divided by the time difference to get mm/s
        bout.iloc[:, 2:7:2] /= dt
        return bout

    def _fish_column_names(self, i_fish, n_segments):
        return [
            "f{:d}_x".format(i_fish),
            "f{:d}_vx".format(i_fish),
            "f{:d}_y".format(i_fish),
            "f{:d}_vy".format(i_fish),
            "f{:d}_theta".format(i_fish),
            "f{:d}_vtheta".format(i_fish),
        ] + ["f{:d}_theta_{:02d}".format(i_fish, i) for i in range(n_segments)]

    def _fish_renames(self, i_fish, n_segments):
        return dict(
            {
                "f{:d}_x".format(i_fish): "x",
                "f{:d}_vx".format(i_fish): "vx",
                "f{:d}_y".format(i_fish): "y",
                "f{:d}_vy".format(i_fish): "vy",
                "f{:d}_theta".format(i_fish): "theta",
                "f{:d}_vtheta".format(i_fish): "vtheta",
            },
            **{
                "f{:d}_theta_{:02d}".format(i_fish, i): "theta_{:02d}".format(
                    i
                )
                for i in range(n_segments)
            },
        )

    def _rename_fish(self, df, i_fish, n_segments):
        return df.filter(
            ["t"] + self._fish_column_names(i_fish, n_segments)
        ).rename(columns=self._fish_renames(i_fish, n_segments))

    def compute_velocity(
        self,
        max_interpolate=2,
        recalculate_vel=False,
        scale=None,
        median_vel=False,
        window_size=7,
    ):
        """Compute the squared total swimming velocity for each fish.
        Add them as new columns to the dataframe log and return the complete dataframe.

        :param max_interpolate: number of points to interpolate if surrounded by NaNs in tracking
        :param recalculate_vel:
        :param scale: mm per pixel, recalculated by default
        :return:
        """
        df = self.behavior_log
        scale = scale or self.camera_px_in_mm
        dt = np.mean(np.diff(df.t[100:200]))

        dfint = df.interpolate(
            "linear", limit=max_interpolate, limit_area="inside"
        )

        fish_velocities = pd.DataFrame(
            np.nan,
            index=self.behavior_log.index,
            columns=[
                "vel2_f{}".format(i_fish) for i_fish in range(self.n_fish)
            ],
        )

        for i_fish in range(self.n_fish):
            if recalculate_vel:
                for thing in ["x", "y", "theta"]:
                    dfint["f{}_v{}".format(i_fish, thing)] = np.r_[
                        np.diff(dfint["f{}_{}".format(i_fish, thing)]), 0
                    ]

            vel2 = (
                dfint["f{}_vx".format(i_fish)] ** 2
                + dfint["f{}_vy".format(i_fish)] ** 2
            ) * ((scale / dt) ** 2)

            if median_vel:
                vel2 = vel2.rolling(window=window_size, min_periods=1).median()

            fish_velocities["vel2_f{}".format(i_fish)] = vel2

        return fish_velocities

    @decorators.cache_results()
    def get_bouts(self, scale=None, threshold=1, **kwargs):
        """Extracts all bouts from a freely-swimming tracking experiment

        :param exp: the experiment object
        :param scale: mm per pixel, recalculated by default
        :param threshold: velocity threshold in mm/s
        :return: tuple: (list of single bout dataframes, list of boolean arrays marking if the
         bout i follows bout i-1)
        """
        n_fish = self.n_fish
        n_segments = self.n_tail_segments
        scale = scale or self.camera_px_in_mm

        fish_velocities = self.compute_velocity()

        bouts = []
        continuous = []

        for i_fish in range(n_fish):
            vel2 = fish_velocities["vel2_f{}".format(i_fish)]
            (
                bout_locations,
                continuity,
            ) = utilities.extract_segments_above_threshold(
                vel2.values, threshold=threshold ** 2, **kwargs
            )
            all_bouts_fish = [
                self._extract_bout(s, e, n_segments, i_fish, scale)
                for s, e in bout_locations
            ]
            bouts.append(all_bouts_fish)
            continuous.append(np.array(continuity))

        return bouts, continuous

    @decorators.cache_results()
    def get_bout_properties(self, continuity=None):
        """Makes a summary of all extracted bouts with basic kinematic parameters and timing.

        :param continuity:
        :return: a dataframe containing all bouts
        """
        headers = [
            "t_start",
            "x_start",
            "y_start",
            "theta_start",
            "t_end",
            "x_end",
            "y_end",
            "theta_end",
        ]

        # Extract experiment bouts
        bouts, _ = self.get_bouts()

        # an array is preallocated loop through the bouts
        bout_data = np.empty(
            (np.sum([len(bouts[i]) for i in range(len(bouts))]), len(headers))
        )
        n_summarized_bouts = 0
        for i_fish in range(len(bouts)):
            for i_bout, bout in enumerate(bouts[i_fish]):
                # slices from 0 to 4 are the start parameters, from 4 to 8 the end parameters
                for sl, idx in zip([slice(0, 4), slice(4, 8)], [0, -1]):
                    bout_data[n_summarized_bouts + i_bout, sl] = [
                        bout.t.iloc[idx],
                        bout.x.iloc[idx],
                        bout.y.iloc[idx],
                        bout.theta.iloc[idx],
                    ]
            n_summarized_bouts += len(bouts[i_fish])

        bout_data_df = pd.DataFrame(bout_data, columns=headers)
        if continuity:
            bout_data_df["follows_previous"] = np.concatenate(continuity)

        # if there are multiple fish tracked in the same experiments, assign the
        # identities (there is no guarantee that the identity will be consistent if the
        # fish cross or go outside of the visible region)
        if len(bouts) > 1:
            origin_fish = np.concatenate(
                [
                    np.full(len(bouts[i]), i, dtype=np.uint8)
                    for i in range(len(bouts))
                ]
            )
            bout_data_df.insert(0, "i_fish", origin_fish)

        return bout_data_df

    @decorators.cache_results(cache_filename="behavior_log")
    def reconstruct_missing_segments(self, continue_curvature=None):

        for i_fish in range(self.n_fish):

            segments = self.behavior_log.loc[
                :, self.tail_columns[i_fish]
            ].values.copy()

            if "f{}_missing_n".format(i_fish) in self.behavior_log.columns:
                revert_pts = self.behavior_log[
                    "f{}_missing_n".format(i_fish)
                ].values
            else:
                revert_pts = None

            # Revert if possible if continue_curvature is None:
            if continue_curvature is None:
                if revert_pts is not None:
                    fixed_segments = utilities.revert_segment_filling(
                        segments,
                        revert_pts=revert_pts,
                    )
                    self.behavior_log.loc[
                        :, self.tail_columns[i_fish]
                    ] = fixed_segments

            # Otherwise, use the parameter to do the filling:
            else:
                fixed_segments, missing_n = utilities.fill_out_segments(
                    segments,
                    continue_curvature=continue_curvature,
                    revert_pts=revert_pts,
                )
                self.behavior_log.loc[
                    :, self.tail_columns[i_fish]
                ] = fixed_segments
                self.behavior_log["f{}_missing_n".format(i_fish)] = missing_n

        return self.behavior_log
