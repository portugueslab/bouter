from bouter import Experiment
from bouter import utilities, get_scale_mm
import numpy as np


class FreelySwimmingExperiment(Experiment):

    def get_n_segments(self, prefix=True):
        if prefix:

            def _tail_part(s):
                ps = s.split("_")
                if len(ps) == 3:
                    return ps[2]
                else:
                    return 0

        else:

            def _tail_part(s):
                ps = s.split("_")
                if len(ps) == 2:
                    return ps[1]
                else:
                    return 0

        tpfn = np.vectorize(_tail_part, otypes=[int])
        return np.max(tpfn(self.behavior_log.columns.values)) + 1


    def get_n_fish(self):
        def _fish_part(s):
            ps = s.split("_")
            if len(ps) == 3:
                return ps[0][1:]
            else:
                return 0

        tpfn = np.vectorize(_fish_part, otypes=[int])
        return np.max(tpfn(self.behavior_log.columns.values)) + 1


    def _extract_bout(self, s, e, n_segments, i_fish=0, scale=1.0, dt=None):
        bout = self._rename_fish(self.behavior_log.iloc[s:e], i_fish, n_segments)
        # scale to physical coordinates
        if dt is None:
            dt = (bout.t.values[-1] - bout.t.values[0]) / bout.shape[0]

        # pixels are scaled to millimeters (columns x, vx, y and vy)
        bout.iloc[:, 1:5] *= scale
        # velocities are additionally divided by the time difference to get mm/s
        bout.iloc[:, 2:7:2] /= dt
        return bout


    def extract_bouts(
        self,
        max_interpolate=2,
        window_size=7,
        recalculate_vel=False,
        median_vel=False,
        scale=None,
        threshold=1,
        **kwargs
    ):
        """ Extracts all bouts from a freely-swimming tracking experiment

        :param exp: the experiment object
        :param max_interpolate: number of points to interpolate if surrounded by NaNs in tracking
        :param scale: mm per pixel, recalculated by default
        :param max_frames: the maximum numbers of frames to process, useful for debugging
        :param threshold: velocity threshold in mm/s
        :param min_duration: minimal number of frames for a bout
        :param pad_before: number of frames that gets added before
        :param pad_after: number of frames added after

        :return: tuple: (list of single bout dataframes, list of boolean arrays marking if the
         bout i follows bout i-1)
        """

        df = self.behavior_log

        scale = scale or get_scale_mm(self)

        dt = np.mean(np.diff(df.t[100:200]))

        n_fish = self.get_n_fish()
        n_segments = self.get_n_segments()
        dfint = df.interpolate("linear", limit=max_interpolate, limit_area="inside")
        bouts = []
        continuous = []
        for i_fish in range(n_fish):
            if recalculate_vel:
                for thing in ["x", "y", "theta"]:
                    dfint["f{}_v{}".format(i_fish, thing)] = np.r_[
                        np.diff(dfint["f{}_{}".format(i_fish, thing)]), 0
                    ]

            vel2 = (
                dfint["f{}_vx".format(i_fish)] ** 2 + dfint["f{}_vy".format(i_fish)] ** 2
            ) * ((scale / dt) ** 2)
            if median_vel:
                vel2 = vel2.rolling(window=window_size, min_periods=1).median()
            bout_locations, continuity = utilities.extract_segments_above_threshold(
                vel2.values, threshold=threshold ** 2, **kwargs
            )
            all_bouts_fish = [
                self._extract_bout(s, e, n_segments, i_fish, scale)
                for s, e in bout_locations
            ]
            bouts.append(all_bouts_fish)
            continuous.append(np.array(continuity))

        return bouts, continuous


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
                "f{:d}_theta_{:02d}".format(i_fish, i): "theta_{:02d}".format(i)
                for i in range(n_segments)
            }
        )


    def _rename_fish(self, df, i_fish, n_segments):
        return df.filter(["t"] + self._fish_column_names(i_fish, n_segments)).rename(
            columns=self._fish_renames(i_fish, n_segments)
        )

