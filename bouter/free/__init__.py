from bouter import Experiment
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


    def get_scale_mm(self):
        """ Return camera pixel size in millimeters

        :param exp:
        :return:
        """
        cal_params = self["stimulus"]["calibration_params"]
        if self["general"]["animal"]["embedded"]:
            return cal_params["mm_px"]
        else:
            proj_mat = np.array(cal_params["cam_to_proj"])
            return (
                np.linalg.norm(np.array([1.0, 0.0]) @ proj_mat[:, :2]) * cal_params["mm_px"]
            )
