from bouter.core_exp import Experiment
from bouter import utilities


class EmbeddedExperiment(Experiment):
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
