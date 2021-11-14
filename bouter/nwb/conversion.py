import numpy as np

from functools import singledispatch
from datetime import datetime
from pathlib import Path

from bouter import Experiment
from bouter.free import FreelySwimmingExperiment
from bouter.embedded import EmbeddedExperiment
from pynwb import NWBFile, NWBHDF5IO
from pynwb.misc import AbstractFeatureSeries


def _save_stimulus(exp: Experiment, nwbfile: NWBFile):
    """Converts the stimulus log into two time NWB time series, one
    for the floating point and the other for integer values"""
    float_columns = []
    int_columns = []
    for col in exp.stimulus_log.columns:
        if np.issubdtype(exp.stimulus_log[col].dtype, np.floating):
            float_columns.append(col)
        elif np.issubdtype(exp.stimulus_log[col].dtype, np.integer):
            int_columns.append(col)
        else:
            raise NotImplementedError(
                f"Serialization of {exp.stimulus_log[col].dtype}"
                f"into NWB not supported for column {col} of the stimulus log."
            )
    nontime_float_columns = [col for col in float_columns if col != "t"]
    stimulus_features_float = AbstractFeatureSeries(
        name="stimulus_features_int",
        features=nontime_float_columns,
        feature_units=["" for _ in nontime_float_columns],
        data=exp.stimulus_log.loc[:, nontime_float_columns].values,
        timestamps=exp.stimulus_log.t.values,
    )

    stimulus_features_int = AbstractFeatureSeries(
        name="stimlus_features_float",
        features=int_columns,
        feature_units=["" for _ in nontime_float_columns],
        data=exp.stimulus_log.loc[:, nontime_float_columns].values,
        timestamps=stimulus_features_float,
    )
    nwbfile.add_stimulus(stimulus_features_float)
    nwbfile.add_stimulus(stimulus_features_int)


@singledispatch
def _save_behavior(exp: Experiment, nwbfile: NWBFile):
    pass


@_save_behavior.register
def _(exp: FreelySwimmingExperiment, nwbfile: NWBFile):
    raise NotImplementedError("Freely-swimming experiment serialization not yet implemented")


@_save_behavior.register
def _(exp: EmbeddedExperiment, nwbfile: NWBFile):
    raise NotImplementedError("Freely-swimming experiment serialization not yet implemented")


def experiment_to_nwb(exp: Experiment, nwb_path: Path):
    nwbfile = NWBFile(
        session_description="demonstrate adding to an NWB file",
        identifier=exp.full_name,
        session_start_time=datetime.fromisoformat(exp["general"]["t_protocol_start"]),
    )

    _save_stimulus(exp, nwbfile)

    nwbfile.save()

    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)
