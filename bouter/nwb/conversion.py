import json
from copy import deepcopy
from datetime import datetime
from functools import singledispatch
from pathlib import Path

import numpy as np
import pynwb.file
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.misc import AbstractFeatureSeries

from bouter import Experiment
from bouter.embedded import EmbeddedExperiment
from bouter.free import FreelySwimmingExperiment


def _get_subject_metadata(exp: Experiment):
    """Converts the metadata about the animal into
    NWB subject metadata.
    """
    return pynwb.file.Subject(
        age=f'P{exp["general"]["animal"]["age"]}D',
        description=exp["general"]["animal"]["comments"],
        genotype=exp["general"]["animal"]["genotype"],
        species=exp["general"]["animal"]["species"],
        subject_id=exp["general"]["animal"]["id"],
    )


def _save_stimulus(exp: Experiment, nwbfile: NWBFile):
    """Converts the stimulus log into two time NWB time series, one
    for the floating point and the other for integer values.

    The stimuli in Stytra have two kinds of properties: static
    ones determined at stimulus creation and dynamic ones that
    depend on the behavior of the animal. The stimuli therefore
    have to have two kinds of serialization, the TimeSeries-
    based one (for the dynamic properties), and the other with

    """
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

    stimulus_intervals = pynwb.epoch.TimeIntervals(
        name="stimulus_log", description="JSON-serialized stimulus log"
    )

    stimulus_intervals.add_column(
        name="stimulus_name", description="stimulus name"
    )
    stimulus_intervals.add_column(
        name="stimulus_data", description="JSON-encoded stimulus data"
    )

    for stim in exp["stimulus"]["log"]:
        stim_copy = deepcopy(stim)
        t_start = stim_copy.pop("t_start")
        t_stop = stim_copy.pop("t_stop")
        stim_name = stim_copy.pop("name")
        stimulus_intervals.add_interval(
            t_start,
            t_stop,
            stimulus_name=stim_name,
            stimulus_data=json.dumps(stim_copy),
        )

    nwbfile.add_time_intervals(stimulus_intervals)


@singledispatch
def _save_behavior(exp: Experiment, nwbfile: NWBFile):
    pass


@_save_behavior.register
def _(exp: FreelySwimmingExperiment, nwbfile: NWBFile):
    raise NotImplementedError(
        "Freely-swimming experiment serialization not yet implemented"
    )


@_save_behavior.register
def _(exp: EmbeddedExperiment, nwbfile: NWBFile):
    tail_shapes = TimeSeries(
        "tail_shape",
        timestamps=exp.behavior_log.t.values,
        data=exp.behavior_log.loc[:, exp.tail_columns].values,
        unit="radians",
    )
    nwbfile.add_acquisition(tail_shapes)


def experiment_to_nwb(exp: Experiment, nwb_path: Path):
    """Convert a bouter Experiment into a nwb file."""
    nwbfile = NWBFile(
        session_description="demonstrate adding to an NWB file",
        identifier=exp.full_name,
        session_start_time=datetime.fromisoformat(
            exp["general"]["t_protocol_start"]
        ),
        subject=_get_subject_metadata(exp),
        experimenter=exp["general"]["basic"]["experimenter_name"],
    )

    _save_stimulus(exp, nwbfile)

    nwbfile.save()

    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)
