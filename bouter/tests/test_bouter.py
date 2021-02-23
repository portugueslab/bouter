import numpy as np
import pytest

import bouter as bt


def test_class_instantiation(embedded_exp_path):
    # Test instantiation modalities:
    assert (
        bt.Experiment(embedded_exp_path)
        == bt.Experiment(str(embedded_exp_path))
        == bt.Experiment(embedded_exp_path / "192316_metadata.json")
    )


# TODO if we test multiple datasets this will have to be improved
@pytest.mark.parametrize(
    "prop_name, outcome",
    [
        ("protocol_name", "closed_open_loop"),
        ("protocol_version", None),
        (
            "protocol_parameters",
            {
                "grating_cycle": 10,
                "grating_duration": 4.0,
                "inter_stim_pause": 0,
                "n_repeats": 1,
                "post_pause": 0.0,
                "pre_pause": 0.0,
            },
        ),
        ("stim_start_times", [0.002]),
        ("stim_end_times", [600.004995]),
    ],
)
def test_class_properties(embedded_exp_path, prop_name, outcome):
    exp = bt.Experiment(embedded_exp_path)

    val = getattr(exp, prop_name)
    if isinstance(val, np.ndarray):
        assert np.isclose(val, outcome)
    else:
        assert val == outcome


@pytest.mark.parametrize(
    "logname, log_props",
    [
        (
            "stimulus_log",
            dict(
                nrows=100,
                columns=[
                    "closed loop 1D_vel",
                    "closed loop 1D_base_vel",
                    "closed loop 1D_gain",
                    "closed loop 1D_lag",
                    "closed loop 1D_fish_swimming",
                    "t",
                ],
            ),
        ),
        ("estimator_log", dict(nrows=100, columns=["vigour", "t"])),
        (
            "behavior_log",
            dict(
                nrows=1200,
                columns=["tail_sum"]
                + [f"theta_{i:02}" for i in range(25)]
                + ["t"],
            ),
        ),
    ],
)
def test_logs_loading(embedded_exp_path, logname, log_props):
    exp = bt.EmbeddedExperiment(embedded_exp_path)
    df = getattr(exp, logname)

    assert len(df) == log_props["nrows"]
    assert all(df.columns == log_props["columns"])


def test_warning_raise(embedded_exp_path):
    exp = bt.EmbeddedExperiment(embedded_exp_path)
    starts, ends = exp.stimulus_starts_ends()
    for s, t in zip(
        [starts, ends], [np.array([0.002]), np.array([600.004995])]
    ):
        assert all(s == t)
