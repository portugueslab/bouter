"""
Rerun this file if the logic of freely-swimming experiments changes
"""

import flammkuchen as fl

from bouter import free
from bouter.tests import ASSETS_PATH


def create_assets():
    source_dataset_path = ASSETS_PATH / "freely_swimming_dataset"
    experiment = free.FreelySwimmingExperiment(source_dataset_path)

    bouts, continuity = experiment.get_bouts()
    velocities_df = experiment.compute_velocity()

    velocities = velocities_df[
        ["vel_f{}".format(i_fish) for i_fish in range(experiment.n_fish)]
    ]

    bouts_summary = experiment.get_bout_properties()

    # Load computed velocities
    fl.save(
        source_dataset_path / "test_extracted_bouts.h5",
        dict(
            bouts_summary=bouts_summary,
            bouts=bouts,
            continuity=continuity,
            velocities=velocities,
        ),
    )


if __name__ == "__main__":
    create_assets()
