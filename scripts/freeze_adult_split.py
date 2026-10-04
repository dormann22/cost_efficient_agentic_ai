"""Freeze (or verify) the active dataset's train/test split.

Run once to create data/processed/<dataset>/split.json — after that, running
this again just confirms the frozen split still matches the raw data and
reports its size; it never regenerates the split.

Run with: python scripts/freeze_adult_split.py
"""

from agentprep.config import load_config
from agentprep.data import freeze_or_load_split, load_raw_dataset


def main() -> None:
    config = load_config()
    dataset_name = config.data.active
    dataset_cfg = config.active_dataset()

    df = load_raw_dataset(config, dataset_name)
    manifest = freeze_or_load_split(
        df,
        dataset_cfg,
        dataset_name,
        seed=config.project.seed,
        processed_dir=config.path("data_processed"),
    )

    split_path = config.path("data_processed") / dataset_name / "split.json"
    print(f"Frozen split for {dataset_name!r} at {split_path}")
    print(f"  n_train={manifest.n_train} n_test={manifest.n_test} seed={manifest.seed}")


if __name__ == "__main__":
    main()
