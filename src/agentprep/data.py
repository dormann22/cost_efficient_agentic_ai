from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from agentprep.config import Config, DatasetCfg

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def load_raw_dataset(config: Config, name: str | None = None) -> pd.DataFrame:
    """Load a raw dataset by name (defaults to config.data.active), unmodified.

    Values are returned exactly as they appear in the source file (leading
    whitespace, "?" sentinels, mixed types, etc. all intact) — cleaning is
    the preprocessing agent's job, not this loader's.
    """
    name = name or config.data.active
    if name != "adult":
        raise NotImplementedError(f"No raw loader registered for dataset {name!r}")

    path = config.path("data_raw") / "adult" / "adult.data"
    return pd.read_csv(path, header=None, names=ADULT_COLUMNS)


@dataclass
class SplitManifest:
    dataset: str
    seed: int
    test_size: float
    stratify_column: str
    n_train: int
    n_test: int
    raw_sha256: str
    train_index: list[int]
    test_index: list[int]

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "seed": self.seed,
            "test_size": self.test_size,
            "stratify_column": self.stratify_column,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "raw_sha256": self.raw_sha256,
            "train_index": self.train_index,
            "test_index": self.test_index,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "SplitManifest":
        return cls(**raw)


def _split_path(dataset_name: str, processed_dir: str | Path) -> Path:
    return Path(processed_dir) / dataset_name / "split.json"


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Content hash of a dataframe's values + index, used to detect a changed raw source."""
    values = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def freeze_or_load_split(
    df: pd.DataFrame,
    dataset_cfg: DatasetCfg,
    dataset_name: str,
    seed: int,
    processed_dir: str | Path,
) -> SplitManifest:
    """Create the dataset's train/test split on first use, or load the frozen one.

    Once `split.json` exists it is only ever read back — never regenerated —
    so every comparison between preprocessing strategies runs against the
    exact same rows. If `df`'s content no longer matches the hash recorded
    when the split was frozen, this raises loudly instead of silently
    producing an inconsistent split.
    """
    path = _split_path(dataset_name, processed_dir)
    raw_sha256 = _hash_dataframe(df)

    if path.exists():
        manifest = SplitManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if manifest.raw_sha256 != raw_sha256:
            raise RuntimeError(
                f"The raw data passed in no longer matches the hash recorded in "
                f"{path} — the frozen split may be stale or invalid. Delete "
                f"{path} to intentionally re-freeze the split."
            )
        return manifest

    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=dataset_cfg.test_size,
        random_state=seed,
        stratify=df[dataset_cfg.target_column],
    )

    manifest = SplitManifest(
        dataset=dataset_name,
        seed=seed,
        test_size=dataset_cfg.test_size,
        stratify_column=dataset_cfg.target_column,
        n_train=len(train_idx),
        n_test=len(test_idx),
        raw_sha256=raw_sha256,
        train_index=sorted(int(i) for i in train_idx),
        test_index=sorted(int(i) for i in test_idx),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return manifest


def load_split(dataset_name: str, processed_dir: str | Path) -> SplitManifest:
    """Read back a previously frozen split. Raises if it was never created."""
    path = _split_path(dataset_name, processed_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No frozen split found at {path}. Run freeze_or_load_split first "
            "(e.g. via scripts/freeze_adult_split.py)."
        )
    return SplitManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
