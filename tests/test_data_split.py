import pandas as pd
import pytest

from agentprep.config import DatasetCfg
from agentprep.data import freeze_or_load_split, load_split


def _make_df(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": range(n),
            "label": ["yes" if i % 2 == 0 else "no" for i in range(n)],
        }
    )


def _dataset_cfg(test_size: float = 0.25) -> DatasetCfg:
    return DatasetCfg(
        target_column="label",
        task="classification",
        positive_class="yes",
        test_size=test_size,
        source_url="",
    )


def test_freeze_creates_and_reload_returns_same_split(tmp_path):
    df = _make_df()
    cfg = _dataset_cfg()

    first = freeze_or_load_split(df, cfg, "synthetic", seed=42, processed_dir=tmp_path)
    second = freeze_or_load_split(df, cfg, "synthetic", seed=42, processed_dir=tmp_path)

    assert first == second
    assert load_split("synthetic", tmp_path) == first


def test_split_file_is_not_rewritten_on_second_call(tmp_path):
    df = _make_df()
    cfg = _dataset_cfg()

    freeze_or_load_split(df, cfg, "synthetic", seed=42, processed_dir=tmp_path)
    path = tmp_path / "synthetic" / "split.json"
    written_once = path.read_text(encoding="utf-8")

    freeze_or_load_split(df, cfg, "synthetic", seed=42, processed_dir=tmp_path)

    assert path.read_text(encoding="utf-8") == written_once


def test_split_sizes_and_disjointness(tmp_path):
    df = _make_df(n=40)
    cfg = _dataset_cfg(test_size=0.25)

    manifest = freeze_or_load_split(df, cfg, "synthetic", seed=42, processed_dir=tmp_path)

    assert manifest.n_train + manifest.n_test == len(df)
    assert manifest.n_test == 10
    assert set(manifest.train_index).isdisjoint(manifest.test_index)
    assert set(manifest.train_index) | set(manifest.test_index) == set(df.index)


def test_changed_data_raises_on_reload(tmp_path):
    df = _make_df()
    cfg = _dataset_cfg()
    freeze_or_load_split(df, cfg, "synthetic", seed=42, processed_dir=tmp_path)

    changed_df = _make_df()
    changed_df.loc[0, "feature"] = 9999

    with pytest.raises(RuntimeError):
        freeze_or_load_split(changed_df, cfg, "synthetic", seed=42, processed_dir=tmp_path)


def test_load_split_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split("does-not-exist", tmp_path)
