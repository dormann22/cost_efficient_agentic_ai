import pandas as pd
import pytest

from agentprep.config import (
    BudgetCfg,
    Config,
    CostCfg,
    DataCfg,
    DatasetCfg,
    EvaluationCfg,
    ModelCfg,
    ModelsCfg,
    PathsCfg,
    ProjectCfg,
)
from agentprep.data import freeze_or_load_split
from agentprep.evaluation import train_and_score

DATASET_NAME = "synthetic"


def _dataset_cfg() -> DatasetCfg:
    return DatasetCfg(
        target_column="label",
        task="classification",
        positive_class="yes",
        test_size=0.25,
        source_url="",
    )


def _make_config(tmp_path, seed: int = 42) -> Config:
    return Config(
        project=ProjectCfg(name="test", seed=seed),
        paths=PathsCfg(data_raw=str(tmp_path), data_processed=str(tmp_path), logs=str(tmp_path)),
        data=DataCfg(active=DATASET_NAME, catalog={DATASET_NAME: _dataset_cfg()}),
        models=ModelsCfg(
            big=ModelCfg(provider="openai", name="gpt-4o"),
            small=ModelCfg(provider="openai", name="gpt-4o-mini"),
        ),
        evaluation=EvaluationCfg(downstream_model="random_forest", metric="f1", f1_average="binary"),
        cost=CostCfg(pricing={}),
        budget=BudgetCfg(max_retries=3, max_usd_per_run=5),
    )


def _make_processed_df(n: int = 100) -> pd.DataFrame:
    # Perfectly separable by a threshold on `x` regardless of which rows
    # land in train vs. test, so a correctly-wired evaluator should score
    # near-perfectly on it.
    x = [float(v) for v in range(n)]
    label = ["yes" if v >= n // 2 else "no" for v in range(n)]
    return pd.DataFrame({"x": x, "label": label})


def _freeze(df: pd.DataFrame, tmp_path, seed: int = 42) -> None:
    freeze_or_load_split(df, _dataset_cfg(), DATASET_NAME, seed=seed, processed_dir=tmp_path)


def test_train_and_score_returns_scores_in_range(tmp_path):
    df = _make_processed_df()
    _freeze(df, tmp_path)

    scores = train_and_score(df, config=_make_config(tmp_path))

    assert set(scores) == {"f1", "precision", "recall"}
    for value in scores.values():
        assert 0.0 <= value <= 1.0
    assert scores["f1"] > 0.95


def test_train_and_score_rejects_non_numeric_features(tmp_path):
    df = _make_processed_df()
    _freeze(df, tmp_path)

    df_with_category = df.copy()
    df_with_category["extra"] = "same-for-all-rows"

    with pytest.raises(ValueError):
        train_and_score(df_with_category, config=_make_config(tmp_path))


def test_train_and_score_rejects_missing_values(tmp_path):
    df = _make_processed_df()
    _freeze(df, tmp_path)

    df_with_nan = df.copy()
    df_with_nan.loc[0, "x"] = float("nan")

    with pytest.raises(ValueError):
        train_and_score(df_with_nan, config=_make_config(tmp_path))


def test_train_and_score_rejects_dropped_rows(tmp_path):
    df = _make_processed_df()
    _freeze(df, tmp_path)

    truncated_df = df.drop(index=0)

    with pytest.raises(ValueError):
        train_and_score(truncated_df, config=_make_config(tmp_path))
