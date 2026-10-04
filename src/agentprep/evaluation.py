from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from agentprep.config import Config, load_config
from agentprep.data import load_split

# Fixed baseline model for now. Future plan: support both classification and
# regression tasks, with the agent choosing among several downstream models
# based on properties of the preprocessed data. RandomForestClassifier is a
# placeholder until that routing logic exists.
FIXED_MODEL_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
)


def train_and_score(processed_df: pd.DataFrame, config: Config | None = None) -> dict[str, float]:
    """Fit the fixed downstream model on the frozen train split, score it on the frozen test split.

    `processed_df` must:
      - keep the original row index produced by `load_raw_dataset` (rows the
        preprocessing dropped are simply excluded from scoring),
      - keep the target column under its original name holding the dataset's
        original raw labels (feature preprocessing should not alter it —
        this function does the positive/negative binarization itself),
      - otherwise be fully numeric with no missing values, since that's what
        preprocessing exists to guarantee.
    """
    config = config or load_config()
    dataset_cfg = config.active_dataset()
    split = load_split(config.data.active, config.path("data_processed"))

    target_column = dataset_cfg.target_column
    positive_class = str(dataset_cfg.positive_class).strip()

    frozen_index = set(split.train_index) | set(split.test_index)
    missing = frozen_index - set(processed_df.index)
    if missing:
        raise ValueError(
            f"processed_df is missing {len(missing)} row(s) present in the frozen split "
            f"(e.g. {sorted(missing)[:5]}). Preprocessing must preserve the original row index."
        )

    def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if target_column not in frame.columns:
            raise ValueError(f"processed_df is missing the target column {target_column!r}")
        # Comparison-only strip so raw label formatting (e.g. " >50K") is
        # matched correctly, without ever mutating processed_df itself.
        y = (frame[target_column].astype(str).str.strip() == positive_class).astype(int)
        X = frame.drop(columns=[target_column])
        non_numeric = X.select_dtypes(exclude="number").columns
        if len(non_numeric) > 0:
            raise ValueError(
                f"processed_df has non-numeric feature column(s) {list(non_numeric)}; "
                "train_and_score requires fully numeric, preprocessed input."
            )
        if X.isna().any().any():
            raise ValueError("processed_df has missing values in its feature columns.")
        return X, y

    X_train, y_train = _xy(processed_df.loc[split.train_index])
    X_test, y_test = _xy(processed_df.loc[split.test_index])

    model = RandomForestClassifier(**FIXED_MODEL_PARAMS, random_state=config.project.seed)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    average = config.evaluation.f1_average
    return {
        "f1": f1_score(y_test, predictions, average=average),
        "precision": precision_score(y_test, predictions, average=average),
        "recall": recall_score(y_test, predictions, average=average),
    }
