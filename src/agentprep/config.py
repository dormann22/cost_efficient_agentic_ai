from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

# Repo root = two levels up from this file (src/agentprep/config.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

class ProjectCfg(BaseModel):
    name: str
    seed: int


class PathsCfg(BaseModel):
    data_raw: str
    data_processed: str
    logs: str


class DatasetCfg(BaseModel):
    target_column: str
    task: str
    positive_class: str | int | None = None
    test_size: float = Field(gt=0, lt=1)
    source_url: str = ""

    @field_validator("task")
    @classmethod
    def _known_task(cls, v: str) -> str:
        allowed = {"classification", "regression"}
        if v not in allowed:
            raise ValueError(f"task must be one of {allowed}, got {v!r}")
        return v


class DataCfg(BaseModel):
    active: str
    catalog: dict[str, DatasetCfg]

    @field_validator("catalog")
    @classmethod
    def _non_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError("data.catalog must contain at least one dataset")
        return v


class ModelCfg(BaseModel):
    provider: str
    name: str
    temperature: float = 0.0
    base_url: str | None = None


class ModelsCfg(BaseModel):
    big: ModelCfg
    small: ModelCfg


class EvaluationCfg(BaseModel):
    downstream_model: str
    metric: str
    f1_average: str = "binary"


class PriceCfg(BaseModel):
    input_per_mtok: float
    output_per_mtok: float


class CostCfg(BaseModel):
    pricing: dict[str, PriceCfg] = {}


class BudgetCfg(BaseModel):
    max_retries: int = Field(ge=0)
    max_usd_per_run: float = Field(gt=0)


class Config(BaseModel):
    project: ProjectCfg
    paths: PathsCfg
    data: DataCfg
    models: ModelsCfg
    evaluation: EvaluationCfg
    cost: CostCfg
    budget: BudgetCfg

    def active_dataset(self) -> DatasetCfg:
        """The dataset config selected by data.active."""
        name = self.data.active
        if name not in self.data.catalog:
            raise KeyError(
                f"data.active={name!r} is not in data.catalog "
                f"({list(self.data.catalog)})"
            )
        return self.data.catalog[name]

    def path(self, key: str) -> Path:
        """Resolve a configured relative path to an absolute Path under repo root."""
        rel = getattr(self.paths, key)
        return (REPO_ROOT / rel).resolve()

@lru_cache(maxsize=1)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


if __name__ == "__main__":
    cfg = load_config()
    print(cfg.model_dump_json(indent=2))
