"""utils.py"""

from datetime import datetime
from functools import lru_cache
from io import IOBase
from pathlib import Path
from typing import TypeVar, Type
from enum import Enum

import yaml
from pydantic import BaseModel

__all__ = ["Asset"]


class Asset(Enum):
    TECHNIQUES: str = "techniques"
    MODIFIERS: str = "modifiers"
    PROVIDERS: str = "providers"

    def get_path(self) -> Path:
        match self:
            case Asset.TECHNIQUES:
                return asset_dir / "techniques.yml"
            case Asset.MODIFIERS:
                return asset_dir / "modifiers.yml"
            case Asset.PROVIDERS:
                return asset_dir / "providers.yml"

    def get(self, key: str) -> dict:
        return load_asset(self)[key]


@lru_cache(maxsize=len(Asset))
def load_asset(asset: Asset) -> dict:
    with open(asset.get_path(), "r", encoding="utf-8") as h:
        return yaml.safe_load(h)


def _last_update(path: Path) -> datetime:
    # todo: get the last update of file from git commit, for info related to say...
    #   why choose this model?
    pass


# Below unused: kept for now.

TBaseModel = TypeVar("TBaseModel", bound=BaseModel)


def load_model_from_yaml(
    path_or_stream: str | IOBase,
    model_cls: Type[TBaseModel],
) -> TBaseModel:
    model_data: dict
    if isinstance(path_or_stream, str):
        path = Path(path_or_stream)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path_or_stream} is not a file. "
                f"Assumes path_or_stream is path when given str."
            )
        with open(path, "r", encoding="utf-8") as h:
            model_data = yaml.safe_load(h)
    else:
        model_data = yaml.safe_load(path_or_stream)
    return model_cls.model_validate(model_data)


asset_dir: Path = Path(__file__).parent.parent / "assets"
assert asset_dir.exists(), f"Asset directory: {asset_dir.absolute()} not found."
