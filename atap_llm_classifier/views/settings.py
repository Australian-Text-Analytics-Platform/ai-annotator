import functools
from typing import Self

import panel as pn
from loguru import logger
from pydantic import model_validator

from pydantic_settings import BaseSettings

__all__ = [
    "get_settings",
]


class ViewSettings(BaseSettings):
    NOTIFICATION_DURATION: int = 3000
    USE_NOTIFICATION_GLOBALLY: bool = False

    PIPE_PROMPT_LIVE_UPDATE: bool = True

    @model_validator(mode="after")
    def setup_panel_notification_config(self) -> Self:
        if self.USE_NOTIFICATION_GLOBALLY:
            pn.config.notifications = True
            logger.info("Enabled global panel notification.")
        return self


@functools.lru_cache(maxsize=1)
def get_settings():
    settings = ViewSettings()
    return settings
