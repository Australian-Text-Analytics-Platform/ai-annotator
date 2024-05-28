import enum
import functools
from typing import Callable, Any
from venv import logger

import panel as pn
from atap_llm_classifier.views.settings import ViewSettings, get_settings

settings: ViewSettings = get_settings()


def catch(raise_err: bool = False):
    """Granular decorator to catch exceptions and use panel's notification."""

    def catch_wrapper(fn: Callable) -> Callable:
        global settings

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not settings.USE_NOTIFICATION_GLOBALLY:
                try:
                    res: Any = fn(*args, **kwargs)
                    return res
                except Exception as e:
                    logger.error(f"Caught exception: {e}.")
                    pn.state.notifications.error(
                        message=str(e), duration=settings.NOTIFICATION_DURATION
                    )
                    if raise_err:
                        raise e
            else:
                return fn(*args, **kwargs)

        return wrapper

    return catch_wrapper


def a_catch(raise_err: bool = False):
    """Granular decorator to catch exceptions and use panel's notification. Async fns."""

    def catch_wrapper(fn):
        global settings

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            if not settings.USE_NOTIFICATION_GLOBALLY:
                try:
                    res: Any = await fn(*args, **kwargs)
                    return res
                except Exception as e:
                    logger.error(f"Caught exception: {e}.")
                    pn.state.notifications.error(
                        message=str(e), duration=settings.NOTIFICATION_DURATION
                    )
                    if raise_err:
                        raise e
            else:
                return await fn(*args, **kwargs)

        return wrapper

    return catch_wrapper


class NotifyEnum(enum.Enum):
    def info(self):
        pn.state.notifications.info(
            message=self.value, duration=settings.NOTIFICATION_DURATION
        )


class PipelineClassification(NotifyEnum):
    MODEL_CONFIG_LOCKED_ON_CLASSIFY: str = (
        "Model configuration is locked for classification."
    )
