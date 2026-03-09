"""
Конфигурация Loguru для всего приложения.

Вызывается один раз при старте в main.py:
    from src.core.logging import setup_logging
    setup_logging(settings.env)
"""

import sys

from loguru import logger


_BASE_EXTRA = {
    "request_id": "-",
    "module": "-",
    "component": "-",
    "node": "-",
}


def _patch_record(record: dict) -> None:
    """Guarantee stable extra keys for both text and JSON log formats."""
    extra = record["extra"]
    for key, value in _BASE_EXTRA.items():
        extra.setdefault(key, value)


def setup_logging(env: str = "prod") -> None:
    """
    Настраивает Loguru:
    - В dev: DEBUG-уровень, подробный формат с цветами.
    - В prod/stage: INFO-уровень, JSON-сериализация для агрегаторов логов.
    """
    logger.remove()
    logger.configure(extra=_BASE_EXTRA, patcher=_patch_record)

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "rid=<magenta>{extra[request_id]}</magenta> | "
        "mod=<yellow>{extra[module]}</yellow> | "
        "cmp=<blue>{extra[component]}</blue> | "
        "node=<cyan>{extra[node]}</cyan> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    if env == "dev":
        logger.add(
            sys.stderr,
            format=fmt,
            level="DEBUG",
            colorize=True,
            backtrace=True,
            diagnose=False,
        )
    else:
        logger.add(
            sys.stderr,
            format=fmt,
            level="INFO",
            colorize=False,
            serialize=True,
            backtrace=False,
            diagnose=False,
        )


def get_logger(**extra):
    """Return application logger, optionally bound with structured context."""
    return logger.bind(**extra)
