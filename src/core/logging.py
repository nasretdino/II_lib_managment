"""
Конфигурация Loguru для всего приложения.

Вызывается один раз при старте в main.py:
    from src.core.logging import setup_logging
    setup_logging(settings.env)
"""

import sys

from loguru import logger


def setup_logging(env: str = "prod") -> None:
    """
    Настраивает Loguru:
    - В dev: DEBUG-уровень, подробный формат с цветами.
    - В prod/stage: INFO-уровень, JSON-сериализация для агрегаторов логов.
    """
    logger.remove()
    logger.configure(extra={"request_id": "-"})

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "rid=<magenta>{extra[request_id]}</magenta> | "
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
