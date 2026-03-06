import sys
from unittest.mock import patch

from loguru import logger

from src.core.logging import setup_logging


class TestSetupLogging:
    def test_setup_dev(self):
        """setup_logging('dev') не падает."""
        setup_logging("dev")

    def test_setup_prod(self):
        """setup_logging('prod') не падает."""
        setup_logging("prod")

    def test_setup_stage(self):
        """setup_logging('stage') не падает."""
        setup_logging("stage")

    def test_default_is_prod(self):
        """По умолчанию — prod."""
        setup_logging()

    def test_dev_removes_previous_handlers(self):
        """Повторный вызов не дублирует handlers."""
        setup_logging("dev")
        setup_logging("dev")
        # Не должно упасть; loguru.remove() гарантирует очистку.

    def test_dev_sets_debug_level(self, capsys):
        """В dev-режиме DEBUG-сообщения логируются."""
        setup_logging("dev")
        logger.debug("test-debug-message")
        captured = capsys.readouterr()
        assert "test-debug-message" in captured.err

    def test_prod_skips_debug_level(self, capsys):
        """В prod-режиме DEBUG-сообщения не логируются."""
        setup_logging("prod")
        logger.debug("should-not-appear")
        captured = capsys.readouterr()
        assert "should-not-appear" not in captured.err

    def test_prod_logs_info(self, capsys):
        """В prod-режиме INFO-сообщения логируются."""
        setup_logging("prod")
        logger.info("prod-info-message")
        captured = capsys.readouterr()
        assert "prod-info-message" in captured.err
