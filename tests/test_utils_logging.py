"""
Unit tests for zeus.utils.logging module.
"""
import pytest
import os
import tempfile
import logging
from logging.handlers import RotatingFileHandler
from unittest.mock import Mock, patch, mock_open
from zeus.utils.logging import maybe_reset_wandb, setup_events_logger, EVENTS_LEVEL_NUM


class TestMaybeResetWandb:
    """Tests for maybe_reset_wandb function."""

    def test_maybe_reset_wandb_wandb_off(self):
        """Test that function returns early when wandb is off."""
        validator = Mock()
        validator.config = Mock()
        validator.config.wandb = Mock()
        validator.config.wandb.off = True
        
        # Should not raise any errors
        maybe_reset_wandb(validator)

    @patch('zeus.utils.logging.wandb')
    def test_maybe_reset_wandb_no_run(self, mock_wandb):
        """Test that function returns early when wandb run is None."""
        validator = Mock()
        validator.config = Mock()
        validator.config.wandb = Mock()
        validator.config.wandb.off = False
        mock_wandb.run = None
        
        # Should return early without raising errors
        maybe_reset_wandb(validator)
        # Since wandb.run is None, finish() should never be called
        # We can't assert on mock_wandb.run.finish because run is None

    @patch('zeus.utils.logging.wandb')
    @patch('builtins.open', new_callable=mock_open, read_data=b'line\n' * 50000)
    def test_maybe_reset_wandb_below_threshold(self, mock_file, mock_wandb):
        """Test that function doesn't reset when below threshold."""
        validator = Mock()
        validator.config = Mock()
        validator.config.wandb = Mock()
        validator.config.wandb.off = False
        mock_wandb.run = Mock()
        mock_wandb.run.dir = "/tmp/test"
        
        maybe_reset_wandb(validator)
        mock_wandb.run.finish.assert_not_called()

    @patch('zeus.utils.logging.bt')
    @patch('zeus.utils.logging.wandb')
    @patch('builtins.open', new_callable=mock_open, read_data=b'line\n' * 100000)
    def test_maybe_reset_wandb_above_threshold(self, mock_file, mock_wandb, mock_bt):
        """Test that function resets when above threshold."""
        validator = Mock()
        validator.config = Mock()
        validator.config.wandb = Mock()
        validator.config.wandb.off = False
        mock_wandb.run = Mock()
        mock_wandb.run.dir = "/tmp/test"
        validator.init_wandb = Mock()
        mock_bt.logging = Mock()
        
        maybe_reset_wandb(validator)
        mock_wandb.run.finish.assert_called_once()
        validator.init_wandb.assert_called_once()


class TestSetupEventsLogger:
    """Tests for setup_events_logger function."""

    def test_setup_events_logger_creates_logger(self):
        """Test that setup_events_logger creates a logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_events_logger(tmpdir, events_retention_size=1024*1024)
            assert logger is not None
            assert logger.name == "event"

    def test_setup_events_logger_sets_level(self):
        """Test that logger level is set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_events_logger(tmpdir, events_retention_size=1024*1024)
            assert logger.isEnabledFor(EVENTS_LEVEL_NUM)

    def test_setup_events_logger_creates_file_handler(self):
        """Test that file handler is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_events_logger(tmpdir, events_retention_size=1024*1024)
            assert len(logger.handlers) > 0
            # Check that at least one handler is a file handler
            assert any(isinstance(h, RotatingFileHandler) for h in logger.handlers)

    def test_setup_events_logger_event_method(self):
        """Test that event method is available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_events_logger(tmpdir, events_retention_size=1024*1024)
            # Should have event method
            assert hasattr(logger, 'event')
            assert callable(logger.event)

