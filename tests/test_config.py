"""Tests for configuration module."""

import os
import pytest
import tempfile
from pathlib import Path

from kucoin_bot.config import Config


class TestConfig:
    """Test cases for Config class."""

    def test_config_load(self, tmp_path: Path) -> None:
        """Test loading configuration from file."""
        config_content = """
api:
  key: test_key
  secret: test_secret
  passphrase: test_pass
  sandbox: true
trading:
  mode: paper
  markets:
    - spot
    - futures
risk:
  max_position_pct: 5.0
  max_leverage: 3
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        config = Config(str(config_file))
        
        assert config.api_key == "test_key"
        assert config.api_secret == "test_secret"
        assert config.api_passphrase == "test_pass"
        assert config.is_sandbox is True
        assert config.trading_mode == "paper"
        assert "spot" in config.markets
        assert "futures" in config.markets

    def test_config_env_var_resolution(self, tmp_path: Path) -> None:
        """Test environment variable resolution."""
        os.environ["TEST_API_KEY"] = "env_test_key"
        os.environ["TEST_API_SECRET"] = "env_test_secret"
        
        config_content = """
api:
  key: "${TEST_API_KEY}"
  secret: "${TEST_API_SECRET}"
  sandbox: true
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        config = Config(str(config_file))
        
        assert config.api_key == "env_test_key"
        assert config.api_secret == "env_test_secret"
        
        # Cleanup
        del os.environ["TEST_API_KEY"]
        del os.environ["TEST_API_SECRET"]

    def test_config_get_nested(self, tmp_path: Path) -> None:
        """Test getting nested configuration values."""
        config_content = """
risk:
  max_position_pct: 5.0
  max_drawdown_pct: 10.0
strategies:
  trend:
    short_period: 20
    long_period: 50
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        config = Config(str(config_file))
        
        assert config.get("risk.max_position_pct") == 5.0
        assert config.get("strategies.trend.short_period") == 20
        assert config.get("nonexistent", "default") == "default"

    def test_config_file_not_found(self) -> None:
        """Test error when config file not found."""
        with pytest.raises(FileNotFoundError):
            Config("nonexistent_config.yaml")
