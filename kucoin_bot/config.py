"""Configuration and secrets management module."""

import os
import re
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Configuration manager with environment variable support."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}. "
                "Copy config.example.yaml to config.yaml and configure."
            )

        with open(self.config_path) as f:
            raw_config = yaml.safe_load(f)

        self._config = self._resolve_env_vars(raw_config)

    def _resolve_env_vars(self, obj: Any) -> Any:
        """Recursively resolve environment variables in config."""
        if isinstance(obj, str):
            # Match ${VAR_NAME} pattern
            pattern = r"\$\{([^}]+)\}"
            matches = re.findall(pattern, obj)
            for var_name in matches:
                env_value = os.environ.get(var_name, "")
                obj = obj.replace(f"${{{var_name}}}", env_value)
            return obj
        elif isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        return obj

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-notation key."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    @property
    def api_key(self) -> str:
        """Get API key."""
        return self.get("api.key", "")

    @property
    def api_secret(self) -> str:
        """Get API secret."""
        return self.get("api.secret", "")

    @property
    def api_passphrase(self) -> str:
        """Get API passphrase."""
        return self.get("api.passphrase", "")

    @property
    def is_sandbox(self) -> bool:
        """Check if sandbox mode."""
        return self.get("api.sandbox", True)

    @property
    def trading_mode(self) -> str:
        """Get trading mode (paper/live)."""
        return self.get("trading.mode", "paper")

    @property
    def markets(self) -> list[str]:
        """Get enabled markets."""
        return self.get("trading.markets", ["spot"])

    @property
    def risk_config(self) -> dict[str, Any]:
        """Get risk management configuration."""
        return self.get("risk", {})

    @property
    def strategy_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return self.get("strategies", {})

    @property
    def backtest_config(self) -> dict[str, Any]:
        """Get backtest configuration."""
        return self.get("backtest", {})

    @property
    def retry_config(self) -> dict[str, Any]:
        """Get retry configuration."""
        return self.get("retry", {})
