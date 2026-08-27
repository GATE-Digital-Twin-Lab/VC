"""Configuration loading.

The protocol requires that no magic numbers live in code (section 1). Every
tunable is read from ``config/default.yaml`` through this module. Missing keys
raise rather than defaulting, so a knob that was never declared cannot silently
acquire a value at runtime.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


_MISSING = object()


class ConfigError(KeyError):
    """A knob was requested that the config file does not declare."""


@dataclass(frozen=True)
class Config:
    """Read-only view over the nested config mapping."""

    data: dict
    path: Path

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is not _MISSING:
                    return default
                raise ConfigError(
                    f"{dotted!r} is not declared in {self.path}. "
                    "Add it there rather than hard-coding a value."
                )
            node = node[part]
        return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def section(self, dotted: str) -> dict:
        node = self.get(dotted)
        if not isinstance(node, dict):
            raise ConfigError(f"{dotted!r} is not a section")
        return node

    # -- derived paths -----------------------------------------------------
    @property
    def data_root(self) -> Path:
        return self._rooted(self.get("run.data_root"))

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def results_root(self) -> Path:
        return self._rooted(self.get("run.results_root"))

    @property
    def runs_root(self) -> Path:
        return self.results_root / self.get("run.runs_subdir")

    def _rooted(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else REPO_ROOT / p

    def ensure_dirs(self) -> None:
        """Only the two shared roots. Run-scoped directories belong to Store,
        which creates them under a timestamp."""
        for d in (self.raw_dir, self.runs_root):
            d.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, overrides: Iterable[str]) -> "Config":
        """Apply ``a.b.c=value`` strings, parsed as YAML scalars."""
        data = copy.deepcopy(self.data)
        for item in overrides:
            if "=" not in item:
                raise ConfigError(f"override {item!r} is not of the form key.path=value")
            dotted, raw = item.split("=", 1)
            node = data
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = yaml.safe_load(raw)
        return Config(data=data, path=self.path)


def load_config(path: str | os.PathLike | None = None,
                overrides: Iterable[str] | None = None) -> Config:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    cfg = Config(data=data, path=cfg_path)
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return cfg
