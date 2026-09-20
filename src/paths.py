"""Repository assets and configurable experiment storage locations."""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict


REPO_ROOT = Path(__file__).resolve().parents[1]


def workspace_root(cfg: DictConfig | None = None) -> Path:
    value = cfg.get("root_dir") if cfg is not None else None
    value = value or os.environ.get("SCBFM_ROOT_DIR") or os.environ.get("SCBFM_CLUSTER_ROOT")
    return Path(value or REPO_ROOT.parent).expanduser().resolve()


def output_root(cfg: DictConfig | None = None) -> Path:
    value = cfg.get("output_dir") if cfg is not None else None
    value = value or os.environ.get("SCBFM_OUTPUT_DIR")
    if not value:
        return workspace_root(cfg) / "output"
    path = Path(value).expanduser()
    return (path if path.is_absolute() else workspace_root(cfg) / path).resolve()


def figure_root() -> Path:
    value = os.environ.get("SCBFM_FIGURE_DIR")
    return Path(value).expanduser().resolve() if value else output_root() / "figures"


def configure_paths(cfg: DictConfig, root_dir: str | Path | None = None) -> None:
    """Set the workspace before any runner resolves dataset/checkpoint paths."""
    with open_dict(cfg):
        cfg.root_dir = str(Path(root_dir).expanduser().resolve() if root_dir else workspace_root(cfg))
        cfg.repo_dir = str(REPO_ROOT)
        cfg.output_dir = str(output_root(cfg))


OmegaConf.register_new_resolver("scbfm_root", lambda: str(workspace_root()), replace=True)
OmegaConf.register_new_resolver("scbfm_repo", lambda: str(REPO_ROOT), replace=True)
