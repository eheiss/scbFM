from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_CHECKPOINT_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _CHECKPOINT_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _CHECKPOINT_HASH_CACHE[key] = value
    return value


def _git_commit(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _source_fingerprint(repo_dir: Path) -> str:
    digest = hashlib.sha256()
    source_dir = repo_dir / "src"
    source_paths = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
    )
    for path in source_paths:
        digest.update(str(path.relative_to(repo_dir)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _checkpoint_records(checkpoint_paths: dict[str, str]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for model_key, configured_path in checkpoint_paths.items():
        if not configured_path:
            continue
        path = Path(configured_path).resolve()
        if not path.is_file():
            records[model_key] = {"path": str(path), "exists": False}
            continue
        stat = path.stat()
        records[model_key] = {
            "path": str(path),
            "exists": True,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256_file(path),
        }
    return records


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def start_run_metadata(
    path: Path,
    payload: dict[str, object],
    *,
    checkpoint_paths: dict[str, str],
    repo_dir: Path,
) -> None:
    enriched = dict(payload)
    enriched.update(
        {
            "run_status": "running",
            "started_at_utc": _utc_now(),
            "git_commit": _git_commit(repo_dir),
            "source_fingerprint_sha256": _source_fingerprint(repo_dir),
            "checkpoint_paths": checkpoint_paths,
            "checkpoint_files": _checkpoint_records(checkpoint_paths),
        }
    )
    _atomic_json(path, enriched)


def complete_run_metadata(path: Path, results_path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Run metadata was not created: {path}")
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not results_path.is_file() or results_path.stat().st_size == 0:
        raise FileNotFoundError(f"Expected non-empty results file: {results_path}")
    metadata.update(
        {
            "run_status": "complete",
            "completed_at_utc": _utc_now(),
            "results_path": str(results_path.resolve()),
            "results_size_bytes": results_path.stat().st_size,
            "results_sha256": _sha256_file(results_path),
        }
    )
    _atomic_json(path, metadata)
