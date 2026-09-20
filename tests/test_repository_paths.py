from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

import main as entrypoint
from paths import REPO_ROOT, configure_paths, output_root, workspace_root


def composed_config(overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "src/configs")):
        return compose(config_name="config", overrides=list(overrides))


def test_root_override_resolves_data_checkpoints_and_repo_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("SCBFM_ROOT_DIR", str(tmp_path / "environment"))
    chosen = tmp_path / "storage with spaces"
    cfg = composed_config([f"root_dir={chosen}"])
    configure_paths(cfg)
    assert workspace_root(cfg) == chosen
    assert Path(cfg.finetune.deconv.pseudo_bulk_data_path) == chosen / "datasets/pseudo_bulk/pseudo_bulk_RAW.h5ad"
    assert Path(cfg.pretrain.gene_list_path) == REPO_ROOT / "data/gene_list.txt"
    assert Path(cfg.finetune.deconv.pretrained_model_paths.pretrain_sc) == chosen / "output/pretrain_sc/pretrain_sc.pth"
    assert Path(cfg.finetune.batch_integration.datasets.covid19.reference_path).is_relative_to(chosen)


def test_root_environment_and_relative_output_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("SCBFM_ROOT_DIR", str(tmp_path))
    cfg = composed_config(["output_dir=results"])
    configure_paths(cfg)
    assert output_root(cfg) == tmp_path / "results"
    assert Path(cfg.finetune.deconv.pretrained_model_paths.pretrain_bulk).is_relative_to(tmp_path / "results")


RUNNER_CLASSES = [
    cls for name, cls in vars(entrypoint).items()
    if name.endswith("Runner") and isinstance(cls, type)
]


@pytest.mark.parametrize("runner_class", RUNNER_CLASSES, ids=lambda cls: cls.__name__)
def test_all_runner_outputs_follow_configured_root(runner_class, tmp_path):
    runner = object.__new__(runner_class)
    runner.cfg = OmegaConf.create({"root_dir": str(tmp_path)})
    runner.task_cfg = OmegaConf.create({"cancer": "BRCA", "output_variant": "test"})
    runner.pretrain_cfg = OmegaConf.create({"model_name": "test"})
    runner.task_name = "test"
    runner._model_name = lambda: "test"
    runner._finetune_mode = lambda: "test"
    runner._output_variant = lambda: "test"
    method = getattr(runner, "_task_output_dir", None) or runner._output_dir
    assert method().is_relative_to(tmp_path / "output")


def test_cli_configuration_from_outside_repository(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/main.py"), "--cfg", "job", "--resolve", f"root_dir={tmp_path}"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    cfg = OmegaConf.create(completed.stdout)
    assert cfg.root_dir == str(tmp_path)
    assert Path(cfg.finetune.canc_type_class.gene_list_path).is_file()
    assert not (tmp_path / "output").exists()


def test_notebooks_have_portable_paths_and_no_saved_outputs():
    paths = [*REPO_ROOT.glob("src/analysis/*.ipynb"), *REPO_ROOT.glob("data/notebooks/**/*.ipynb")]
    for path in paths:
        notebook = json.loads(path.read_text())
        assert notebook["cells"][0]["id"] == "repository-paths", path
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert not cell["outputs"], path
            assert cell["execution_count"] is None, path
            source = "".join(cell["source"])
            ast.parse(source)
            assert "/Users/" not in source and "/cluster/work/" not in source, path


def test_container_launcher_preserves_arguments_and_custom_paths(tmp_path):
    # A fake container executable captures arguments; no training or scheduler runs.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "arguments.json"
    executable = bin_dir / "apptainer"
    executable.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    executable.chmod(0o755)
    image = tmp_path / "container with spaces.sif"
    image.touch()
    overrides = ["task=finetune.deconv", "finetune.deconv.model_keys=[random_init]"]
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               CAPTURE=str(capture), SCBFM_ROOT_DIR=str(tmp_path),
               SCBFM_REPO_DIR=str(REPO_ROOT), SCBFM_SIF=str(image), SCBFM_NPROC="1")
    subprocess.run(["bash", str(REPO_ROOT / "cluster/run-job.sh"), *overrides], env=env, check=True)
    arguments = json.loads(capture.read_text())
    assert str(image) in arguments
    assert f"{REPO_ROOT}:{REPO_ROOT}" in arguments
    assert "--nproc_per_node=1" in arguments
    assert arguments[-2:] == overrides


def test_coordinate_generator_only_supports_classification_tasks():
    from analysis.umap_pca.generate_coordinates import TASK_RUNNERS, load_config
    assert set(TASK_RUNNERS) == {"canc_type_class", "canc_type_class_33", "disease_class"}
    for task in TASK_RUNNERS:
        cfg = load_config(task)
        assert Path(cfg.finetune[task].gene_list_path).is_file()
        assert cfg.finetune[task].finetune_mode == "full_ft"
