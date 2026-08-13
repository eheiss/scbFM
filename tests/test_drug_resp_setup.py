import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from finetune.drug_resp import (  # noqa: E402
    DrugRespBulkFormerPCARFRunner,
    DrugRespPCARFRunner,
    DrugRespRawMLPRunner,
    DrugRespRawPCARFRunner,
    DrugRespScGPTPCARFRunner,
)
from finetune.drug_resp.runner import DrugRespRunner  # noqa: E402
from finetune.drug_resp.raw_mlp_runner import RawDrugRespMLP  # noqa: E402
from finetune.canc_type_class.raw_mlp_runner import solve_hidden_dim  # noqa: E402


class DrugResponseSetupTest(unittest.TestCase):
    def test_all_evaluation_runners_use_drug_response_task_identity(self) -> None:
        for runner_class in (
            DrugRespRunner,
            DrugRespPCARFRunner,
            DrugRespRawMLPRunner,
            DrugRespRawPCARFRunner,
            DrugRespBulkFormerPCARFRunner,
            DrugRespScGPTPCARFRunner,
        ):
            self.assertEqual(runner_class.task_name, "drug_resp")

    def test_pair_fold_manifest_is_reused_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "folds.csv"
            runner = object.__new__(DrugRespRunner)
            runner.task_cfg = SimpleNamespace(
                cv_folds=5,
                random_seed=42,
                cv_fold_manifest_path=str(manifest),
            )
            runner.rank = 0
            cell_ids = np.asarray([f"cell-{index % 7}" for index in range(25)])
            drug_ids = np.asarray([f"drug-{index}" for index in range(25)])
            targets = np.linspace(-2.0, 2.0, 25, dtype=np.float32)

            first = runner._build_or_load_cv_splits(cell_ids, drug_ids, targets)
            first_fingerprint = runner._cv_fold_fingerprint
            second = runner._build_or_load_cv_splits(cell_ids, drug_ids, targets)

            self.assertEqual(first_fingerprint, runner._cv_fold_fingerprint)
            self.assertEqual(len(first), 5)
            for (first_train, first_test), (second_train, second_test) in zip(
                first,
                second,
            ):
                np.testing.assert_array_equal(first_train, second_train)
                np.testing.assert_array_equal(first_test, second_test)
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(targets))
            self.assertEqual({int(row["fold"]) for row in rows}, {1, 2, 3, 4, 5})

    def test_raw_mlp_supports_expression_and_drug_only_variants(self) -> None:
        runner = object.__new__(DrugRespRawMLPRunner)
        matrix = np.ones((3, 5), dtype=np.float32)
        training_cells = np.arange(3)

        runner.task_cfg = SimpleNamespace(raw_mlp_feature_mode="all_genes")
        np.testing.assert_array_equal(
            runner._select_training_hvg_indices(matrix, training_cells),
            np.arange(5),
        )
        runner.task_cfg = SimpleNamespace(raw_mlp_feature_mode="drug_only")
        np.testing.assert_array_equal(
            runner._select_training_hvg_indices(matrix, training_cells),
            np.empty(0, dtype=np.int64),
        )
        runner.task_cfg = SimpleNamespace(raw_mlp_feature_mode="invalid")
        with self.assertRaisesRegex(ValueError, "all_genes, hvg1199, drug_only"):
            runner._select_training_hvg_indices(matrix, training_cells)

    def test_mad_selection_uses_distinct_training_cell_lines(self) -> None:
        runner = object.__new__(DrugRespRunner)
        runner.selected_gene_count = 2
        runner.task_cfg = SimpleNamespace(hvg_selection_method="mad")
        matrix = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 10.0, 4.0],
                [0.0, 4.0, 0.0, 8.0],
            ],
            dtype=np.float32,
        )

        selected = runner._select_training_hvg_indices(
            matrix,
            np.asarray([0, 1, 1, 2, 2, 2]),
        )

        np.testing.assert_array_equal(selected, np.asarray([1, 3]))

    def test_drug_only_training_loader_emits_empty_raw_expression(self) -> None:
        runner = object.__new__(DrugRespRawMLPRunner)
        runner.task_cfg = SimpleNamespace(
            batch_size=2,
            num_workers=0,
            random_seed=42,
            raw_mlp_standardize=True,
        )
        runner.device = torch.device("cpu")
        runner.is_distributed = False
        runner.is_master = False
        runner.fold_gene_indices = np.empty(0, dtype=np.int64)

        runner._build_train_loader(
            X_cell=np.ones((2, 4), dtype=np.float32),
            drug_emb_matrix=np.ones((2, 3), dtype=np.float32),
            cell_idxs_train=np.asarray([0, 1]),
            drug_idxs_train=np.asarray([0, 1]),
            ic50_train=np.asarray([0.1, 0.2], dtype=np.float32),
        )

        _, batch, drug_embeddings, targets = next(iter(runner.train_loader))
        self.assertEqual(tuple(batch["raw_expr"].shape), (2, 0))
        self.assertEqual(tuple(drug_embeddings.shape), (2, 3))
        self.assertEqual(tuple(targets.shape), (2,))

    def test_drug_only_mlp_matches_parameter_target_without_expression(self) -> None:
        drug_embedding_dim = 512
        target_params = 6_560_000
        hidden_dim = solve_hidden_dim(
            input_dim=drug_embedding_dim,
            output_dim=1,
            target_params=target_params,
            hidden_layers=2,
        )
        model = RawDrugRespMLP(
            expression_dim=0,
            drug_emb_dim=drug_embedding_dim,
            hidden_dim=hidden_dim,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        self.assertLess(abs(parameter_count - target_params), 10_000)
        predictions = model(
            {"raw_expr": torch.empty((3, 0), dtype=torch.float32)},
            torch.ones((3, drug_embedding_dim), dtype=torch.float32),
        )
        self.assertEqual(tuple(predictions.shape), (3,))

    def test_external_regression_prefixes_match_config(self) -> None:
        bulkformer = object.__new__(DrugRespBulkFormerPCARFRunner)
        scgpt = object.__new__(DrugRespScGPTPCARFRunner)
        self.assertEqual(bulkformer._pca_prefix(), "bulkformer_pca")
        self.assertEqual(bulkformer._rf_prefix(), "bulkformer_rf")
        self.assertEqual(scgpt._pca_prefix(), "scgpt_pca")
        self.assertEqual(scgpt._rf_prefix(), "scgpt_rf")

    def test_bulkformer_accepts_signed_normalized_expression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            expression_path = root / "gdsc_expression.csv"
            gene_info_path = root / "bulkformer_gene_info.csv"
            pd.DataFrame(
                [[3.0, 4.0], [-1.8, 2.0]],
                index=["cell-b", "cell-a"],
                columns=["GENE1", "GENE2"],
            ).to_csv(expression_path)
            pd.DataFrame(
                {
                    "gene_symbol": ["GENE1", "GENE2"],
                    "ensg_id": ["ENSG1", "ENSG2"],
                }
            ).to_csv(gene_info_path, index=False)

            runner = object.__new__(DrugRespBulkFormerPCARFRunner)
            runner.task_cfg = SimpleNamespace(
                bulkformer_expected_gene_count=2,
                bulkformer_max_expected_expression=30.0,
            )
            runner._canonical_cell_ids = ["cell-a", "cell-b"]

            expression, missing_fraction = runner._prepare_bulkformer_expression(
                {"expression": expression_path, "gene_info": gene_info_path}
            )

            np.testing.assert_array_equal(
                expression,
                np.asarray([[-1.8, 2.0], [3.0, 4.0]], dtype=np.float32),
            )
            self.assertEqual(missing_fraction, 0.0)

    def test_bulkformer_cleanup_does_not_wait_for_master_only_rf(self) -> None:
        runner = object.__new__(DrugRespBulkFormerPCARFRunner)
        runner.is_distributed = True
        runner._setup_runtime = Mock(side_effect=RuntimeError("stop after setup"))

        with (
            patch(
                "finetune.drug_resp.bulkformer_pca_rf_runner.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "finetune.drug_resp.bulkformer_pca_rf_runner.dist.barrier"
            ) as barrier,
            patch(
                "finetune.drug_resp.bulkformer_pca_rf_runner.dist.destroy_process_group"
            ) as destroy_process_group,
            self.assertRaisesRegex(RuntimeError, "stop after setup"),
        ):
            runner.run()

        barrier.assert_not_called()
        destroy_process_group.assert_called_once_with()

    def test_resume_reuses_only_complete_benchmark_compatible_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            out_dir = Path(temporary_directory)
            runner = object.__new__(DrugRespRunner)
            runner.task_cfg = SimpleNamespace(
                resume_completed_models=True,
                cv_folds=2,
                epochs=3,
                warmup_epochs=2,
                batch_size=4,
                grad_accumulation_steps=4,
            )
            runner.model_cfg = SimpleNamespace()
            runner.world_size = 4
            runner._cv_fold_fingerprint = "shared-folds"
            runner._task_output_dir = lambda: out_dir
            runner._output_prefix = lambda: "drug_resp_adapters"
            runner._finetune_mode = lambda: "adapters"
            checkpoint_paths = {"pretrain_sc": "checkpoint.pth"}
            metadata = {
                "task": "drug_resp",
                "finetune_mode": "adapters",
                "cv_folds": 2,
                "cv_fold_fingerprint": "shared-folds",
                "epochs": 3,
                "warmup_epochs": 2,
                "batch_size_per_gpu": 4,
                "gradient_accumulation_steps": 4,
                "world_size": 4,
                "checkpoint_paths": checkpoint_paths,
            }
            (out_dir / "drug_resp_adapters_run_metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            (out_dir / "drug_resp_adapters_config.yaml").write_text(
                "pretrain: {}\nfinetune:\n  drug_resp: {}\n",
                encoding="utf-8",
            )

            def write_rows(name: str, rows: list[dict]) -> None:
                runner._write_csv(
                    out_dir / f"drug_resp_adapters_pretrain_sc_{name}.csv",
                    rows,
                )

            write_rows(
                "evaluation_metrics",
                [{"model": "pretrain_sc", "checkpoint_path": "checkpoint.pth"}],
            )
            write_rows(
                "fold_metrics",
                [
                    {"model": "pretrain_sc", "fold": fold}
                    for fold in (1, 2)
                ],
            )
            write_rows(
                "cell_line_metrics",
                [{"model": "pretrain_sc", "cell_line_id": "cell-a"}],
            )
            write_rows(
                "training_curves",
                [
                    {"model": "pretrain_sc", "fold": fold, "epoch": epoch}
                    for fold in (1, 2)
                    for epoch in (1, 2, 3)
                ],
            )

            completed = runner._load_completed_model_results(checkpoint_paths)
            self.assertEqual(set(completed), {"pretrain_sc"})

    def test_submission_matrix_has_expected_jobs(self) -> None:
        job_dir = REPO.parent / "job_files" / "finetune" / "drug_resp"
        expected = {
            "drug_resp_head_only-job.sh",
            "drug_resp_adapters-job.sh",
            "drug_resp_full_ft-job.sh",
            "drug_resp_pca_rf-job.sh",
            "drug_resp_raw_mlp_all_genes-job.sh",
            "drug_resp_raw_mlp_hvg1199-job.sh",
            "drug_resp_raw_mlp_drug_only-job.sh",
            "drug_resp_raw_pca_rf_all_genes-job.sh",
            "drug_resp_raw_pca_rf_hvg1199-job.sh",
            "drug_resp_bulkformer_pca_rf-job.sh",
            "drug_resp_scgpt_pca_rf-job.sh",
            "drug_resp_scgpt_preadapt_pca_rf-job.sh",
        }
        expected.update(
            f"drug_resp_head_only_pretrain_bulk_{size}-job.sh"
            for size in ("10k", "50k", "100k", "200k", "400k")
        )
        self.assertTrue(expected.issubset({path.name for path in job_dir.glob("*.sh")}))
        self.assertTrue((job_dir / "drug_resp_raw_mlp_drug_only-job.sh").exists())
        self.assertFalse((job_dir / "drug_resp_raw_mlp_hvg1199_deep-job.sh").exists())


if __name__ == "__main__":
    unittest.main()
