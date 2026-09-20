import csv
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from finetune.gene_essent import (  # noqa: E402
    GeneEssentBulkFormerPCARFRunner,
    GeneEssentPCARFRunner,
    GeneEssentRawMLPRunner,
    GeneEssentRawPCARFRunner,
    GeneEssentRunner,
    GeneEssentScGPTPCARFRunner,
)
from finetune.gene_essent.runner import (  # noqa: E402
    CancerFoundationGeneEssentModel,
    GeneEssentPredHead,
)


class _IdentityBackbone(torch.nn.Module):
    def forward(self, gene_ids, expr, src_key_padding_mask=None):
        del gene_ids, expr, src_key_padding_mask
        return torch.tensor(
            [[[100.0, 100.0], [1.0, 2.0], [3.0, 4.0]]],
            dtype=torch.float32,
        )


class GeneEssentSetupTest(unittest.TestCase):
    def test_all_evaluation_runners_use_gene_essentiality_identity(self) -> None:
        for runner_class in (
            GeneEssentRunner,
            GeneEssentPCARFRunner,
            GeneEssentRawMLPRunner,
            GeneEssentRawPCARFRunner,
            GeneEssentBulkFormerPCARFRunner,
            GeneEssentScGPTPCARFRunner,
        ):
            self.assertEqual(runner_class.task_name, "gene_essent")

    def test_neural_head_uses_only_each_contextualized_gene_state(self) -> None:
        head = GeneEssentPredHead(embedding_dim=2, hidden_dim=2, bottleneck_dim=2)
        with torch.no_grad():
            for module in head.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.fill_(1.0)
                    module.bias.zero_()
        model = CancerFoundationGeneEssentModel(_IdentityBackbone(), head)
        predictions = model(
            {
                "gene_ids": torch.zeros((1, 3), dtype=torch.long),
                "expr": torch.zeros((1, 3), dtype=torch.float32),
            }
        )
        self.assertEqual(tuple(predictions.shape), (1, 2))
        self.assertLess(float(predictions[0, 0]), float(predictions[0, 1]))

    def test_mad_selection_is_training_fold_only_and_deterministic(self) -> None:
        runner = object.__new__(GeneEssentRunner)
        runner.selected_gene_count = 2
        expression = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 10.0, 4.0],
                [0.0, 4.0, 0.0, 8.0],
                [0.0, 1000.0, 1000.0, 0.0],
            ],
            dtype=np.float32,
        )
        selected = runner._select_training_hvg_indices(
            expression,
            np.asarray([0, 1, 2]),
        )
        np.testing.assert_array_equal(selected, np.asarray([1, 3]))

    def test_cell_line_fold_manifest_is_reused_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "folds.csv"
            runner = object.__new__(GeneEssentRunner)
            runner.task_cfg = SimpleNamespace(
                cv_folds=5,
                random_seed=42,
                cv_fold_manifest_path=str(manifest),
            )
            runner.is_master = True
            runner.is_distributed = False
            cell_ids = [f"cell-{index}" for index in range(10)]

            first = runner._build_or_load_cv_splits(cell_ids)
            first_fingerprint = runner._cv_fold_fingerprint
            second = runner._build_or_load_cv_splits(cell_ids)

            self.assertEqual(first_fingerprint, runner._cv_fold_fingerprint)
            for left, right in zip(first, second):
                np.testing.assert_array_equal(left[0], right[0])
                np.testing.assert_array_equal(left[1], right[1])
            with manifest.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 10)

    def test_gene_pair_metrics_average_correlations_over_cell_lines(self) -> None:
        targets = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        predictions = np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        metrics = GeneEssentPCARFRunner._metrics_from_matrices(
            predictions,
            targets,
        )
        self.assertAlmostEqual(float(metrics["pcc"]), 0.0, places=6)
        self.assertEqual(len(metrics["pcc_per_cell_line"]), 2)

    def test_final_correlation_mean_is_over_cell_lines_not_fold_means(self) -> None:
        rows = [{"pcc": 1.0}, {"pcc": 1.0}, {"pcc": -1.0}]
        self.assertAlmostEqual(
            GeneEssentRunner._mean_cell_line_metric(rows, "pcc"),
            1.0 / 3.0,
        )

    def test_checkpoint_subset_supports_bulk_size_runs(self) -> None:
        runner = object.__new__(GeneEssentRunner)
        runner.task_cfg = SimpleNamespace(
            model_keys=["pretrain_bulk"],
            pretrained_model_paths={"pretrain_bulk": "checkpoint.pth"},
        )
        self.assertEqual(set(runner._get_checkpoint_paths()), {"pretrain_bulk"})

    def test_bulkformer_accepts_bounded_signed_depmap_expression(self) -> None:
        expression = np.asarray(
            [[-2.62, -0.12, 0.0], [1.25, 9.16, 17.30]],
            dtype=np.float32,
        )
        observed_min, observed_max = (
            GeneEssentBulkFormerPCARFRunner._validate_bulkformer_expression_scale(
                expression,
                absolute_limit=30.0,
            )
        )
        self.assertAlmostEqual(observed_min, -2.62, places=5)
        self.assertAlmostEqual(observed_max, 17.30, places=5)

    def test_bulkformer_rejects_depmap_expression_on_wrong_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "bounded magnitude"):
            GeneEssentBulkFormerPCARFRunner._validate_bulkformer_expression_scale(
                np.asarray([[-31.0, 1.0]], dtype=np.float32),
                absolute_limit=30.0,
            )

    def test_external_pca_rf_stops_distributed_work_before_cpu_fit(self) -> None:
        for runner_class in (
            GeneEssentScGPTPCARFRunner,
            GeneEssentBulkFormerPCARFRunner,
        ):
            source = inspect.getsource(runner_class.run)
            shutdown = source.index("self._finish_distributed_embedding_phase()")
            cpu_fit = source.index("self._fit_predict_gene_features(", shutdown)
            self.assertLess(shutdown, cpu_fit, runner_class.__name__)
            self.assertNotIn("dist.all_reduce", source, runner_class.__name__)



if __name__ == "__main__":
    unittest.main()
