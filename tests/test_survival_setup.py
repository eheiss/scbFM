import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from finetune.surv_pred import (  # noqa: E402
    SurvPredBulkFormerPCARFRunner,
    SurvPredPCARFRunner,
    SurvPredRawMLPRunner,
    SurvPredRawPCARFRunner,
    SurvPredRunner,
    SurvPredScGPTPCARFRunner,
)
from finetune.surv_pred_binary import (  # noqa: E402
    SurvPredBinaryBulkFormerPCARFRunner,
    SurvPredBinaryPCARFRunner,
    SurvPredBinaryRawMLPRunner,
    SurvPredBinaryRawPCARFRunner,
    SurvPredBinaryRunner,
    SurvPredBinaryScGPTPCARFRunner,
)
from finetune.surv_pred_survboard import (  # noqa: E402
    SurvPredSurvBoardBulkFormerPCARFRunner,
    SurvPredSurvBoardPCARFRunner,
    SurvPredSurvBoardRawMLPRunner,
    SurvPredSurvBoardRawPCARFRunner,
    SurvPredSurvBoardRunner,
    SurvPredSurvBoardScGPTPCARFRunner,
)
from finetune.surv_pred_survboard.runner import (  # noqa: E402
    cox_partial_log_likelihood,
    d_calibration,
)


class SurvivalSetupTest(unittest.TestCase):
    def test_all_runner_families_have_complete_task_identity(self) -> None:
        families = {
            "surv_pred": (
                SurvPredRunner,
                SurvPredRawMLPRunner,
                SurvPredRawPCARFRunner,
                SurvPredPCARFRunner,
                SurvPredScGPTPCARFRunner,
                SurvPredBulkFormerPCARFRunner,
            ),
            "surv_pred_binary": (
                SurvPredBinaryRunner,
                SurvPredBinaryRawMLPRunner,
                SurvPredBinaryRawPCARFRunner,
                SurvPredBinaryPCARFRunner,
                SurvPredBinaryScGPTPCARFRunner,
                SurvPredBinaryBulkFormerPCARFRunner,
            ),
            "surv_pred_survboard": (
                SurvPredSurvBoardRunner,
                SurvPredSurvBoardRawMLPRunner,
                SurvPredSurvBoardRawPCARFRunner,
                SurvPredSurvBoardPCARFRunner,
                SurvPredSurvBoardScGPTPCARFRunner,
                SurvPredSurvBoardBulkFormerPCARFRunner,
            ),
        }
        for task, classes in families.items():
            for runner_class in classes:
                self.assertEqual(runner_class.task_name, task)

    def test_cox_loss_uses_breslow_tie_denominator(self) -> None:
        hazards = torch.tensor([0.0, np.log(2.0), 0.0], dtype=torch.float64)
        times = torch.tensor([2.0, 2.0, 1.0], dtype=torch.float64)
        events = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
        expected = -(np.log(1.0) + np.log(2.0) - 2.0 * np.log(4.0)) / 2.0
        self.assertAlmostEqual(
            float(cox_partial_log_likelihood(hazards, times, events)),
            expected,
            places=8,
        )

    def test_survboard_d_calibration_defaults_to_five_bins(self) -> None:
        probabilities = np.asarray(
            [[0.95, 0.75], [0.8, 0.4], [0.7, 0.2]], dtype=float
        )
        statistic, p_value = d_calibration(
            probabilities,
            np.asarray([1.0, 2.0]),
            np.asarray([1.0, 2.0, 2.0]),
            np.asarray([1, 0, 1]),
        )
        self.assertTrue(np.isfinite(statistic))
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_pan_survival_uses_five_disjoint_stratified_folds(self) -> None:
        runner = object.__new__(SurvPredRunner)
        runner.task_cfg = SimpleNamespace(
            cv_folds=5,
            random_seed=42,
        )
        labels = np.repeat(np.asarray(["A", "B", "C"]), 20)
        splits = runner._build_cv_splits(labels)
        self.assertEqual(len(splits), 5)
        all_test_indices = []
        for train_idx, test_idx in splits:
            self.assertEqual(len(train_idx), 48)
            self.assertEqual(len(test_idx), 12)
            self.assertEqual(set(labels[test_idx]), {"A", "B", "C"})
            all_test_indices.extend(test_idx.tolist())
        self.assertEqual(sorted(all_test_indices), list(range(len(labels))))

    def test_pan_stratifies_by_cohort_and_event_when_possible(self) -> None:
        runner = object.__new__(SurvPredRunner)
        runner.task_cfg = SimpleNamespace(cv_folds=5)
        cohorts = np.repeat(np.asarray(["A", "B"]), 10)
        events = np.tile(np.repeat(np.asarray([0, 1]), 5), 2)
        labels = runner._survival_stratification_labels(events, cohorts)
        self.assertEqual(set(labels), {"A|0", "A|1", "B|0", "B|1"})

    def test_pan_metadata_clears_inherited_survboard_protocol_fields(self) -> None:
        source = (SRC / "finetune" / "surv_pred" / "runner.py").read_text()
        self.assertIn('"cv_folds": int(getattr(self.task_cfg, "cv_folds", 5))', source)
        self.assertIn('"n_outer_splits": None', source)
        self.assertIn('"survboard_split_fingerprint": None', source)

    def test_survboard_split_reader_skips_csv_header_and_hashes_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split_dir = root / "splits" / "TCGA"
            split_dir.mkdir(parents=True)
            pd.DataFrame([[0, 1, np.nan], [1, 2, np.nan]]).to_csv(
                split_dir / "BRCA_train_splits.csv", index=False
            )
            pd.DataFrame([[2, np.nan], [0, np.nan]]).to_csv(
                split_dir / "BRCA_test_splits.csv", index=False
            )
            runner = object.__new__(SurvPredSurvBoardRunner)
            runner.task_cfg = SimpleNamespace(
                survboard_data_dir=str(root),
                cancer="BRCA",
                project="TCGA",
                expected_outer_splits=2,
            )
            train, test = runner._load_splits(3)
            self.assertEqual(len(train), 2)
            np.testing.assert_array_equal(train[0], np.asarray([0, 1]))
            np.testing.assert_array_equal(test[1], np.asarray([0]))
            self.assertEqual(len(runner._survboard_split_fingerprint), 64)

    def test_checkpoint_subset_supports_bulk_size_models(self) -> None:
        for runner_class in (SurvPredRunner, SurvPredBinaryRunner):
            runner = object.__new__(runner_class)
            runner.task_cfg = SimpleNamespace(
                model_keys=["pretrain_bulk"],
                pretrained_model_paths={"pretrain_bulk": "checkpoint.pth"},
            )
            self.assertEqual(
                runner._get_checkpoint_paths(), {"pretrain_bulk": "checkpoint.pth"}
            )

    def test_config_and_container_contracts(self) -> None:
        for config_name in (
            "surv_pred.yaml",
            "surv_pred_binary.yaml",
            "surv_pred_survboard.yaml",
        ):
            text = (SRC / "configs" / "finetune" / config_name).read_text()
            self.assertIn("hvg_selection_method: mad", text)
            self.assertIn("warmup_epochs: 2", text)
            self.assertNotIn("first_cycle_steps:", text)
            self.assertNotIn("warmup_steps:", text)
        pan_config = (SRC / "configs" / "finetune" / "surv_pred.yaml").read_text()
        self.assertIn("cv_folds: 5", pan_config)
        self.assertNotIn("cv_repeats:", pan_config)
        self.assertNotIn("cv_test_size:", pan_config)
        binary_config = (
            SRC / "configs" / "finetune" / "surv_pred_binary.yaml"
        ).read_text()
        self.assertIn("cv_folds: 5", binary_config)
        for definition in ("scbfm.def", "scgpt.def", "bulkformer.def"):
            self.assertNotIn(
                "scikit-survival", (REPO / "cluster" / definition).read_text()
            )
        survival_source = "\n".join(
            path.read_text()
            for root in (
                SRC / "finetune" / "surv_pred",
                SRC / "finetune" / "surv_pred_survboard",
            )
            for path in root.glob("*.py")
        )
        survival_source += (SRC / "finetune" / "survival_regression.py").read_text()
        self.assertNotIn("sksurv", survival_source)
        self.assertNotIn("from pycox", survival_source)

    def test_submission_matrices_are_complete_without_deep_mlp(self) -> None:
        job_root = REPO.parent / "job_files" / "finetune"
        common_variants = {
            "head_only",
            "adapters",
            "full_ft",
            "pca_rf",
            "raw_mlp_all_genes",
            "raw_mlp_hvg1199",
            "raw_pca_rf_all_genes",
            "raw_pca_rf_hvg1199",
            "bulkformer_pca_rf",
            "scgpt_pca_rf",
            "scgpt_preadapt_pca_rf",
        }
        for task in ("surv_pred", "surv_pred_binary", "surv_pred_survboard"):
            names = {path.name for path in (job_root / task).glob("*.sh")}
            expected = {f"{task}_{variant}-job.sh" for variant in common_variants}
            expected.update(
                f"{task}_head_only_pretrain_bulk_{size}-job.sh"
                for size in ("10k", "50k", "100k", "200k", "400k")
            )
            self.assertTrue(expected.issubset(names), expected.difference(names))
            self.assertFalse(any("deep" in name for name in names))

        array_job = (
            job_root
            / "surv_pred_survboard"
            / "surv_pred_survboard_head_only-job.sh"
        ).read_text()
        self.assertIn("#SBATCH --array=0-20", array_job)
        common = (
            job_root
            / "surv_pred_survboard"
            / "surv_pred_survboard_common.sh"
        ).read_text()
        self.assertIn("BLCA BRCA", common)
        self.assertIn("GBM READ", common)
        self.assertIn("survboard_repeated_five_fold_cross_validation", common)
        self.assertIn("n_outer_splits", common)

    def test_notebook_has_all_three_survival_comparison_sections(self) -> None:
        notebook = (SRC / "analysis" / "task_performances.ipynb").read_text()
        for task in ("surv_pred", "surv_pred_binary", "surv_pred_survboard"):
            self.assertIn(f'task = \\\\"{task}\\\\"', notebook)
        self.assertIn("Raw PCA+RF", notebook)
        self.assertIn("Raw PCA+RF", notebook)
        self.assertIn("head-only performance by pretraining size", notebook)
        self.assertIn("Held-out loss", notebook)


if __name__ == "__main__":
    unittest.main()
