import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from finetune.deconv import (  # noqa: E402
    DeconvBulkFormerPCARFRunner,
    DeconvPCARFRunner,
    DeconvRawMLPRunner,
    DeconvRawPCARFRunner,
    DeconvRunner,
    DeconvScGPTPCARFRunner,
)
from finetune.deconv.runner import DeconvDataset  # noqa: E402


class DeconvSetupTest(unittest.TestCase):
    def test_all_evaluation_runners_use_deconv_identity(self) -> None:
        for runner_class in (
            DeconvRunner,
            DeconvPCARFRunner,
            DeconvRawMLPRunner,
            DeconvRawPCARFRunner,
            DeconvBulkFormerPCARFRunner,
            DeconvScGPTPCARFRunner,
        ):
            self.assertEqual(runner_class.task_name, "deconv")

    def test_mad_selection_is_training_fold_only_and_deterministic(self) -> None:
        runner = object.__new__(DeconvRunner)
        runner.selected_gene_count = 2
        runner.task_cfg = SimpleNamespace(hvg_selection_method="mad")
        adata = ad.AnnData(
            X=np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 10.0, 4.0],
                    [0.0, 4.0, 0.0, 8.0],
                ],
                dtype=np.float32,
            )
        )

        selected = runner._select_training_hvg_indices(adata)

        np.testing.assert_array_equal(selected, np.asarray([1, 3]))

    def test_dataset_uses_compact_selected_expression_with_original_gene_ids(self) -> None:
        dataset = DeconvDataset(
            np.asarray([[5.0, 7.0]], dtype=np.float32),
            np.asarray([[0.25, 0.75]], dtype=np.float32),
            bin_num=51,
            cls_gene_id=0,
            gene_token_offset=2,
            cls_value=-2.0,
            selected_gene_count=2,
            seed=42,
            do_binning=False,
            fixed_gene_indices=np.asarray([1, 3]),
        )

        features, target = dataset[0]

        np.testing.assert_array_equal(features["gene_ids"].numpy(), [0, 3, 5])
        np.testing.assert_allclose(features["expr"].numpy(), [-2.0, 5.0, 7.0])
        np.testing.assert_allclose(target.numpy(), [0.25, 0.75])

    def test_raw_pseudobulk_is_log1p_transformed_after_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "pseudo.h5ad"
            gene_path = root / "genes.txt"
            gene_path.write_text("ENSG1\nENSG2\n", encoding="utf-8")
            obs = pd.DataFrame(
                {
                    "dataset_id": ["d1", "d2"],
                    "donor_id": ["a", "b"],
                    "tissue_general": ["t1", "t2"],
                    "prop__A": [0.25, 0.75],
                    "prop__B": [0.75, 0.25],
                },
                index=["sample-1", "sample-2"],
            )
            ad.AnnData(
                X=np.asarray([[0.0, 3.0], [8.0, 0.0]], dtype=np.float32),
                obs=obs,
                var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
            ).write_h5ad(data_path)
            runner = object.__new__(DeconvRunner)
            runner.task_cfg = SimpleNamespace(
                pseudo_bulk_data_path=str(data_path),
                gene_list_path=str(gene_path),
                preprocess=True,
                split_by_context=True,
                context_columns=["dataset_id", "donor_id", "tissue_general"],
            )
            runner.model_cfg = SimpleNamespace(gene_num=2, bin_num=51)

            adata, targets, groups = runner._prepare_cv_data()

            np.testing.assert_allclose(
                np.asarray(adata.X),
                np.log1p(np.asarray([[0.0, 3.0], [8.0, 0.0]], dtype=np.float32)),
            )
            self.assertEqual(targets.shape, (2, 2))
            self.assertEqual(groups.tolist(), ["d1||a||t1", "d2||b||t2"])

    def test_grouped_fold_manifest_is_reused_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "folds.csv"
            adata = ad.AnnData(
                X=np.ones((10, 2), dtype=np.float32),
                obs=pd.DataFrame(index=[f"sample-{index}" for index in range(10)]),
            )
            groups = np.asarray([f"group-{index // 2}" for index in range(10)])
            runner = object.__new__(DeconvRunner)
            runner.task_cfg = SimpleNamespace(
                cv_folds=5,
                random_seed=42,
                cv_fold_manifest_path=str(manifest),
            )
            runner.rank = 0

            first = runner._build_or_load_cv_splits(adata, groups)
            first_fingerprint = runner._cv_fold_fingerprint
            second = runner._build_or_load_cv_splits(adata, groups)

            self.assertEqual(first_fingerprint, runner._cv_fold_fingerprint)
            for left, right in zip(first, second):
                np.testing.assert_array_equal(left[0], right[0])
                np.testing.assert_array_equal(left[1], right[1])
            with manifest.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 10)

    def test_rf_predictions_are_projected_to_probability_simplex(self) -> None:
        predictions = np.asarray([[1.5, -0.5], [0.0, 0.0]], dtype=np.float32)
        normalized = DeconvRunner._normalize_composition_predictions(
            predictions,
            np.asarray([0.25, 0.75], dtype=np.float32),
        )

        np.testing.assert_allclose(normalized.sum(axis=1), 1.0)
        self.assertTrue(np.all(normalized >= 0))
        np.testing.assert_allclose(normalized[1], [0.25, 0.75])

    def test_checkpoint_subset_supports_bulk_size_runs(self) -> None:
        runner = object.__new__(DeconvRunner)
        runner.task_cfg = SimpleNamespace(
            model_keys=["pretrain_bulk"],
            pretrained_model_paths={"pretrain_bulk": "checkpoint.pth"},
        )
        self.assertEqual(set(runner._get_checkpoint_paths()), {"pretrain_bulk"})

    def test_submission_matrix_has_expected_jobs(self) -> None:
        job_dir = REPO.parent / "job_files" / "finetune" / "deconv"
        expected = {
            "deconv_head_only-job.sh",
            "deconv_adapters-job.sh",
            "deconv_full_ft-job.sh",
            "deconv_pca_rf-job.sh",
            "deconv_raw_mlp_all_genes-job.sh",
            "deconv_raw_mlp_hvg1199-job.sh",
            "deconv_raw_pca_rf_all_genes-job.sh",
            "deconv_raw_pca_rf_hvg1199-job.sh",
            "deconv_bulkformer_pca_rf-job.sh",
            "deconv_scgpt_pca_rf-job.sh",
            "deconv_scgpt_preadapt_pca_rf-job.sh",
        }
        expected.update(
            f"deconv_head_only_pretrain_bulk_{size}-job.sh"
            for size in ("10k", "50k", "100k", "200k", "400k")
        )
        observed = {path.name for path in job_dir.glob("*.sh")}
        self.assertTrue(expected.issubset(observed))
        self.assertFalse((job_dir / "deconv_raw_mlp_hvg1199_deep-job.sh").exists())
        for path in job_dir.glob("*-job.sh"):
            text = path.read_text(encoding="utf-8")
            requested_memory = int(text.split("#SBATCH --mem=", 1)[1].split("G", 1)[0])
            self.assertGreaterEqual(requested_memory, 128, path.name)


if __name__ == "__main__":
    unittest.main()
