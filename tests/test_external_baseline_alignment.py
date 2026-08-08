import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from finetune.canc_type_class.bulkformer_pca_rf_runner import (  # noqa: E402
    CancTypeClassBulkFormerPCARFRunner,
)
from finetune.canc_type_class.runner import CancTypeClassRunner  # noqa: E402
from finetune.canc_type_class.scgpt_pca_rf_runner import (  # noqa: E402
    CancTypeClassScGPTPCARFRunner,
    _ScGPTExpressionDataset,
    _bin_scgpt_examples_safely,
)
from utils import SequentialDistributedSampler  # noqa: E402


class _FakeVocab:
    def __init__(self, tokens: dict[str, int]) -> None:
        self.tokens = tokens

    def __contains__(self, token: str) -> bool:
        return token in self.tokens

    def __getitem__(self, token: str) -> int:
        return self.tokens[token]


class ExternalBaselineAlignmentTest(unittest.TestCase):
    def test_aligned_bulkformer_benchmarks_use_paper_max_pooling(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        workspace = repository.parent
        task_names = ("canc_type_class", "canc_type_class_33", "drug_resp")

        for task_name in task_names:
            config = (
                repository / "src" / "configs" / "finetune" / f"{task_name}.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("bulkformer_aggregate_type: max", config)

            job = (
                workspace
                / "job_files"
                / "finetune"
                / task_name
                / f"{task_name}_bulkformer_pca_rf-job.sh"
            ).read_text(encoding="utf-8")
            self.assertIn(f"finetune.{task_name}.bulkformer_aggregate_type=max", job)

    def test_scgpt_retains_selected_zero_genes_for_cls_only_profile(self) -> None:
        dataset = _ScGPTExpressionDataset(
            matrix=np.zeros((1, 3), dtype=np.float32),
            labels=np.asarray([2]),
            gene_ids=np.asarray([10, 11, 12]),
            cls_token_id=1,
            cls_value=-2.0,
        )

        example, label = dataset[0]

        np.testing.assert_array_equal(example["genes"].numpy(), [1, 10, 11, 12])
        np.testing.assert_array_equal(
            example["expressions"].numpy(),
            [-2.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(int(label), 2)

    def test_scgpt_keeps_nonzero_only_tokenization_for_normal_profile(self) -> None:
        dataset = _ScGPTExpressionDataset(
            matrix=np.asarray([[0.0, 4.0, 0.0]], dtype=np.float32),
            labels=np.asarray([1]),
            gene_ids=np.asarray([10, 11, 12]),
            cls_token_id=1,
            cls_value=-2.0,
        )

        example, _label = dataset[0]

        np.testing.assert_array_equal(example["genes"].numpy(), [1, 11])
        np.testing.assert_array_equal(example["expressions"].numpy(), [-2.0, 4.0])

    def test_scgpt_binning_bypasses_all_zero_values_only(self) -> None:
        examples = [
            {
                "genes": torch.tensor([1, 10, 11]),
                "expressions": torch.tensor([-2.0, 0.0, 0.0]),
            },
            {
                "genes": torch.tensor([1, 10]),
                "expressions": torch.tensor([-2.0, 4.0]),
            },
        ]
        calls = []

        def fake_binning(*, row, n_bins):
            calls.append((row.clone(), n_bins))
            return torch.full_like(row, 7.0)

        binned = _bin_scgpt_examples_safely(examples, fake_binning)

        torch.testing.assert_close(
            binned[0]["expressions"],
            torch.tensor([-2.0, 0.0, 0.0]),
        )
        torch.testing.assert_close(
            binned[1]["expressions"],
            torch.tensor([-2.0, 7.0]),
        )
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(calls[0][0], torch.tensor([4.0]))
        self.assertEqual(calls[0][1], 51)
        torch.testing.assert_close(
            examples[1]["expressions"],
            torch.tensor([-2.0, 4.0]),
        )

    def test_scgpt_model_key_distinguishes_preadapted_checkpoint(self) -> None:
        published = object.__new__(CancTypeClassScGPTPCARFRunner)
        published.task_cfg = SimpleNamespace()
        preadapted = object.__new__(CancTypeClassScGPTPCARFRunner)
        preadapted.task_cfg = SimpleNamespace(scgpt_model_key="scgpt_preadapt")

        self.assertEqual(published._model_key(), "scgpt")
        self.assertEqual(preadapted._model_key(), "scgpt_preadapt")

    def test_distributed_sequential_sampler_preserves_order_after_truncation(self) -> None:
        dataset = list(range(10))
        gathered_indices = []
        for rank in range(4):
            sampler = SequentialDistributedSampler(
                dataset,
                batch_size=4,
                world_size=4,
                rank=rank,
            )
            gathered_indices.extend(list(sampler))

        self.assertEqual(gathered_indices[: len(dataset)], list(range(len(dataset))))

    def test_bulkformer_reorders_full_gene_data_to_canonical_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            canonical_path = root / "tcga.h5ad"
            full_path = root / "TCGA_cancer_data.h5ad"
            gene_info_path = root / "bulkformer_gene_info.csv"

            canonical_obs = pd.DataFrame(
                {
                    "sample_id": ["sample-a", "sample-b"],
                    "patient_id": ["patient-a", "patient-b"],
                    "project": ["BRCA", "GBM"],
                },
                index=["canonical-a", "canonical-b"],
            )
            canonical = ad.AnnData(
                X=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                obs=canonical_obs,
                var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
            )
            canonical.write_h5ad(canonical_path)

            full_obs = pd.DataFrame(
                {
                    "sample_id": ["sample-b", "sample-extra", "sample-a"],
                    "patient_id": ["patient-b", "patient-extra", "patient-a"],
                    "project": ["GBM", "LUAD", "BRCA"],
                },
                index=["raw-b", "raw-extra", "raw-a"],
            )
            full_var = pd.DataFrame(
                {"ensg_id": ["ENSG1.5", "ENSG2", "ENSG3"]},
                index=["gene-a", "gene-b", "gene-c"],
            )
            full = ad.AnnData(
                X=np.asarray(
                    [[30.0, 40.0, 50.0], [60.0, 70.0, 80.0], [10.0, 20.0, 25.0]],
                    dtype=np.float32,
                ),
                obs=full_obs,
                var=full_var,
            )
            full.write_h5ad(full_path)
            pd.DataFrame({"ensg_id": ["ENSG1", "ENSG2", "ENSG3"]}).to_csv(
                gene_info_path,
                index=False,
            )

            runner = object.__new__(CancTypeClassBulkFormerPCARFRunner)
            runner.task_cfg = SimpleNamespace(
                tcga_data_dir=str(canonical_path),
                cohorts=["BRCA", "GBM"],
                merge_gbm_lgg=True,
                bulkformer_expected_gene_count=3,
                bulkformer_max_expected_expression=100.0,
            )
            expression, labels, groups, aligned, missing_fraction = (
                runner._prepare_bulkformer_data(
                    {"tcga_data": full_path, "gene_info": gene_info_path}
                )
            )

            self.assertEqual(aligned.obs["sample_id"].tolist(), ["sample-a", "sample-b"])
            np.testing.assert_array_equal(
                expression,
                np.asarray([[10.0, 20.0, 25.0], [30.0, 40.0, 50.0]], dtype=np.float32),
            )
            np.testing.assert_array_equal(labels, np.asarray([0, 1]))
            self.assertIsNone(groups)
            self.assertEqual(missing_fraction, 0.0)
            self.assertEqual(runner._bulkformer_matched_gene_count, 3)
            self.assertEqual(runner._bulkformer_missing_gene_count, 0)

    def test_scgpt_maps_unique_vocab_gene_pool_before_fold_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            gene_info_path = Path(temporary_directory) / "gene_info.csv"
            pd.DataFrame(
                {
                    "ensg_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
                    "gene_symbol": ["A", "B", "A", "C"],
                }
            ).to_csv(gene_info_path, index=False)
            adata = ad.AnnData(
                X=np.ones((2, 4), dtype=np.float32),
                var=pd.DataFrame(index=["ENSG1", "ENSG2", "ENSG3", "ENSG4"]),
            )

            runner = object.__new__(CancTypeClassScGPTPCARFRunner)
            runner.task_cfg = SimpleNamespace(scgpt_expected_source_gene_count=4)
            runner.selected_gene_count = 2
            mapped = runner._add_scgpt_gene_symbols(
                adata,
                gene_info_path,
                _FakeVocab({"A": 10, "B": 11, "C": 12}),
            )

            self.assertEqual(mapped.n_vars, 3)
            self.assertEqual(mapped.var["scgpt_gene_symbol"].tolist(), ["A", "B", "C"])
            self.assertEqual(runner._scgpt_source_gene_count, 4)
            self.assertEqual(runner._scgpt_vocab_matched_gene_count, 3)


class SharedFoldManifestTest(unittest.TestCase):
    @staticmethod
    def _runner(manifest_path: Path) -> CancTypeClassRunner:
        runner = object.__new__(CancTypeClassRunner)
        runner.task_cfg = SimpleNamespace(
            cv_fold_manifest_path=str(manifest_path),
            cv_folds=2,
            random_seed=42,
            cohorts=["BRCA", "GBM"],
            merge_gbm_lgg=True,
        )
        runner.rank = 0
        return runner

    def test_base_runner_declares_cv_manifest_identity(self) -> None:
        self.assertEqual(CancTypeClassRunner.task_name, "canc_type_class")
        self.assertEqual(CancTypeClassRunner.config_node, "canc_type_class")

    def test_manifest_is_reused_without_recomputing_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "folds.csv"
            adata = ad.AnnData(
                X=np.ones((6, 1), dtype=np.float32),
                obs=pd.DataFrame(
                    {
                        "sample_id": [f"sample-{index}" for index in range(6)],
                        "patient_id": [f"patient-{index}" for index in range(6)],
                    },
                    index=[f"row-{index}" for index in range(6)],
                ),
            )
            labels = np.asarray(["A", "A", "A", "B", "B", "B"])

            creator = self._runner(manifest_path)
            created_splits = creator._build_or_load_cv_splits(adata, labels)
            self.assertTrue(manifest_path.exists())

            loader = self._runner(manifest_path)
            loader._build_cv_splits = lambda *_args, **_kwargs: self.fail(
                "Existing fold manifest should be loaded instead of recomputing splits."
            )
            loaded_splits = loader._build_or_load_cv_splits(adata, labels)

            for created, loaded in zip(created_splits, loaded_splits):
                np.testing.assert_array_equal(created[0], loaded[0])
                np.testing.assert_array_equal(created[1], loaded[1])
            self.assertEqual(creator._cv_fold_fingerprint, loader._cv_fold_fingerprint)


if __name__ == "__main__":
    unittest.main()
