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

from finetune.disease_class import (  # noqa: E402
    DiseaseClassBulkFormerPCARFRunner,
    DiseaseClassPCARFRunner,
    DiseaseClassRawMLPRunner,
    DiseaseClassRawPCARFRunner,
    DiseaseClassRunner,
    DiseaseClassScGPTPCARFRunner,
)


class DiseaseClassSetupTest(unittest.TestCase):
    def test_all_evaluation_runners_share_disease_task_identity(self) -> None:
        for runner_class in (
            DiseaseClassRunner,
            DiseaseClassPCARFRunner,
            DiseaseClassRawMLPRunner,
            DiseaseClassRawPCARFRunner,
            DiseaseClassBulkFormerPCARFRunner,
            DiseaseClassScGPTPCARFRunner,
        ):
            self.assertEqual(runner_class.task_name, "disease_class")
            self.assertEqual(runner_class.config_node, "disease_class")

    def test_loader_keeps_cases_and_uses_stable_gsm_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "disignatlas.h5ad"
            obs = pd.DataFrame(
                {
                    "GSM_ID": ["GSM1", "GSM2", "GSM3"],
                    "label": ["Disease A", "Control", "Disease B"],
                    "annot": ["Study 1", "Study 1", "Study 2"],
                    "binary_label": ["case", "control", "case"],
                },
                index=["row-1", "row-2", "row-3"],
            )
            ad.AnnData(
                X=np.ones((3, 2), dtype=np.float32),
                obs=obs,
                var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
            ).write_h5ad(data_path)

            for runner_class in (
                DiseaseClassRunner,
                DiseaseClassPCARFRunner,
                DiseaseClassRawMLPRunner,
                DiseaseClassRawPCARFRunner,
                DiseaseClassBulkFormerPCARFRunner,
                DiseaseClassScGPTPCARFRunner,
            ):
                runner = object.__new__(runner_class)
                runner.task_cfg = SimpleNamespace(
                    disignatlas_data_path=str(data_path),
                    disease_label_col="label",
                    binary_label_col="binary_label",
                    disease_sample_id_col="GSM_ID",
                )
                loaded = runner._load_disignatlas()

                self.assertEqual(loaded.obs_names.tolist(), ["GSM1", "GSM3"])
                self.assertEqual(
                    loaded.obs["disease_label"].tolist(),
                    ["Disease A", "Disease B"],
                )
                self.assertEqual(loaded.obs["sample_id"].tolist(), ["GSM1", "GSM3"])

    def test_checkpoint_subset_uses_standard_classification_behavior(self) -> None:
        runner = object.__new__(DiseaseClassRunner)
        runner.task_cfg = SimpleNamespace(
            finetune_mode="head_only",
            output_suffix="pretrain_bulk_10k",
            model_keys=["pretrain_bulk"],
            pretrained_model_paths={"pretrain_bulk": "checkpoint.pth"},
        )

        self.assertEqual(set(runner._get_checkpoint_paths()), {"pretrain_bulk"})
        self.assertEqual(
            runner._output_variant(),
            "head_only_pretrain_bulk_10k",
        )

    def test_submission_matrix_has_expected_jobs(self) -> None:
        job_dir = REPO.parent / "job_files" / "finetune" / "disease_class"
        expected = {
            "disease_class_head_only-job.sh",
            "disease_class_adapters-job.sh",
            "disease_class_full_ft-job.sh",
            "disease_class_pca_rf-job.sh",
            "disease_class_raw_mlp_all_genes-job.sh",
            "disease_class_raw_mlp_hvg1199-job.sh",
            "disease_class_raw_pca_rf_all_genes-job.sh",
            "disease_class_raw_pca_rf_hvg1199-job.sh",
            "disease_class_bulkformer_pca_rf-job.sh",
            "disease_class_scgpt_pca_rf-job.sh",
            "disease_class_scgpt_preadapt_pca_rf-job.sh",
        }
        expected.update(
            f"disease_class_head_only_pretrain_bulk_{size}-job.sh"
            for size in ("10k", "50k", "100k", "200k", "400k")
        )
        observed = {path.name for path in job_dir.glob("*.sh")}
        self.assertTrue(expected.issubset(observed))
        self.assertFalse(
            (job_dir / "disease_class_raw_mlp_hvg1199_deep-job.sh").exists()
        )


if __name__ == "__main__":
    unittest.main()
