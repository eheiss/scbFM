import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from finetune.canc_type_class_33 import (  # noqa: E402
    CancTypeClass33BulkFormerPCARFRunner,
    CancTypeClass33RawPCARFRunner,
    CancTypeClass33Runner,
    CancTypeClass33ScGPTPCARFRunner,
)


class CancTypeClass33ConfigurationTest(unittest.TestCase):
    def test_all_added_runners_share_the_33_type_task_identity(self) -> None:
        for runner_class in (
            CancTypeClass33Runner,
            CancTypeClass33RawPCARFRunner,
            CancTypeClass33BulkFormerPCARFRunner,
            CancTypeClass33ScGPTPCARFRunner,
        ):
            self.assertEqual(runner_class.task_name, "canc_type_class_33")
            self.assertEqual(runner_class.config_node, "canc_type_class_33")

    def test_checkpoint_subset_and_output_suffix_use_base_runner_behavior(self) -> None:
        runner = object.__new__(CancTypeClass33Runner)
        runner.task_cfg = SimpleNamespace(
            finetune_mode="head_only",
            output_suffix="pretrain_bulk_10k",
            model_keys=["pretrain_bulk"],
            pretrained_model_paths={"pretrain_bulk": "checkpoint.pth"},
        )

        self.assertEqual(
            set(runner._get_checkpoint_paths()),
            {"pretrain_bulk"},
        )
        self.assertEqual(
            runner._output_variant(),
            "head_only_pretrain_bulk_10k",
        )

    def test_loader_keeps_gbm_and_lgg_as_distinct_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "tcga.h5ad"
            obs = pd.DataFrame(
                {
                    "sample_id": ["sample-brca", "sample-gbm", "sample-lgg", "sample-x"],
                    "patient_id": ["patient-brca", "patient-gbm", "patient-lgg", "patient-x"],
                    "project": ["BRCA", "GBM", "LGG", "OTHER"],
                },
                index=["row-brca", "row-gbm", "row-lgg", "row-x"],
            )
            ad.AnnData(
                X=np.ones((4, 2), dtype=np.float32),
                obs=obs,
                var=pd.DataFrame(index=["ENSG1", "ENSG2"]),
            ).write_h5ad(data_path)

            runner = object.__new__(CancTypeClass33Runner)
            runner.task_cfg = SimpleNamespace(
                tcga_data_dir=str(data_path),
                cohorts=["BRCA", "GBM", "LGG"],
            )
            loaded = runner._load_tcga()

            self.assertEqual(
                loaded.obs["cancer_type"].tolist(),
                ["BRCA", "GBM", "LGG"],
            )


if __name__ == "__main__":
    unittest.main()
