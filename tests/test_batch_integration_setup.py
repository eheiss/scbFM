import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from finetune.batch_integration import BatchIntegrationRunner  # noqa: E402
from finetune.single_cell import FrozenSingleCellBenchmark  # noqa: E402


class BatchIntegrationSetupTest(unittest.TestCase):
    def test_task_uses_scgpt_zero_shot_protocol(self) -> None:
        self.assertEqual(BatchIntegrationRunner.task_name, "batch_integration")
        config = (SRC / "configs" / "finetune" / "batch_integration.yaml").read_text()
        self.assertIn("dataset_keys: [covid19, lung_kim]", config)
        self.assertIn("expected_reference_cells: 15997", config)
        self.assertIn("expected_query_cells: 4003", config)
        self.assertIn("expected_total_cells: 30472", config)
        self.assertIn("gene_selection_method: mad", config)
        self.assertIn("distributed_timeout_minutes: 360", config)
        self.assertIn("include_external_scgpt: true", config)
        self.assertNotIn("Neftel", config)
        self.assertNotIn("epochs:", config)
        self.assertNotIn("finetune_mode:", config)

    def test_gene_selection_uses_reference_cells_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            reference_path = directory / "reference.h5ad"
            query_path = directory / "query.h5ad"
            gene_list_path = directory / "gene_list.txt"
            gene_info_path = directory / "gene_info.csv"
            gene_list_path.write_text("ENSG1\nENSG2\nENSG3\nENSG4\n", encoding="utf-8")
            pd.DataFrame(
                {
                    "gene_symbol": ["G1", "G2", "G3", "G4"],
                    "ensg_id": ["ENSG1.5", "ENSG2", "ENSG3", "ENSG4"],
                }
            ).to_csv(gene_info_path, index=False)
            var = pd.DataFrame(
                {"gene_name": ["G1", "G2", "G3", "G4"]},
                index=["v1", "v2", "v3", "v4"],
            )
            ad.AnnData(
                X=np.asarray(
                    [
                        [0, 0, 1, 0],
                        [10, 0, 1, 0],
                        [0, 1, 1, 0],
                        [10, 1, 1, 0],
                    ],
                    dtype=np.float32,
                ),
                obs=pd.DataFrame(
                    {"celltype": ["A", "A", "B", "B"], "batch": ["1", "1", "2", "2"]},
                    index=["r1", "r2", "r3", "r4"],
                ),
                var=var,
            ).write_h5ad(reference_path)
            # G3 is extremely variable only in query cells and therefore must
            # not influence the selected reference-derived genes.
            ad.AnnData(
                X=np.asarray([[1, 0, 0, 0], [1, 0, 1000, 0]], dtype=np.float32),
                obs=pd.DataFrame(
                    {"celltype": ["A", "B"], "batch": ["3", "3"]},
                    index=["q1", "q2"],
                ),
                var=var.copy(),
            ).write_h5ad(query_path)

            cfg = OmegaConf.create(
                {
                    "pretrain": {
                        "gene_num": 4,
                        "selected_gene_count": 2,
                        "max_seq_len": 3,
                        "bin_num": 51,
                        "pad_value": -2.0,
                    }
                }
            )
            task_cfg = OmegaConf.create(
                {
                    "datasets": {
                        "toy": {
                            "display_name": "Toy",
                            "reference_path": str(reference_path),
                            "query_path": str(query_path),
                            "gene_key": "gene_name",
                            "label_key": "celltype",
                            "batch_key": "batch",
                            "preprocessing": "none",
                        }
                    },
                    "dataset_keys": ["toy"],
                    "gene_list_path": str(gene_list_path),
                    "gene_info_path": str(gene_info_path),
                    "gene_selection_method": "mad",
                }
            )
            benchmark = FrozenSingleCellBenchmark(cfg, task_cfg)
            bundle = benchmark.load_dataset("toy")

            self.assertEqual(bundle.reference.var_names.tolist(), ["ENSG1", "ENSG2"])
            self.assertEqual(bundle.query.var_names.tolist(), ["ENSG1", "ENSG2"])
            self.assertEqual(bundle.combined.n_obs, 6)
            self.assertEqual(
                bundle.manifest["selection_scope"].unique().tolist(),
                ["reference_cells_only"],
            )
            self.assertEqual(bundle.mapping_stats["shared_mapped_gene_count"], 4)

    def test_short_source_vocabulary_is_zero_filled_to_fixed_length(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            gene_list_path = directory / "gene_list.txt"
            gene_info_path = directory / "gene_info.csv"
            reference_path = directory / "reference.h5ad"
            query_path = directory / "query.h5ad"
            gene_list_path.write_text("ENSG1\nENSG2\nENSG3\n", encoding="utf-8")
            pd.DataFrame(
                {
                    "gene_symbol": ["G1", "G2", "G3"],
                    "ensg_id": ["ENSG1", "ENSG2", "ENSG3"],
                }
            ).to_csv(gene_info_path, index=False)
            var = pd.DataFrame({"gene_name": ["G1"]}, index=["v1"])
            obs = pd.DataFrame(
                {"celltype": ["A", "A"], "batch": ["1", "2"]},
                index=["c1", "c2"],
            )
            ad.AnnData(X=np.asarray([[1.0], [2.0]]), obs=obs, var=var).write_h5ad(
                reference_path
            )
            ad.AnnData(X=np.asarray([[3.0], [4.0]]), obs=obs, var=var).write_h5ad(
                query_path
            )
            cfg = OmegaConf.create(
                {
                    "pretrain": {
                        "gene_num": 3,
                        "selected_gene_count": 2,
                        "max_seq_len": 3,
                        "bin_num": 51,
                        "pad_value": -2.0,
                    }
                }
            )
            task_cfg = OmegaConf.create(
                {
                    "datasets": {
                        "toy": {
                            "reference_path": str(reference_path),
                            "query_path": str(query_path),
                            "gene_key": "gene_name",
                            "label_key": "celltype",
                            "batch_key": "batch",
                            "preprocessing": "none",
                        }
                    },
                    "dataset_keys": ["toy"],
                    "gene_list_path": str(gene_list_path),
                    "gene_info_path": str(gene_info_path),
                }
            )
            bundle = FrozenSingleCellBenchmark(cfg, task_cfg).load_dataset("toy")
            self.assertEqual(bundle.combined.n_vars, 2)
            self.assertEqual(int(bundle.combined.var["source_present"].sum()), 1)
            self.assertEqual(bundle.mapping_stats["zero_filled_gene_count"], 1)
            self.assertTrue(np.allclose(bundle.combined.X[:, 1], 0.0))

    def test_scgpt_score_is_computed_from_exact_five_metrics(self) -> None:
        raw = pd.DataFrame(
            {
                "Leiden NMI": [0.6],
                "Leiden ARI": [0.3],
                "Silhouette label": [0.9],
                "Silhouette batch": [0.8],
                "Graph connectivity": [0.4],
                "PCR comparison": [0.99],
            },
            index=["pretrain_sc"],
        )
        metrics, summary = BatchIntegrationRunner._scgpt_result_rows(
            raw, ["pretrain_sc"], "toy"
        )
        self.assertEqual({row["metric"] for row in metrics}, {
            "nmi_cell", "ari_cell", "asw_cell", "asw_batch", "graph_connectivity"
        })
        self.assertAlmostEqual(summary[0]["avg_bio"], 0.6)
        self.assertAlmostEqual(summary[0]["avg_batch"], 0.6)
        self.assertAlmostEqual(summary[0]["overall"], 0.6)

    def test_submission_files_use_official_scgpt_inputs(self) -> None:
        job = (REPO / "cluster" / "run-job.sh").read_text()
        config = (REPO / "src" / "configs" / "finetune" / "batch_integration.yaml").read_text()
        prepare = (REPO / "data" / "README.md").read_text()
        definition = (REPO / "cluster" / "scbfm_single_cell.def").read_text()
        self.assertIn("#SBATCH --gres=gpu:4", job)
        self.assertIn("SCBFM_SIF", job)
        self.assertIn("batch_covid_subsampled_train.h5ad", config)
        self.assertIn("sample_proc_lung_test.h5ad", config)
        self.assertIn("1jSPoPunGQOmd71vDsK0FS7UvmDhGdhQS", prepare)
        self.assertIn("1gbfO7VqxCOkfzgHAih6hO88zFv6pd8wO", prepare)
        self.assertIn("scib-metrics==0.5.1", definition)
        self.assertIn("import scib_metrics", definition)
        self.assertIn("python -m torch.distributed.run", job)
        self.assertNotIn("\n  torchrun ", job)
        self.assertNotIn("neftel", "\n".join([job.lower(), prepare.lower()]))


if __name__ == "__main__":
    unittest.main()
