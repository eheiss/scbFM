import sys
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from finetune.cell_type_annotation import CellTypeAnnotationRunner  # noqa: E402


class CellTypeAnnotationSetupTest(unittest.TestCase):
    def test_ten_neighbor_majority_vote(self) -> None:
        reference = np.asarray([[float(index), 0.0] for index in range(12)])
        query = np.asarray([[5.5, 0.0]])
        labels = np.asarray(["A"] * 7 + ["B"] * 5)
        predictions, distances = CellTypeAnnotationRunner._predict_knn(
            reference,
            query,
            labels,
            n_neighbors=10,
        )
        self.assertEqual(predictions.tolist(), ["A"])
        self.assertEqual(distances.shape, (1, 10))

    def test_metrics_include_scgpt_and_benchmark_f1(self) -> None:
        summary, per_class = CellTypeAnnotationRunner._evaluate_predictions(
            dataset_key="toy",
            model_key="pretrain_sc",
            truth=np.asarray(["A", "A", "B", "B"]),
            predictions=np.asarray(["A", "B", "B", "B"]),
        )
        self.assertIn("macro_f1", summary)
        self.assertIn("weighted_f1", summary)
        self.assertEqual(summary["accuracy"], 0.75)
        self.assertEqual({row["cell_type"] for row in per_class}, {"A", "B"})

    def test_config_and_job_are_zero_shot(self) -> None:
        config = (SRC / "configs" / "finetune" / "cell_type_annotation.yaml").read_text()
        job_dir = REPO.parent / "job_files" / "finetune" / "batch_integration"
        job = (job_dir / "cell_type_annotation-job.sh").read_text()
        main = (SRC / "main.py").read_text()
        self.assertIn("n_neighbors: 10", config)
        self.assertIn("dataset_keys: [covid19, lung_kim]", config)
        self.assertIn("distributed_timeout_minutes: 360", config)
        self.assertNotIn("epochs:", config)
        self.assertNotIn("finetune_mode:", config)
        self.assertIn("task=finetune.cell_type_annotation", job)
        self.assertIn("#SBATCH --gres=gpu:4", job)
        self.assertIn(
            "/cluster/customapps/biomed/boeva/eheiss/singularity/scbfm_single_cell.sif",
            job,
        )
        self.assertIn("python -m torch.distributed.run", job)
        self.assertNotIn("\n  torchrun ", job)
        self.assertIn("CellTypeAnnotationRunner", main)
        self.assertIn('task_name == "finetune.cell_type_annotation"', main)


if __name__ == "__main__":
    unittest.main()
