import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from pretrain.scgpt_preadapt_runner import (  # noqa: E402
    ScGPTBulkDataset,
    ScGPTPreadaptRunner,
    SeededScGPTCollator,
    _random_train_validation_split,
)


class _RandomNativeCollator:
    def __call__(self, examples):
        values = torch.stack([example["expressions"] for example in examples])
        order = torch.randperm(values.shape[1])
        mask = torch.bernoulli(torch.full_like(values, 0.4)).bool()
        return {
            "gene": torch.stack([example["genes"] for example in examples])[:, order],
            "expr": values[:, order],
            "masked_expr": values[:, order].masked_fill(mask[:, order], -1),
        }


class SampleSplitTest(unittest.TestCase):
    def test_split_is_sample_level_deterministic_and_exact(self) -> None:
        first = _random_train_validation_split(101, 0.1, seed=2021)
        repeated = _random_train_validation_split(101, 0.1, seed=2021)
        changed = _random_train_validation_split(101, 0.1, seed=2022)

        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(int((first == "validation").sum()), 10)
        self.assertEqual(int((first == "train").sum()), 91)
        self.assertFalse(np.array_equal(first, changed))

    def test_split_rejects_invalid_fraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly between"):
            _random_train_validation_split(10, 0.0, seed=1)


class SeededCollatorTest(unittest.TestCase):
    @staticmethod
    def _examples():
        return [
            {
                "genes": torch.arange(8),
                "expressions": torch.arange(8, dtype=torch.float32),
                "record_index": torch.tensor(index),
                "source_id": torch.tensor(index % 2),
            }
            for index in (3, 9)
        ]

    def test_validation_collation_is_fixed(self) -> None:
        collator = SeededScGPTCollator(_RandomNativeCollator(), seed=17, fixed=True)
        collator.set_epoch(1)
        first = collator(self._examples())
        collator.set_epoch(99)
        repeated = collator(self._examples())

        for key in ("gene", "expr", "masked_expr", "record_index", "source_id"):
            torch.testing.assert_close(first[key], repeated[key])

    def test_training_collation_changes_between_epochs(self) -> None:
        collator = SeededScGPTCollator(_RandomNativeCollator(), seed=17, fixed=False)
        collator.set_epoch(1)
        first = collator(self._examples())
        collator.set_epoch(2)
        second = collator(self._examples())

        self.assertFalse(torch.equal(first["masked_expr"], second["masked_expr"]))


class ScGPTBulkDatasetTest(unittest.TestCase):
    @staticmethod
    def _dataset(values: np.ndarray) -> ScGPTBulkDataset:
        dataset = ScGPTBulkDataset(
            data_paths=[Path("unused.h5ad")],
            file_indices=np.asarray([0]),
            row_indices=np.asarray([0]),
            record_indices=np.asarray([17]),
            source_ids=np.asarray([3]),
            source_gene_indices=[np.arange(values.size)],
            vocab_gene_ids=[np.arange(100, 100 + values.size)],
            cls_token_id=7,
            cls_value=-2.0,
        )
        dataset._adatas[0] = SimpleNamespace(X=values.reshape(1, -1))
        return dataset

    def test_only_nonzero_genes_are_tokenized(self) -> None:
        example = self._dataset(np.asarray([0.0, 2.0, 0.0, 4.0, 0.0]))[0]

        torch.testing.assert_close(example["genes"], torch.tensor([7, 101, 103]))
        torch.testing.assert_close(
            example["expressions"], torch.tensor([-2.0, 2.0, 4.0])
        )

    def test_profile_without_expressed_mapped_genes_is_rejected(self) -> None:
        dataset = self._dataset(np.zeros(4, dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "no non-zero vocabulary-matched genes"):
            dataset[0]


class GradientNormalizationTest(unittest.TestCase):
    def test_scaled_accumulated_sums_match_full_masked_mean(self) -> None:
        torch.manual_seed(5)
        features = torch.randn(7, 3)
        targets = torch.randn(7, 2)
        mask = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        full_model = nn.Linear(3, 2)
        accumulated_model = nn.Linear(3, 2)
        accumulated_model.load_state_dict(full_model.state_dict())

        full_error = ((full_model(features) - targets) * mask).square()
        (full_error.sum() / mask.sum()).backward()

        runner = object.__new__(ScGPTPreadaptRunner)
        runner.optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)
        runner.device = torch.device("cpu")
        runner.is_distributed = False
        runner.world_size = 1
        fixed_scale = 16.0
        accumulated_count = 0.0
        for start, end in ((0, 3), (3, 6), (6, 7)):
            error = (
                (accumulated_model(features[start:end]) - targets[start:end])
                * mask[start:end]
            ).square()
            (error.sum() / fixed_scale).backward()
            accumulated_count += float(mask[start:end].sum()) / fixed_scale
        self.assertTrue(runner._normalize_accumulated_gradients(accumulated_count))

        for expected, actual in zip(full_model.parameters(), accumulated_model.parameters()):
            torch.testing.assert_close(expected.grad, actual.grad)


class CheckpointNormalizationTest(unittest.TestCase):
    def test_module_prefix_is_removed(self) -> None:
        checkpoint = {
            "model_state_dict": {
                "module.encoder.weight": torch.ones(2, 3),
                "metadata": "ignored",
            }
        }
        normalized = ScGPTPreadaptRunner._checkpoint_state_dict(checkpoint)
        self.assertEqual(set(normalized), {"encoder.weight"})


if __name__ == "__main__":
    unittest.main()
