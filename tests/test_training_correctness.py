import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from finetune.canc_type_class.runner import (  # noqa: E402
    CancTypeClassRunner,
    GroupedCosineAnnealingWarmupRestarts,
    GroupedCosineWarmupUpdateScheduler,
    _set_finetune_training_mode,
)
from finetune.deconv.runner import DeconvRunner  # noqa: E402
from finetune.gene_essent.runner import GeneEssentRunner  # noqa: E402
from finetune.surv_pred_binary.runner import SurvPredBinaryRunner  # noqa: E402
from finetune.surv_pred_survboard.runner import (  # noqa: E402
    cox_partial_log_likelihood,
)
from finetune.training_correctness import (  # noqa: E402
    add_optimizer_parameter_group,
    normalize_accumulated_gradients,
)
from pretrain.runner import PreTrainRunner  # noqa: E402


class GradientAccumulationTest(unittest.TestCase):
    def _classification_runner(
        self,
        model: nn.Module,
        weight: torch.Tensor | None = None,
    ) -> CancTypeClassRunner:
        runner = object.__new__(CancTypeClassRunner)
        runner.loss_fn = nn.CrossEntropyLoss(weight=weight)
        runner.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        runner.device = torch.device("cpu")
        runner.is_distributed = False
        runner.world_size = 1
        return runner

    def _assert_classification_gradient_matches_full_batch(
        self,
        weight: torch.Tensor | None,
    ) -> None:
        torch.manual_seed(7)
        features = torch.randn(5, 4)
        labels = torch.tensor([0, 1, 2, 1, 0])
        full_batch_model = nn.Linear(4, 3)
        accumulated_model = nn.Linear(4, 3)
        accumulated_model.load_state_dict(full_batch_model.state_dict())

        nn.CrossEntropyLoss(weight=weight)(full_batch_model(features), labels).backward()

        runner = self._classification_runner(accumulated_model, weight)
        accumulated_normalizer = 0.0
        for feature, label in zip(features, labels):
            microbatch_labels = label.unsqueeze(0)
            loss = runner.loss_fn(
                accumulated_model(feature.unsqueeze(0)),
                microbatch_labels,
            )
            normalizer = runner._mean_loss_normalizer(microbatch_labels)
            (loss * normalizer).backward()
            accumulated_normalizer += normalizer
        runner._normalize_accumulated_gradients(accumulated_normalizer)

        for expected, actual in zip(
            full_batch_model.parameters(),
            accumulated_model.parameters(),
        ):
            torch.testing.assert_close(expected.grad, actual.grad)

    def test_partial_classification_window_matches_full_batch(self) -> None:
        self._assert_classification_gradient_matches_full_batch(weight=None)

    def test_weighted_classification_window_matches_full_batch(self) -> None:
        self._assert_classification_gradient_matches_full_batch(
            weight=torch.tensor([0.5, 2.0, 1.5])
        )

    def test_masked_pretraining_window_matches_full_batch(self) -> None:
        torch.manual_seed(11)
        features = torch.randn(5, 4)
        labels = torch.tensor(
            [
                [1.0, -100.0, 3.0],
                [-100.0, 2.0, -100.0],
                [0.0, 4.0, 2.0],
                [3.0, -100.0, -100.0],
                [-100.0, 1.0, 4.0],
            ]
        )
        full_gene = nn.Linear(4, 3)
        full_cls = nn.Linear(4, 3)
        accumulated_gene = nn.Linear(4, 3)
        accumulated_cls = nn.Linear(4, 3)
        accumulated_gene.load_state_dict(full_gene.state_dict())
        accumulated_cls.load_state_dict(full_cls.state_dict())

        full_runner = object.__new__(PreTrainRunner)
        full_runner.label_ignore_id = -100
        full_runner.cls_loss_weight = 0.7
        full_loss = full_runner._compute_losses(
            full_gene(features),
            full_cls(features),
            labels,
        )["total"]
        full_loss.backward()

        accumulated_runner = object.__new__(PreTrainRunner)
        accumulated_runner.label_ignore_id = -100
        accumulated_runner.cls_loss_weight = 0.7
        accumulated_runner.optimizer = torch.optim.SGD(
            [*accumulated_gene.parameters(), *accumulated_cls.parameters()],
            lr=0.1,
        )
        accumulated_runner.device = torch.device("cpu")
        accumulated_runner.is_distributed = False
        accumulated_runner.world_size = 1
        accumulated_normalizer = 0.0
        for feature, label in zip(features, labels):
            loss_parts = accumulated_runner._compute_losses(
                accumulated_gene(feature.unsqueeze(0)),
                accumulated_cls(feature.unsqueeze(0)),
                label.unsqueeze(0),
            )
            loss_parts["total_sums"].sum().backward()
            accumulated_normalizer += float(loss_parts["normalizers"].sum().item())
        self.assertTrue(
            accumulated_runner._normalize_accumulated_gradients(
                accumulated_normalizer
            )
        )

        for expected, actual in zip(
            [*full_gene.parameters(), *full_cls.parameters()],
            [*accumulated_gene.parameters(), *accumulated_cls.parameters()],
        ):
            torch.testing.assert_close(expected.grad, actual.grad)

    def test_final_partial_window_is_a_synchronization_boundary(self) -> None:
        boundaries = [
            CancTypeClassRunner._is_accumulation_boundary(step, 19, 16)
            for step in range(1, 20)
        ]
        self.assertEqual(
            [index + 1 for index, is_boundary in enumerate(boundaries) if is_boundary],
            [16, 19],
        )

    def test_regression_accumulation_matches_full_batch(self) -> None:
        torch.manual_seed(23)
        features = torch.randn(5, 3)
        targets = torch.randn(5, 2)
        full_model = nn.Linear(3, 2)
        accumulated_model = nn.Linear(3, 2)
        accumulated_model.load_state_dict(full_model.state_dict())

        nn.functional.mse_loss(full_model(features), targets).backward()
        optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)
        normalizer = 0.0
        for feature, target in zip(features, targets):
            loss_sum = nn.functional.mse_loss(
                accumulated_model(feature.unsqueeze(0)),
                target.unsqueeze(0),
                reduction="sum",
            )
            loss_sum.backward()
            normalizer += target.numel()
        normalize_accumulated_gradients(
            accumulated_model.parameters(),
            normalizer,
            device=torch.device("cpu"),
            is_distributed=False,
            world_size=1,
        )

        for expected, actual in zip(full_model.parameters(), accumulated_model.parameters()):
            torch.testing.assert_close(expected.grad, actual.grad)

    def test_deconvolution_loss_components_reconstruct_mean(self) -> None:
        runner = object.__new__(DeconvRunner)
        runner.task_cfg = SimpleNamespace(loss="mse")
        logits = torch.randn(3, 4)
        targets = torch.softmax(torch.randn(3, 4), dim=-1)
        loss_sum, normalizer = runner._loss_sum_and_normalizer(logits, targets)
        self.assertEqual(normalizer, targets.numel())
        torch.testing.assert_close(
            loss_sum / normalizer,
            runner._compute_loss(logits, targets),
        )

    def test_masked_mse_uses_finite_target_count(self) -> None:
        predictions = torch.tensor([[1.0, 4.0], [3.0, 8.0]])
        targets = torch.tensor([[2.0, float("nan")], [5.0, 9.0]])
        loss_sum, normalizer = GeneEssentRunner._masked_mse_sum_and_normalizer(
            predictions,
            targets,
        )
        self.assertEqual(normalizer, 3.0)
        torch.testing.assert_close(loss_sum, torch.tensor(6.0))

    def test_cox_accumulation_builds_one_effective_risk_set(self) -> None:
        torch.manual_seed(29)
        features = torch.randn(7, 3)
        times = torch.tensor([9.0, 8.0, 7.0, 5.0, 4.0, 2.0, 1.0])
        events = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        full_model = nn.Linear(3, 1)
        window_model = nn.Linear(3, 1)
        window_model.load_state_dict(full_model.state_dict())

        cox_partial_log_likelihood(
            full_model(features).squeeze(-1),
            times,
            events,
        ).backward()
        window_hazards = [
            window_model(features[:3]).squeeze(-1),
            window_model(features[3:]).squeeze(-1),
        ]
        cox_partial_log_likelihood(
            torch.cat(window_hazards),
            times,
            events,
        ).backward()

        for expected, actual in zip(full_model.parameters(), window_model.parameters()):
            torch.testing.assert_close(expected.grad, actual.grad)


class HeadOnlyTrainingModeTest(unittest.TestCase):
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
            self.to_out = nn.Sequential(nn.Linear(2, 1), nn.Dropout(0.5))

    def test_head_only_keeps_frozen_backbone_in_evaluation_mode(self) -> None:
        model = self._Model()

        _set_finetune_training_mode(model, "head_only")

        self.assertTrue(model.training)
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.to_out.training)

    def test_trainable_backbone_modes_keep_complete_model_in_training_mode(self) -> None:
        for finetune_mode in ("adapters", "full_ft"):
            model = self._Model()

            _set_finetune_training_mode(model, finetune_mode)

            self.assertTrue(model.training)
            self.assertTrue(model.backbone.training)
            self.assertTrue(model.to_out.training)


class AtomicCheckpointTest(unittest.TestCase):
    def test_checkpoint_contains_resume_history_and_is_atomically_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            runner = object.__new__(PreTrainRunner)
            runner.is_master = True
            runner.model = nn.Linear(2, 2)
            runner.expr_decoder = nn.Linear(2, 1)
            runner.cls_decoder = nn.Linear(2, 1)
            parameters = [
                *runner.model.parameters(),
                *runner.expr_decoder.parameters(),
                *runner.cls_decoder.parameters(),
            ]
            runner.optimizer = torch.optim.Adam(parameters, lr=1e-3)
            runner.scheduler = torch.optim.lr_scheduler.StepLR(
                runner.optimizer,
                step_size=1,
            )
            runner.gene_num = 10
            runner.selected_gene_count = 4
            runner.gene_sampling = "uniform_full_vocab"
            runner.max_seq_len = 5
            runner.bin_num = 51
            runner.cls_loss_weight = 1.0
            runner.loaded_optimizer_state = False
            runner.sample_limit = 4
            runner.source_sample_count = 10
            runner.effective_sample_count = 4
            runner.train_sample_count = 3
            runner.validation_sample_count = 1
            runner._model_name = lambda: "checkpoint_test"
            runner._output_dir = lambda: Path(temporary_directory)
            history = [{"epoch": 1, "train_loss": 0.5}]
            epoch_rows = [{"epoch": 1, "split": "train", "loss": 0.5}]
            bin_rows = [{"epoch": 1, "split": "train", "token_id": 0}]

            checkpoint_path = runner._save_checkpoint(
                1,
                0.5,
                history,
                epoch_rows,
                bin_rows,
            )

            self.assertIsNotNone(checkpoint_path)
            self.assertTrue(checkpoint_path.exists())
            self.assertFalse(checkpoint_path.with_suffix(".pth.tmp").exists())
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["history"], history)
            self.assertEqual(checkpoint["epoch_metric_rows"], epoch_rows)
            self.assertEqual(checkpoint["bin_metric_rows"], bin_rows)
            self.assertEqual(checkpoint["sample_limit"], 4)
            self.assertEqual(checkpoint["effective_sample_count"], 4)


class SampleLimitTest(unittest.TestCase):
    def test_sample_limits_are_exact_deterministic_and_nested(self) -> None:
        small = PreTrainRunner._select_sample_indices(100, 10, seed=2021)
        medium = PreTrainRunner._select_sample_indices(100, 50, seed=2021)
        full = PreTrainRunner._select_sample_indices(100, None, seed=2021)
        repeated = PreTrainRunner._select_sample_indices(100, 10, seed=2021)

        self.assertEqual(len(small), 10)
        self.assertEqual(len(medium), 50)
        self.assertTrue(torch.equal(torch.from_numpy(small), torch.from_numpy(repeated)))
        self.assertTrue(torch.equal(torch.from_numpy(small), torch.from_numpy(medium[:10])))
        self.assertTrue(torch.equal(torch.from_numpy(medium), torch.from_numpy(full[:50])))

    def test_sample_limit_cannot_exceed_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 9 samples"):
            PreTrainRunner._select_sample_indices(9, 10, seed=2021)

    def test_train_and_validation_partitions_remain_nested(self) -> None:
        selected = PreTrainRunner._select_sample_indices(100, 100, seed=2021)
        small_train, small_validation = PreTrainRunner._split_sample_indices(
            selected[:40],
            0.2,
        )
        large_train, large_validation = PreTrainRunner._split_sample_indices(
            selected[:80],
            0.2,
        )
        self.assertTrue(
            torch.equal(
                torch.from_numpy(small_train),
                torch.from_numpy(large_train[: len(small_train)]),
            )
        )
        self.assertTrue(
            torch.equal(
                torch.from_numpy(small_validation),
                torch.from_numpy(large_validation[: len(small_validation)]),
            )
        )

    def test_validation_mask_generator_is_reproducible(self) -> None:
        runner = object.__new__(PreTrainRunner)
        runner.device = torch.device("cpu")
        runner.pretrain_cfg = SimpleNamespace(
            mask_prob=0.5,
            exclude_masked_from_attention=True,
        )
        runner.pad_gene_id = 1
        runner.label_ignore_id = -100
        runner.mask_ignore_values = set()
        runner.mask_value = -1.0
        batch = {
            "gene_ids": torch.tensor([[0, 2, 3, 4], [0, 5, 6, 7]]),
            "expr": torch.tensor([[-2.0, 1.0, 2.0, 3.0], [-2.0, 4.0, 5.0, 6.0]]),
        }
        first_generator = torch.Generator().manual_seed(991)
        second_generator = torch.Generator().manual_seed(991)
        first = runner._mask_batch(batch, generator=first_generator)
        second = runner._mask_batch(batch, generator=second_generator)
        for first_tensor, second_tensor in zip(first, second):
            torch.testing.assert_close(first_tensor, second_tensor)


class BurnInOptimizerTest(unittest.TestCase):
    def test_adding_backbone_group_preserves_head_adam_state(self) -> None:
        head = nn.Linear(3, 2)
        backbone = nn.Linear(3, 3)
        optimizer = torch.optim.Adam(
            [{"params": head.parameters(), "lr": 1e-3, "name": "head"}]
        )
        scheduler = GroupedCosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=5,
            max_lrs=[1e-3],
            min_lr_ratio=0.1,
            warmup_steps=1,
        )
        head(torch.randn(2, 3)).sum().backward()
        optimizer.step()
        head_parameter = next(head.parameters())
        head_step = optimizer.state[head_parameter]["step"].clone()

        add_optimizer_parameter_group(
            optimizer=optimizer,
            scheduler=scheduler,
            parameters=backbone.parameters(),
            max_lr=1e-4,
            name="backbone",
        )

        torch.testing.assert_close(optimizer.state[head_parameter]["step"], head_step)
        self.assertEqual([group["name"] for group in optimizer.param_groups], ["head", "backbone"])
        self.assertEqual(len(scheduler.base_max_lrs), 2)

    def test_update_scheduler_adds_backbone_at_current_decay_position(self) -> None:
        head = nn.Linear(3, 2)
        backbone = nn.Linear(3, 3)
        optimizer = torch.optim.Adam(
            [{"params": head.parameters(), "lr": 1e-3, "name": "head"}]
        )
        scheduler = GroupedCosineWarmupUpdateScheduler(
            optimizer,
            max_lrs=[1e-3],
            min_lr_ratio=0.1,
            updates_per_epoch=2,
            epochs=5,
            warmup_epochs=1,
        )

        scheduler.step()
        head(torch.randn(2, 3)).sum().backward()
        optimizer.step()
        head_parameter = next(head.parameters())
        head_step = optimizer.state[head_parameter]["step"].clone()

        add_optimizer_parameter_group(
            optimizer=optimizer,
            scheduler=scheduler,
            parameters=backbone.parameters(),
            max_lr=1e-4,
            name="backbone",
        )

        torch.testing.assert_close(optimizer.state[head_parameter]["step"], head_step)
        self.assertEqual([group["name"] for group in optimizer.param_groups], ["head", "backbone"])
        self.assertAlmostEqual(
            optimizer.param_groups[1]["lr"],
            optimizer.param_groups[0]["lr"] / 10.0,
        )
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-4)


class UpdateBasedWarmupSchedulerTest(unittest.TestCase):
    def test_two_epoch_warmup_and_cosine_decay_reach_exact_boundaries(self) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1e-4)
        scheduler = GroupedCosineWarmupUpdateScheduler(
            optimizer,
            max_lrs=[1e-4],
            min_lr_ratio=0.01,
            updates_per_epoch=2,
            epochs=5,
            warmup_epochs=2,
        )

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6)
        observed = []
        for _ in range(10):
            scheduler.step()
            observed.append(optimizer.param_groups[0]["lr"])

        self.assertAlmostEqual(observed[0], 1e-6 + (1e-4 - 1e-6) / 4)
        self.assertAlmostEqual(observed[3], 1e-4)
        self.assertTrue(all(a >= b for a, b in zip(observed[3:], observed[4:])))
        self.assertAlmostEqual(observed[-1], 1e-6)

    def test_scheduler_rejects_updates_beyond_configured_training(self) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1e-4)
        scheduler = GroupedCosineWarmupUpdateScheduler(
            optimizer,
            max_lrs=[1e-4],
            min_lr_ratio=0.01,
            updates_per_epoch=1,
            epochs=2,
            warmup_epochs=1,
        )
        scheduler.step()
        scheduler.step()
        with self.assertRaisesRegex(RuntimeError, "more optimizer updates"):
            scheduler.step()


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_rejects_checkpoint_with_different_sequence_configuration(self) -> None:
        runner = object.__new__(CancTypeClassRunner)
        runner.model_cfg = SimpleNamespace(gene_num=13004, bin_num=51)
        runner.selected_gene_count = 1199
        runner.max_seq_len = 1200
        checkpoint = {
            "backbone": "cancerfoundation",
            "gene_num": 13004,
            "selected_gene_count": 13004,
            "max_seq_len": 13005,
            "epoch": 5,
            "model_state_dict": {},
        }

        with self.assertRaisesRegex(ValueError, "incompatible"):
            runner._validate_backbone_checkpoint(checkpoint, "checkpoint.pth")


class SharedRunnerBridgeTest(unittest.TestCase):
    def test_binary_survival_runner_exposes_shared_training_helpers(self) -> None:
        required_helpers = {
            "_validate_backbone_checkpoint",
            "_mean_loss_normalizer",
            "_per_sample_loss_components",
            "_normalize_accumulated_gradients",
            "_training_epoch_metrics",
        }
        self.assertTrue(
            all(hasattr(SurvPredBinaryRunner, name) for name in required_helpers)
        )
        self.assertTrue(
            SurvPredBinaryRunner._is_accumulation_boundary(19, 19, 16)
        )


if __name__ == "__main__":
    unittest.main()
