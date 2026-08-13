from __future__ import annotations

import logging
import os
from pathlib import Path

import anndata as ad
import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from finetune.canc_type_class.runner import (
    CHECKPOINT_MODEL_KEYS as BASE_CHECKPOINT_MODEL_KEYS,
    RANDOM_INIT_MODEL_KEY as BASE_RANDOM_INIT_MODEL_KEY,
    CancTypeClassRunner,
)
from utils import distributed_concat

log = logging.getLogger(__name__)


class SurvPredBinaryRunner:
    """Binary TCGA survival-status prediction with AUROC output.

    This task intentionally uses the cancer-type-classification TCGA AnnData
    stack, but replaces the target with a binary survival endpoint. The default
    endpoint is the TCGA-style ``OS`` event/status column: 1 = event/deceased,
    0 = censored/alive. This intentionally does not add fixed-time censoring
    logic, because that is not documented in the BulkFormer task definition.
    """

    task_name = "surv_pred_binary"
    config_node = "surv_pred_binary"
    checkpoint_model_keys = BASE_CHECKPOINT_MODEL_KEYS
    random_init_model_key = BASE_RANDOM_INIT_MODEL_KEY

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "surv_pred_binary" in cfg.finetune:
            return cfg.finetune.surv_pred_binary
        raise ValueError(
            "Could not find a binary survival prediction config. "
            "Expected cfg.finetune.surv_pred_binary."
        )

    @staticmethod
    def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
        if "pretrain" in cfg:
            return cfg.pretrain
        raise ValueError("Could not find model architecture config. Expected cfg.pretrain.")

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.task_cfg = self._resolve_task_cfg(cfg)
        self.model_cfg = self._resolve_model_cfg(cfg)

        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_distributed = self.world_size > 1
        self.is_master = self.rank == 0
        self.device = torch.device("cpu")

        self.cls_gene_id = 0
        self.pad_gene_id = 1
        self.gene_token_offset = 2
        self.num_gene_tokens = int(self.model_cfg.gene_num) + self.gene_token_offset
        self.cls_value = float(getattr(self.model_cfg, "pad_value", -2.0))
        self.selected_gene_count = int(getattr(self.model_cfg, "selected_gene_count", 1199))
        self.max_seq_len = int(
            getattr(self.model_cfg, "max_seq_len", self.selected_gene_count + 1)
        )
        if self.max_seq_len != self.selected_gene_count + 1:
            raise ValueError(
                "Binary survival prediction expects max_seq_len to equal "
                "selected_gene_count + 1 for the <cls> token."
            )

        self.label_dict: np.ndarray | None = None
        self.train_loader = None
        self.test_loader = None
        self.test_dataset_size = 0
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.train_class_weights: torch.Tensor | None = None
        self.backbone_optimizer_enabled = False

    def _get_checkpoint_paths(self) -> dict[str, str]:
        paths_cfg = getattr(self.task_cfg, "pretrained_model_paths", None)
        if paths_cfg is None:
            raise ValueError(
                f"finetune.{self.config_node}.pretrained_model_paths must define "
                f"{', '.join(self.checkpoint_model_keys)}."
            )

        all_model_keys = (*self.checkpoint_model_keys, self.random_init_model_key)
        requested = getattr(self.task_cfg, "model_keys", None)
        model_keys = all_model_keys if requested is None else tuple(map(str, requested))
        invalid = sorted(set(model_keys).difference(all_model_keys))
        if invalid:
            raise ValueError(f"Unknown binary survival model_keys: {invalid}.")
        checkpoint_paths: dict[str, str] = {}
        missing = []
        for key in model_keys:
            if key == self.random_init_model_key:
                checkpoint_paths[key] = ""
                continue
            value = paths_cfg.get(key)
            if value:
                checkpoint_paths[key] = str(Path(hydra.utils.to_absolute_path(str(value))))
            else:
                missing.append(key)
        if missing:
            raise ValueError(
                f"Missing checkpoint paths in finetune.{self.config_node}.pretrained_model_paths: "
                f"{missing}"
            )
        return checkpoint_paths

    def _load_tcga(self) -> ad.AnnData:
        configured_path = getattr(self.task_cfg, "tcga_data_dir", None)
        if not configured_path:
            raise ValueError("finetune.surv_pred_binary.tcga_data_dir must be set.")
        data_path = Path(hydra.utils.to_absolute_path(str(configured_path)))
        if not data_path.exists():
            raise FileNotFoundError(f"TCGA h5ad file not found: {data_path}")

        adata = ad.read_h5ad(data_path)
        required_obs = {"sample_id", "patient_id", "project"}
        missing_obs = sorted(required_obs.difference(adata.obs.columns))
        if missing_obs:
            raise ValueError(f"TCGA AnnData is missing required obs columns: {missing_obs}.")

        adata.obs["project"] = adata.obs["project"].astype(str).str.strip().str.upper()
        cohorts = [str(cohort).upper() for cohort in getattr(self.task_cfg, "cohorts", [])]
        if cohorts:
            keep_mask = adata.obs["project"].isin(set(cohorts)).to_numpy()
            adata = adata[keep_mask].copy()

        if adata.n_obs == 0:
            raise ValueError(f"No TCGA samples matched cohorts {cohorts} in obs['project'].")

        adata.obs_names = adata.obs["sample_id"].astype(str)
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        return adata

    def _load_input_adata(self) -> ad.AnnData:
        log.info("Loading TCGA cohorts for binary survival prediction")
        return self._load_tcga()

    @staticmethod
    def _numeric_obs(adata: ad.AnnData, column: str) -> np.ndarray:
        if column not in adata.obs:
            raise ValueError(
                f"TCGA AnnData is missing obs['{column}']. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        return np.asarray(
            adata.obs[column]
            .astype(str)
            .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
            .astype(float),
            dtype=float,
        )

    def _derive_binary_survival_labels(self, adata: ad.AnnData) -> np.ndarray:
        event_col = str(getattr(self.task_cfg, "survival_event_col", "OS"))
        event = self._numeric_obs(adata, event_col)
        valid = np.isfinite(event)

        if valid.sum() < int(getattr(self.task_cfg, "cv_folds", 5)) * 2:
            raise ValueError(
                f"Only {int(valid.sum())} samples have usable binary survival labels."
            )

        before = adata.n_obs
        if not np.all(valid):
            adata._inplace_subset_obs(valid)
            event = event[valid]
        labels = np.where(event > 0, "1", "0").astype(str)

        classes, counts = np.unique(labels, return_counts=True)
        if set(classes.tolist()) != {"0", "1"}:
            raise ValueError(
                "Binary survival endpoint must contain both classes '0' and '1'; "
                f"got counts {dict(zip(classes.tolist(), counts.tolist()))}."
            )

        if self.is_master:
            log.info(
                "Derived binary survival labels from event_col=%s: "
                "%d -> %d samples | class counts=%s",
                event_col,
                before,
                adata.n_obs,
                dict(zip(classes.tolist(), counts.tolist())),
            )
        return labels

    def _prepare_cv_data(self) -> tuple[ad.AnnData, np.ndarray, np.ndarray | None]:
        adata = self._load_input_adata()
        labels = self._derive_binary_survival_labels(adata)
        adata.obs["survival_binary_label"] = labels
        adata.obs["cancer_type"] = labels

        adata = self._preprocess_adata(adata)
        labels = np.asarray(adata.obs["survival_binary_label"]).astype(str)
        self.label_dict = np.asarray(["0", "1"])

        groups = None
        patient_ids = adata.obs["patient_id"].astype(str).to_numpy()
        if len(np.unique(patient_ids)) < len(patient_ids):
            groups = patient_ids
            log.info("Duplicate TCGA patient_id values detected; using patient-grouped CV.")

        if self.is_master:
            log.info(
                "Prepared binary survival prediction data: samples=%d, genes=%d, labels=%s",
                adata.n_obs,
                adata.n_vars,
                dict(zip(*np.unique(labels, return_counts=True))),
            )
        return adata, labels, groups

    def _evaluate(self) -> dict:
        self.model.eval()
        loss_numerators = []
        loss_normalizers = []
        predictions = []
        truths = []
        probabilities = []

        positive_idx = int(np.where(self.label_dict == "1")[0][0])

        if self.is_distributed:
            dist.barrier()

        with torch.no_grad():
            for data, labels in self.test_loader:
                data = self._move_batch_to_device(data)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(data)
                batch_loss_numerators, batch_loss_normalizers = (
                    self._per_sample_loss_components(logits, labels)
                )
                loss_numerators.append(batch_loss_numerators)
                loss_normalizers.append(batch_loss_normalizers)
                probs = torch.softmax(logits, dim=-1)[:, positive_idx]
                predictions.append(logits.argmax(dim=-1))
                probabilities.append(probs)
                truths.append(labels)

        loss_numerators = torch.cat(loss_numerators, dim=0)
        loss_normalizers = torch.cat(loss_normalizers, dim=0)
        predictions = torch.cat(predictions, dim=0)
        probabilities = torch.cat(probabilities, dim=0)
        truths = torch.cat(truths, dim=0)

        if self.is_distributed:
            loss_numerators = distributed_concat(
                loss_numerators,
                self.test_dataset_size,
                self.world_size,
            )
            loss_normalizers = distributed_concat(
                loss_normalizers,
                self.test_dataset_size,
                self.world_size,
            )
            predictions = distributed_concat(predictions, self.test_dataset_size, self.world_size)
            probabilities = distributed_concat(probabilities, self.test_dataset_size, self.world_size)
            truths = distributed_concat(truths, self.test_dataset_size, self.world_size)

        predictions_np = predictions.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        truths_np = truths.cpu().numpy()
        precision, recall, fscore, support = precision_recall_fscore_support(
            truths_np,
            predictions_np,
            labels=np.arange(len(self.label_dict)),
            zero_division=0,
        )

        test_loss = float(
            (loss_numerators.sum() / loss_normalizers.sum()).detach().cpu().item()
        )

        try:
            auroc = float(roc_auc_score(truths_np == positive_idx, probabilities_np))
        except ValueError:
            auroc = float("nan")
        try:
            auprc = float(average_precision_score(truths_np == positive_idx, probabilities_np))
        except ValueError:
            auprc = float("nan")

        return {
            "loss": test_loss,
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": float(accuracy_score(truths_np, predictions_np)),
            "f1_macro": float(
                f1_score(
                    truths_np,
                    predictions_np,
                    labels=np.arange(len(self.label_dict)),
                    average="macro",
                    zero_division=0,
                )
            ),
            "f1_weighted": float(
                f1_score(
                    truths_np,
                    predictions_np,
                    labels=np.arange(len(self.label_dict)),
                    average="weighted",
                    zero_division=0,
                )
            ),
            "confusion_matrix": confusion_matrix(
                truths_np,
                predictions_np,
                labels=np.arange(len(self.label_dict)),
            ),
            "classification_report": classification_report(
                truths_np,
                predictions_np,
                labels=np.arange(len(self.label_dict)),
                target_names=["alive_or_censored", "event"],
                digits=4,
                zero_division=0,
            ),
            "precision_per_class": precision,
            "recall_per_class": recall,
            "f1_per_class": fscore,
            "support_per_class": support,
            "label_dict": self.label_dict.tolist(),
            "n_test_samples": int(len(truths_np)),
            "truth_indices": truths_np,
            "prediction_indices": predictions_np,
            "positive_probabilities": probabilities_np,
        }

    def _prediction_rows(
        self,
        model_key: str,
        fold: int,
        checkpoint_path: str,
        test_adata: ad.AnnData,
        test_metrics: dict[str, object],
    ) -> list[dict[str, object]]:
        truth_indices = np.asarray(test_metrics["truth_indices"], dtype=int)
        prediction_indices = np.asarray(test_metrics["prediction_indices"], dtype=int)
        probabilities = np.asarray(test_metrics["positive_probabilities"], dtype=float)
        labels = self.label_dict.tolist()
        patient_ids = test_adata.obs["patient_id"].astype(str).to_numpy()
        projects = test_adata.obs["project"].astype(str).to_numpy()
        sample_ids = test_adata.obs["sample_id"].astype(str).to_numpy()

        event_col = str(getattr(self.task_cfg, "survival_event_col", "OS"))
        time_col = str(getattr(self.task_cfg, "survival_time_col", "OS.time"))
        events = (
            test_adata.obs[event_col].astype(str).to_numpy()
            if event_col in test_adata.obs
            else np.asarray([""] * test_adata.n_obs)
        )
        times = (
            test_adata.obs[time_col].astype(str).to_numpy()
            if time_col in test_adata.obs
            else np.asarray([""] * test_adata.n_obs)
        )

        rows: list[dict[str, object]] = []
        for idx, (truth_idx, pred_idx, prob) in enumerate(
            zip(truth_indices, prediction_indices, probabilities)
        ):
            rows.append(
                {
                    "model": model_key,
                    "fold": fold,
                    "finetune_mode": self._finetune_mode(),
                    "checkpoint_path": checkpoint_path,
                    "sample_id": sample_ids[idx],
                    "patient_id": patient_ids[idx],
                    "project": projects[idx],
                    "survival_event": events[idx],
                    "survival_time": times[idx],
                    "true_idx": int(truth_idx),
                    "pred_idx": int(pred_idx),
                    "true_label": labels[int(truth_idx)],
                    "pred_label": labels[int(pred_idx)],
                    "prob_event": float(prob),
                    "correct": int(truth_idx == pred_idx),
                }
            )
        return rows


# Keep this runner task-specific while reusing the generic DDP/HVG/optimizer
# mechanics already implemented for CancerFoundation classification. The class
# deliberately does not inherit from CancTypeClassRunner.
_GENERIC_METHODS_FROM_CANCER_CLASS_RUNNER = (
    "_setup_runtime",
    "_write_csv",
    "_write_json",
    "_get_git_commit",
    "_save_run_metadata",
    "_aggregate_numeric_rows",
    "_resolve_gene_list_path",
    "_should_preprocess_input",
    "_preprocess_adata",
    "_select_training_hvg_indices",
    "_build_cv_splits",
    "_cv_manifest_source_rows",
    "_resolve_cv_fold_manifest_path",
    "_fold_assignments_from_splits",
    "_load_cv_fold_manifest",
    "_build_or_load_cv_splits",
    "_build_loaders",
    "_strip_module_prefix",
    "_validate_backbone_checkpoint",
    "_build_model",
    "_build_optimization",
    "_maybe_enable_backbone_optimizer",
    "_optimizer_parameters",
    "_is_accumulation_boundary",
    "_mean_loss_normalizer",
    "_per_sample_loss_components",
    "_normalize_accumulated_gradients",
    "_training_epoch_metrics",
    "_move_batch_to_device",
    "_train_one_epoch",
    "_flatten_fold_metrics",
    "_write_confusion_matrix",
    "_task_output_dir",
    "_output_prefix",
    "_finetune_mode",
    "_output_suffix",
    "_output_variant",
    "_write_model_results",
    "_cleanup_fold_state",
    "run",
)
_STATIC_METHODS_FROM_CANCER_CLASS_RUNNER = {
    "_write_csv",
    "_write_json",
    "_get_git_commit",
    "_aggregate_numeric_rows",
    "_strip_module_prefix",
    "_is_accumulation_boundary",
    "_fold_assignments_from_splits",
}
for _method_name in _GENERIC_METHODS_FROM_CANCER_CLASS_RUNNER:
    _method = getattr(CancTypeClassRunner, _method_name)
    if _method_name in _STATIC_METHODS_FROM_CANCER_CLASS_RUNNER:
        _method = staticmethod(_method)
    setattr(
        SurvPredBinaryRunner,
        _method_name,
        _method,
    )
del _method_name, _method
