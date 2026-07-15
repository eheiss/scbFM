from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from finetune.surv_pred.runner import SurvPredRunner
from finetune.surv_pred_survboard.raw_mlp_runner import SurvPredSurvBoardRawMLPRunner


class SurvPredRawMLPRunner(SurvPredRunner):
    """Raw-expression MLP Cox baseline for BulkRNABert-style TCGA survival."""

    @staticmethod
    def _resolve_task_cfg(cfg: DictConfig) -> DictConfig:
        if "finetune" in cfg and cfg.finetune is not None and "surv_pred" in cfg.finetune:
            return cfg.finetune.surv_pred
        raise ValueError("Could not find survival prediction config. Expected cfg.finetune.surv_pred.")

    _get_checkpoint_paths = SurvPredSurvBoardRawMLPRunner._get_checkpoint_paths
    _finetune_mode = SurvPredSurvBoardRawMLPRunner._finetune_mode
    _output_suffix = SurvPredSurvBoardRawMLPRunner._output_suffix
    _feature_mean_std = staticmethod(SurvPredSurvBoardRawMLPRunner._feature_mean_std)
    _build_loaders = SurvPredSurvBoardRawMLPRunner._build_loaders
    _build_model = SurvPredSurvBoardRawMLPRunner._build_model
    _build_optimization = SurvPredSurvBoardRawMLPRunner._build_optimization
    _maybe_enable_backbone_optimizer = SurvPredSurvBoardRawMLPRunner._maybe_enable_backbone_optimizer

    def _select_training_hvg_indices(self, train_adata):
        feature_mode = str(getattr(self.task_cfg, "raw_mlp_feature_mode", "all_genes"))
        if feature_mode == "all_genes":
            return np.arange(train_adata.n_vars, dtype=np.int64)
        if feature_mode == "hvg1199":
            return super()._select_training_hvg_indices(train_adata)
        raise ValueError("raw_mlp_feature_mode must be one of: all_genes, hvg1199.")
