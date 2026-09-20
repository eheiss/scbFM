from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import hydra
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runner import PreTrainRunner
from paths import configure_paths

log = logging.getLogger(__name__)


class PreadaptValEvaluator(PreTrainRunner):
    """Evaluate a checkpoint on the pre-adaptation validation split only.

    This is used to insert the PA0 point: the validation loss on the
    pre-adaptation validation set before any pre-adaptation updates.
    """

    def run(self) -> dict:
        try:
            self._setup_runtime()
            train_indices, val_indices, data_path = self._load_data()
            if val_indices is None:
                raise ValueError("pretrain.validation_split must be > 0 to compute PA0.")
            if not self.pretrain_cfg.resume_checkpoint:
                raise ValueError("pretrain.resume_checkpoint must point to the pretrained checkpoint.")

            self._build_loaders(train_indices, val_indices, data_path)
            self._build_model()

            val_result = self._validate(epoch=0)
            if val_result is None:
                raise RuntimeError("Validation loader was not created.")
            val_metrics, val_epoch_row, val_bin_rows = val_result

            if self.is_master:
                out_dir = self._output_dir()
                out_dir.mkdir(parents=True, exist_ok=True)
                prefix = self._output_prefix()
                epoch_path = out_dir / f"{prefix}_pa0_epoch_metrics.csv"
                bin_path = out_dir / f"{prefix}_pa0_bin_metrics.csv"
                config_path = out_dir / f"{prefix}_pa0_config.yaml"

                self._write_csv(epoch_path, [val_epoch_row])
                self._write_csv(bin_path, val_bin_rows)
                config_path.write_text(
                    OmegaConf.to_yaml(self.cfg, resolve=True),
                    encoding="utf-8",
                )
                log.info(
                    "PA0 validation on %s from %s | loss=%.6f | gene_loss=%.6f | cls_loss=%.6f",
                    data_path,
                    self.pretrain_cfg.resume_checkpoint,
                    val_metrics["val_loss"],
                    val_metrics["val_gene_loss"],
                    val_metrics["val_cls_loss"],
                )
                return {
                    "pa0_epoch_metrics": str(epoch_path),
                    "pa0_bin_metrics": str(bin_path),
                    **val_metrics,
                }
            return {}
        finally:
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_paths(cfg)
    results = PreadaptValEvaluator(cfg).run()
    if int(os.environ.get("RANK", 0)) == 0:
        print("\n=== Final Results ===")
        print(results)


if __name__ == "__main__":
    main()
