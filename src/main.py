import os

import hydra
from omegaconf import DictConfig

from finetune.canc_type_class import CancTypeClassRunner
from pretrain import PreTrainRunner


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    task_name = str(cfg.task)

    if task_name == "pretrain":
        runner = PreTrainRunner(cfg)
    elif task_name == "finetune.canc_type_class":
        runner = CancTypeClassRunner(cfg)
    else:
        raise ValueError(
            f"Unsupported task '{task_name}'. "
            "Expected one of: ['pretrain', 'finetune.canc_type_class']."
        )
    results = runner.run()

    if int(os.environ.get("RANK", 0)) == 0:
        print("\n=== Final Results ===")
        print(results)


if __name__ == "__main__":
    main()
