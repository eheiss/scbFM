import os

import hydra
from omegaconf import DictConfig

from pretrain import PreTrainRunner


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    runner = PreTrainRunner(cfg)
    results = runner.run()

    if int(os.environ.get("RANK", 0)) == 0:
        print("\n=== Final Results ===")
        print(results)


if __name__ == "__main__":
    main()
