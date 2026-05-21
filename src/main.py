import os

import hydra
from omegaconf import DictConfig

from finetune.canc_type_class import CancTypeClassRunner
from finetune.canc_type_class_33 import CancTypeClass33Runner
from finetune.deconv import DeconvRunner
from finetune.disease_class import DiseaseClassRunner
from finetune.drug_resp import DrugRespRunner
from finetune.gene_essent import GeneEssentRunner
from finetune.surv_pred import SurvPredRunner
from pretrain import PreTrainRunner


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    task_name = str(cfg.task)

    if task_name == "pretrain":
        runner = PreTrainRunner(cfg)
    elif task_name == "finetune.canc_type_class":
        runner = CancTypeClassRunner(cfg)
    elif task_name == "finetune.canc_type_class_33":
        runner = CancTypeClass33Runner(cfg)
    elif task_name == "finetune.deconv":
        runner = DeconvRunner(cfg)
    elif task_name == "finetune.surv_pred":
        runner = SurvPredRunner(cfg)
    elif task_name == "finetune.disease_class":
        runner = DiseaseClassRunner(cfg)
    elif task_name == "finetune.gene_essent":
        runner = GeneEssentRunner(cfg)
    elif task_name == "finetune.drug_resp":
        runner = DrugRespRunner(cfg)
    else:
        raise ValueError(
            f"Unsupported task '{task_name}'. "
            "Expected one of: ['pretrain', 'finetune.canc_type_class', "
            "'finetune.canc_type_class_33', 'finetune.deconv', 'finetune.surv_pred', "
            "'finetune.disease_class', 'finetune.gene_essent', 'finetune.drug_resp']."
        )
    results = runner.run()

    if int(os.environ.get("RANK", 0)) == 0:
        print("\n=== Final Results ===")
        print(results)


if __name__ == "__main__":
    main()
