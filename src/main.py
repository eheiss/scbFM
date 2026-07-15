import os

import hydra
from omegaconf import DictConfig

from finetune.canc_type_class import (
    CancTypeClassBulkFormerPCARFRunner,
    CancTypeClassPCARFRunner,
    CancTypeClassRawMLPRunner,
    CancTypeClassRawPCARFRunner,
    CancTypeClassRunner,
    CancTypeClassScGPTPCARFRunner,
)
from finetune.canc_type_class_33 import (
    CancTypeClass33PCARFRunner,
    CancTypeClass33RawMLPRunner,
    CancTypeClass33Runner,
)
from finetune.deconv import DeconvRawMLPRunner, DeconvRunner
from finetune.disease_class import DiseaseClassPCARFRunner, DiseaseClassRawMLPRunner, DiseaseClassRunner
from finetune.drug_resp import DrugRespRawMLPRunner, DrugRespRunner
from finetune.gene_essent import GeneEssentRawMLPRunner, GeneEssentRunner
from finetune.surv_pred import SurvPredRawMLPRunner, SurvPredRunner
from finetune.surv_pred_binary import SurvPredBinaryRawMLPRunner, SurvPredBinaryRunner
from finetune.surv_pred_survboard import SurvPredSurvBoardRawMLPRunner, SurvPredSurvBoardRunner
from pretrain import PreTrainRunner, ScGPTPreadaptRunner


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    task_name = str(cfg.task)

    if task_name == "pretrain":
        runner = PreTrainRunner(cfg)
    elif task_name == "pretrain.scgpt_preadapt":
        runner = ScGPTPreadaptRunner(cfg)
    elif task_name == "finetune.canc_type_class":
        runner = CancTypeClassRunner(cfg)
    elif task_name == "finetune.canc_type_class_raw_mlp":
        runner = CancTypeClassRawMLPRunner(cfg)
    elif task_name == "finetune.canc_type_class_raw_pca_rf":
        runner = CancTypeClassRawPCARFRunner(cfg)
    elif task_name == "finetune.canc_type_class_pca_rf":
        runner = CancTypeClassPCARFRunner(cfg)
    elif task_name == "finetune.canc_type_class_bulkformer_pca_rf":
        runner = CancTypeClassBulkFormerPCARFRunner(cfg)
    elif task_name == "finetune.canc_type_class_scgpt_pca_rf":
        runner = CancTypeClassScGPTPCARFRunner(cfg)
    elif task_name == "finetune.canc_type_class_33":
        runner = CancTypeClass33Runner(cfg)
    elif task_name == "finetune.canc_type_class_33_raw_mlp":
        runner = CancTypeClass33RawMLPRunner(cfg)
    elif task_name == "finetune.canc_type_class_33_pca_rf":
        runner = CancTypeClass33PCARFRunner(cfg)
    elif task_name == "finetune.deconv":
        runner = DeconvRunner(cfg)
    elif task_name == "finetune.deconv_raw_mlp":
        runner = DeconvRawMLPRunner(cfg)
    elif task_name == "finetune.surv_pred":
        runner = SurvPredRunner(cfg)
    elif task_name == "finetune.surv_pred_raw_mlp":
        runner = SurvPredRawMLPRunner(cfg)
    elif task_name == "finetune.surv_pred_survboard":
        runner = SurvPredSurvBoardRunner(cfg)
    elif task_name == "finetune.surv_pred_survboard_raw_mlp":
        runner = SurvPredSurvBoardRawMLPRunner(cfg)
    elif task_name == "finetune.surv_pred_binary":
        runner = SurvPredBinaryRunner(cfg)
    elif task_name == "finetune.surv_pred_binary_raw_mlp":
        runner = SurvPredBinaryRawMLPRunner(cfg)
    elif task_name == "finetune.disease_class":
        runner = DiseaseClassRunner(cfg)
    elif task_name == "finetune.disease_class_raw_mlp":
        runner = DiseaseClassRawMLPRunner(cfg)
    elif task_name == "finetune.disease_class_pca_rf":
        runner = DiseaseClassPCARFRunner(cfg)
    elif task_name == "finetune.gene_essent":
        runner = GeneEssentRunner(cfg)
    elif task_name == "finetune.gene_essent_raw_mlp":
        runner = GeneEssentRawMLPRunner(cfg)
    elif task_name == "finetune.drug_resp":
        runner = DrugRespRunner(cfg)
    elif task_name == "finetune.drug_resp_raw_mlp":
        runner = DrugRespRawMLPRunner(cfg)
    else:
        raise ValueError(
            f"Unsupported task '{task_name}'. "
            "Expected one of: ['pretrain', 'pretrain.scgpt_preadapt', "
            "'finetune.canc_type_class', "
            "'finetune.canc_type_class_raw_mlp', 'finetune.canc_type_class_raw_pca_rf', "
            "'finetune.canc_type_class_pca_rf', "
            "'finetune.canc_type_class_bulkformer_pca_rf', "
            "'finetune.canc_type_class_scgpt_pca_rf', "
            "'finetune.canc_type_class_33', "
            "'finetune.canc_type_class_33_raw_mlp', 'finetune.canc_type_class_33_pca_rf', "
            "'finetune.deconv', 'finetune.deconv_raw_mlp', 'finetune.surv_pred', "
            "'finetune.surv_pred_raw_mlp', 'finetune.surv_pred_survboard', "
            "'finetune.surv_pred_survboard_raw_mlp', 'finetune.surv_pred_binary', "
            "'finetune.surv_pred_binary_raw_mlp', 'finetune.disease_class', "
            "'finetune.disease_class_raw_mlp', 'finetune.disease_class_pca_rf', "
            "'finetune.gene_essent', 'finetune.gene_essent_raw_mlp', "
            "'finetune.drug_resp', "
            "'finetune.drug_resp_raw_mlp']."
        )
    results = runner.run()

    if int(os.environ.get("RANK", 0)) == 0:
        print("\n=== Final Results ===")
        print(results)


if __name__ == "__main__":
    main()
