import os

import hydra
from omegaconf import DictConfig

from finetune.batch_integration import BatchIntegrationRunner
from finetune.cell_type_annotation import CellTypeAnnotationRunner
from finetune.canc_type_class import (
    CancTypeClassBulkFormerPCARFRunner,
    CancTypeClassPCARFRunner,
    CancTypeClassRawMLPRunner,
    CancTypeClassRawPCARFRunner,
    CancTypeClassRunner,
    CancTypeClassScGPTPCARFRunner,
)
from finetune.canc_type_class_33 import (
    CancTypeClass33BulkFormerPCARFRunner,
    CancTypeClass33PCARFRunner,
    CancTypeClass33RawMLPRunner,
    CancTypeClass33RawPCARFRunner,
    CancTypeClass33Runner,
    CancTypeClass33ScGPTPCARFRunner,
)
from finetune.deconv import (
    DeconvBulkFormerPCARFRunner,
    DeconvPCARFRunner,
    DeconvRawMLPRunner,
    DeconvRawPCARFRunner,
    DeconvRunner,
    DeconvScGPTPCARFRunner,
)
from finetune.disease_class import (
    DiseaseClassBulkFormerPCARFRunner,
    DiseaseClassPCARFRunner,
    DiseaseClassRawMLPRunner,
    DiseaseClassRawPCARFRunner,
    DiseaseClassRunner,
    DiseaseClassScGPTPCARFRunner,
)
from finetune.drug_resp import (
    DrugRespBulkFormerPCARFRunner,
    DrugRespPCARFRunner,
    DrugRespRawMLPRunner,
    DrugRespRawPCARFRunner,
    DrugRespRunner,
    DrugRespScGPTPCARFRunner,
)
from finetune.gene_essent import (
    GeneEssentBulkFormerPCARFRunner,
    GeneEssentPCARFRunner,
    GeneEssentRawMLPRunner,
    GeneEssentRawPCARFRunner,
    GeneEssentRunner,
    GeneEssentScGPTPCARFRunner,
)
from finetune.surv_pred import (
    SurvPredBulkFormerPCARFRunner,
    SurvPredPCARFRunner,
    SurvPredRawMLPRunner,
    SurvPredRawPCARFRunner,
    SurvPredRunner,
    SurvPredScGPTPCARFRunner,
)
from finetune.surv_pred_binary import (
    SurvPredBinaryBulkFormerPCARFRunner,
    SurvPredBinaryPCARFRunner,
    SurvPredBinaryRawMLPRunner,
    SurvPredBinaryRawPCARFRunner,
    SurvPredBinaryRunner,
    SurvPredBinaryScGPTPCARFRunner,
)
from finetune.surv_pred_survboard import (
    SurvPredSurvBoardBulkFormerPCARFRunner,
    SurvPredSurvBoardPCARFRunner,
    SurvPredSurvBoardRawMLPRunner,
    SurvPredSurvBoardRawPCARFRunner,
    SurvPredSurvBoardRunner,
    SurvPredSurvBoardScGPTPCARFRunner,
)
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
    elif task_name == "finetune.canc_type_class_33_raw_pca_rf":
        runner = CancTypeClass33RawPCARFRunner(cfg)
    elif task_name == "finetune.canc_type_class_33_pca_rf":
        runner = CancTypeClass33PCARFRunner(cfg)
    elif task_name == "finetune.canc_type_class_33_bulkformer_pca_rf":
        runner = CancTypeClass33BulkFormerPCARFRunner(cfg)
    elif task_name == "finetune.canc_type_class_33_scgpt_pca_rf":
        runner = CancTypeClass33ScGPTPCARFRunner(cfg)
    elif task_name == "finetune.deconv":
        runner = DeconvRunner(cfg)
    elif task_name == "finetune.deconv_raw_mlp":
        runner = DeconvRawMLPRunner(cfg)
    elif task_name == "finetune.deconv_raw_pca_rf":
        runner = DeconvRawPCARFRunner(cfg)
    elif task_name == "finetune.deconv_pca_rf":
        runner = DeconvPCARFRunner(cfg)
    elif task_name == "finetune.deconv_bulkformer_pca_rf":
        runner = DeconvBulkFormerPCARFRunner(cfg)
    elif task_name == "finetune.deconv_scgpt_pca_rf":
        runner = DeconvScGPTPCARFRunner(cfg)
    elif task_name == "finetune.surv_pred":
        runner = SurvPredRunner(cfg)
    elif task_name == "finetune.surv_pred_raw_mlp":
        runner = SurvPredRawMLPRunner(cfg)
    elif task_name == "finetune.surv_pred_raw_pca_rf":
        runner = SurvPredRawPCARFRunner(cfg)
    elif task_name == "finetune.surv_pred_pca_rf":
        runner = SurvPredPCARFRunner(cfg)
    elif task_name == "finetune.surv_pred_scgpt_pca_rf":
        runner = SurvPredScGPTPCARFRunner(cfg)
    elif task_name == "finetune.surv_pred_bulkformer_pca_rf":
        runner = SurvPredBulkFormerPCARFRunner(cfg)
    elif task_name == "finetune.surv_pred_survboard":
        runner = SurvPredSurvBoardRunner(cfg)
    elif task_name == "finetune.surv_pred_survboard_raw_mlp":
        runner = SurvPredSurvBoardRawMLPRunner(cfg)
    elif task_name == "finetune.surv_pred_survboard_raw_pca_rf":
        runner = SurvPredSurvBoardRawPCARFRunner(cfg)
    elif task_name == "finetune.surv_pred_survboard_pca_rf":
        runner = SurvPredSurvBoardPCARFRunner(cfg)
    elif task_name == "finetune.surv_pred_survboard_scgpt_pca_rf":
        runner = SurvPredSurvBoardScGPTPCARFRunner(cfg)
    elif task_name == "finetune.surv_pred_survboard_bulkformer_pca_rf":
        runner = SurvPredSurvBoardBulkFormerPCARFRunner(cfg)
    elif task_name == "finetune.surv_pred_binary":
        runner = SurvPredBinaryRunner(cfg)
    elif task_name == "finetune.surv_pred_binary_raw_mlp":
        runner = SurvPredBinaryRawMLPRunner(cfg)
    elif task_name == "finetune.surv_pred_binary_raw_pca_rf":
        runner = SurvPredBinaryRawPCARFRunner(cfg)
    elif task_name == "finetune.surv_pred_binary_pca_rf":
        runner = SurvPredBinaryPCARFRunner(cfg)
    elif task_name == "finetune.surv_pred_binary_scgpt_pca_rf":
        runner = SurvPredBinaryScGPTPCARFRunner(cfg)
    elif task_name == "finetune.surv_pred_binary_bulkformer_pca_rf":
        runner = SurvPredBinaryBulkFormerPCARFRunner(cfg)
    elif task_name == "finetune.disease_class":
        runner = DiseaseClassRunner(cfg)
    elif task_name == "finetune.disease_class_raw_mlp":
        runner = DiseaseClassRawMLPRunner(cfg)
    elif task_name == "finetune.disease_class_raw_pca_rf":
        runner = DiseaseClassRawPCARFRunner(cfg)
    elif task_name == "finetune.disease_class_pca_rf":
        runner = DiseaseClassPCARFRunner(cfg)
    elif task_name == "finetune.disease_class_bulkformer_pca_rf":
        runner = DiseaseClassBulkFormerPCARFRunner(cfg)
    elif task_name == "finetune.disease_class_scgpt_pca_rf":
        runner = DiseaseClassScGPTPCARFRunner(cfg)
    elif task_name == "finetune.gene_essent":
        runner = GeneEssentRunner(cfg)
    elif task_name == "finetune.gene_essent_raw_mlp":
        runner = GeneEssentRawMLPRunner(cfg)
    elif task_name == "finetune.gene_essent_raw_pca_rf":
        runner = GeneEssentRawPCARFRunner(cfg)
    elif task_name == "finetune.gene_essent_pca_rf":
        runner = GeneEssentPCARFRunner(cfg)
    elif task_name == "finetune.gene_essent_bulkformer_pca_rf":
        runner = GeneEssentBulkFormerPCARFRunner(cfg)
    elif task_name == "finetune.gene_essent_scgpt_pca_rf":
        runner = GeneEssentScGPTPCARFRunner(cfg)
    elif task_name == "finetune.drug_resp":
        runner = DrugRespRunner(cfg)
    elif task_name == "finetune.drug_resp_raw_mlp":
        runner = DrugRespRawMLPRunner(cfg)
    elif task_name == "finetune.drug_resp_raw_pca_rf":
        runner = DrugRespRawPCARFRunner(cfg)
    elif task_name == "finetune.drug_resp_pca_rf":
        runner = DrugRespPCARFRunner(cfg)
    elif task_name == "finetune.drug_resp_bulkformer_pca_rf":
        runner = DrugRespBulkFormerPCARFRunner(cfg)
    elif task_name == "finetune.drug_resp_scgpt_pca_rf":
        runner = DrugRespScGPTPCARFRunner(cfg)
    elif task_name == "finetune.batch_integration":
        runner = BatchIntegrationRunner(cfg)
    elif task_name == "finetune.cell_type_annotation":
        runner = CellTypeAnnotationRunner(cfg)
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
            "'finetune.canc_type_class_33_raw_mlp', "
            "'finetune.canc_type_class_33_raw_pca_rf', "
            "'finetune.canc_type_class_33_pca_rf', "
            "'finetune.canc_type_class_33_bulkformer_pca_rf', "
            "'finetune.canc_type_class_33_scgpt_pca_rf', "
            "'finetune.deconv', 'finetune.deconv_raw_mlp', "
            "'finetune.deconv_raw_pca_rf', 'finetune.deconv_pca_rf', "
            "'finetune.deconv_bulkformer_pca_rf', "
            "'finetune.deconv_scgpt_pca_rf', 'finetune.surv_pred', "
            "'finetune.surv_pred_raw_mlp', 'finetune.surv_pred_raw_pca_rf', "
            "'finetune.surv_pred_pca_rf', 'finetune.surv_pred_bulkformer_pca_rf', "
            "'finetune.surv_pred_scgpt_pca_rf', 'finetune.surv_pred_survboard', "
            "'finetune.surv_pred_survboard_raw_mlp', "
            "'finetune.surv_pred_survboard_raw_pca_rf', "
            "'finetune.surv_pred_survboard_pca_rf', "
            "'finetune.surv_pred_survboard_bulkformer_pca_rf', "
            "'finetune.surv_pred_survboard_scgpt_pca_rf', "
            "'finetune.surv_pred_binary', 'finetune.surv_pred_binary_raw_mlp', "
            "'finetune.surv_pred_binary_raw_pca_rf', "
            "'finetune.surv_pred_binary_pca_rf', "
            "'finetune.surv_pred_binary_bulkformer_pca_rf', "
            "'finetune.surv_pred_binary_scgpt_pca_rf', 'finetune.disease_class', "
            "'finetune.disease_class_raw_mlp', 'finetune.disease_class_pca_rf', "
            "'finetune.disease_class_raw_pca_rf', "
            "'finetune.disease_class_bulkformer_pca_rf', "
            "'finetune.disease_class_scgpt_pca_rf', "
            "'finetune.gene_essent', 'finetune.gene_essent_raw_mlp', "
            "'finetune.gene_essent_raw_pca_rf', 'finetune.gene_essent_pca_rf', "
            "'finetune.gene_essent_bulkformer_pca_rf', "
            "'finetune.gene_essent_scgpt_pca_rf', "
            "'finetune.drug_resp', "
            "'finetune.drug_resp_raw_mlp', 'finetune.drug_resp_raw_pca_rf', "
            "'finetune.drug_resp_pca_rf', "
            "'finetune.drug_resp_bulkformer_pca_rf', "
            "'finetune.drug_resp_scgpt_pca_rf', "
            "'finetune.batch_integration', "
            "'finetune.cell_type_annotation']."
        )
    results = runner.run()

    if int(os.environ.get("RANK", 0)) == 0:
        print("\n=== Final Results ===")
        print(results)


if __name__ == "__main__":
    main()
