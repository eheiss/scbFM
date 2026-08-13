from importlib import import_module

__all__ = [
    "SurvPredSurvBoardRunner",
    "SurvPredSurvBoardRawMLPRunner",
    "SurvPredSurvBoardPCARFRunner",
    "SurvPredSurvBoardRawPCARFRunner",
    "SurvPredSurvBoardScGPTPCARFRunner",
    "SurvPredSurvBoardBulkFormerPCARFRunner",
]

_RUNNER_MODULES = {
    "SurvPredSurvBoardRunner": ".runner",
    "SurvPredSurvBoardRawMLPRunner": ".raw_mlp_runner",
    "SurvPredSurvBoardPCARFRunner": ".pca_rf_runner",
    "SurvPredSurvBoardRawPCARFRunner": ".raw_pca_rf_runner",
    "SurvPredSurvBoardScGPTPCARFRunner": ".scgpt_pca_rf_runner",
    "SurvPredSurvBoardBulkFormerPCARFRunner": ".bulkformer_pca_rf_runner",
}


def __getattr__(name: str):
    """Load runners on demand so survival packages can import each other safely."""
    module_name = _RUNNER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
