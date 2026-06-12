"""SmallMatPrep - small-sample materials data diagnostics and imputation toolkit."""
__version__ = "0.1.0"

from .data.loaders import load_csv, load_config
from .inspect.missingness import missing_report
from .inspect.diagnosis import sample_diagnosis, missing_pattern_report
from .impute.baseline import impute_median, impute_knn
from .impute.iterative import KNNExtraTreesImputer, impute_knn_extratrees
from .impute.constraints import (
    apply_composition_constraint,
    apply_range_constraint,
    apply_similarity_constraint,
    impute_with_uncertainty,
    apply_all_constraints,
    constrained_impute,
    electrolyte_default_constraints,
    alloy_default_constraints,
)
from .features.cep import add_cep_feature, simple_electrolyte_cep
from .modeling.recommend import recommend_model
from .evaluate.metrics import evaluate_model
from .evaluate.decompose import bias_variance_decomposition, decomposition_summary
from .report.summary import build_summary


def __getattr__(name: str):
    if name == "ai":
        try:
            import importlib
            return importlib.import_module("smallmatprep.ai")
        except ImportError as e:
            raise ImportError(
                "AI features require extra dependencies. "
                "Install with: pip install smallmatprep[ai]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
