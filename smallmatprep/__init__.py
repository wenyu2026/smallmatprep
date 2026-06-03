"""SmallMatPrep - small-sample materials data diagnostics and imputation toolkit."""
__version__ = "0.1.0"

from .data.loaders import load_csv, load_config
from .inspect.missingness import missing_report
from .impute.baseline import impute_median, impute_knn
from .modeling.recommend import recommend_model
from .evaluate.metrics import evaluate_model
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
