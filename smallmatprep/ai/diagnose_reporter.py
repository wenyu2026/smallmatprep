"""Generate a natural-language diagnostic report for a dataset."""
import pandas as pd

from .client import LLMClient
from ..inspect.missingness import missing_report


def _load_prompt(name: str) -> str:
    """Load a prompt template shipped with the package (works in dev & installed)."""
    from importlib.resources import files
    return (files("smallmatprep.ai") / "prompts" / f"{name}.txt").read_text(encoding="utf-8")


def diagnose(
    df: pd.DataFrame,
    target: str | None = None,
    api_key: str = "",
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    """Produce a natural-language diagnosis of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to diagnose.
    target : str, optional
        Name of the target column.
    api_key : str
        LLM API key.
    base_url : str, optional
        Endpoint base URL.
    model : str, optional
        Model name.

    Returns
    -------
    str
        Natural-language diagnostic report.
    """
    # Gather basic stats
    shape_info = f"Shape: {df.shape[0]} rows, {df.shape[1]} columns"
    missing = missing_report(df).to_string()
    dtype_summary = df.dtypes.to_string()

    # Build the user prompt
    parts = [shape_info, f"\nColumn dtypes:\n{dtype_summary}", f"\nMissing report:\n{missing}"]
    if target:
        parts.append(f"\nTarget column: {target}")
    user_prompt = "\n".join(parts)

    client = LLMClient(api_key=api_key, base_url=base_url, model=model or "gpt-4o-mini")
    system_prompt = _load_prompt("diagnose_reporter")
    return str(client.chat(system_prompt, user_prompt, json_mode=False))
