"""Generate SmallMatPrep config from natural language description."""
import json
from pathlib import Path

import pandas as pd

from .client import LLMClient
from ..config_schema import validate_config


def _load_prompt(name: str) -> str:
    """Load a prompt template shipped with the package (works in dev & installed)."""
    from importlib.resources import files
    return (files("smallmatprep.ai") / "prompts" / f"{name}.txt").read_text(encoding="utf-8")


def generate_config(
    description: str,
    api_key: str,
    base_url: str | None = None,
    model: str | None = None,
) -> dict:
    """Turn a natural language description into a SmallMatPrep config dict.

    Parameters
    ----------
    description : str
        User's natural language description of the data and task.
    api_key : str
        LLM API key.
    base_url : str, optional
        Endpoint base URL (e.g. Kimi, DeepSeek). Defaults to OpenAI.
    model : str, optional
        Model name. Defaults to the client's default.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    client = LLMClient(api_key=api_key, base_url=base_url, model=model or "gpt-4o-mini")
    system_prompt = _load_prompt("config_generator")
    user_prompt = f"请根据以下描述生成 SmallMatPrep 配置文件：\n{description.strip()}"
    result = client.chat(system_prompt, user_prompt, json_mode=True)
    if isinstance(result, dict):
        # Clean up internal marker and validate
        result.pop("_inferred", None)
        _validate_or_warn(result, columns=None)
        return result
    # Fallback: try to parse raw string as JSON
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse AI response as JSON. Raw response:\n{result[:500]}"
        ) from e
    parsed.pop("_inferred", None)
    _validate_or_warn(parsed, columns=None)
    return parsed


def _validate_or_warn(config: dict, columns: list[str] | None) -> None:
    """Validate generated config and warn about issues (don't reject)."""
    issues = validate_config(config, columns=columns)
    if issues:
        import warnings
        warnings.warn(
            "AI-generated config has issues:\n  " + "\n  ".join(issues),
            stacklevel=3,
        )


def generate_config_with_columns(
    description: str,
    api_key: str,
    columns: list[str] | pd.DataFrame | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> dict:
    """Generate config with actual column names as reference.

    This is the **recommended** function for most use cases. By providing the
    actual column names (or the DataFrame itself), the AI can produce a config
    that references real columns instead of guessing.

    Parameters
    ----------
    description : str
        Natural language description of your data and what you want to predict.
        Example: "我想预测电导率，温度列缺失较多，溶剂列中有两种溶剂含量"

    api_key : str
        LLM API key (OpenAI / DeepSeek / Kimi / etc.).

    columns : list[str] or pd.DataFrame, optional
        - If a list of strings: treated as column names the AI can reference.
        - If a DataFrame: its .columns.tolist() will be used.
        - If None: the AI will infer column names from the description only
          (less reliable).

    base_url : str, optional
        Endpoint base URL. Defaults to OpenAI.
        DeepSeek: "https://api.deepseek.com/v1"
        Kimi:     "https://api.moonshot.cn/v1"

    model : str, optional
        Model name. Defaults to "gpt-4o-mini".
        DeepSeek: "deepseek-chat"
        Kimi:     "moonshot-v1-8k"

    Returns
    -------
    dict
        Configuration dictionary with fields: target, id_column, feature_groups,
        blank_as_zero_columns, categorical_columns, drop_columns, impute, model.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.read_csv("my_data.csv")
    >>> config = generate_config_with_columns(
    ...     "预测电导率，温度列缺失多，溶剂有A和B两种",
    ...     api_key="sk-xxx",
    ...     columns=df,               # 或 columns=df.columns.tolist()
    ...     base_url="https://api.deepseek.com/v1",
    ...     model="deepseek-chat",
    ... )
    """
    # Normalize columns argument to a list of strings
    if columns is None:
        col_list: list[str] = []
    elif isinstance(columns, pd.DataFrame):
        col_list = columns.columns.tolist()
    else:
        col_list = list(columns)

    # Build an enriched user prompt that injects the real column names
    parts = [
        f"请根据以下描述生成 SmallMatPrep 配置文件。",
        f"",
        f"用户的自然语言描述：",
        f"{description.strip()}",
    ]
    if col_list:
        parts.append(f"")
        parts.append(f"=== 数据集中真实存在的列名（请只使用这些列名，不要编造）===")
        parts.append(f"列名列表: {json.dumps(col_list, ensure_ascii=False)}")
        parts.append(f"共 {len(col_list)} 列。请从这些列中选择 target、id_column，")
        parts.append(f"将相关列归入 feature_groups，并判断哪些列需要放入")
        parts.append(f"blank_as_zero_columns / categorical_columns / drop_columns。")

    user_prompt = "\n".join(parts)

    client = LLMClient(api_key=api_key, base_url=base_url, model=model or "gpt-4o-mini")
    system_prompt = _load_prompt("config_generator")
    result = client.chat(system_prompt, user_prompt, json_mode=True)

    if isinstance(result, dict):
        result.pop("_inferred", None)
        _validate_or_warn(result, columns=col_list if col_list else None)
        return result
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse AI response as JSON. Raw response:\n{result[:500]}"
        ) from e
    parsed.pop("_inferred", None)
    _validate_or_warn(parsed, columns=col_list if col_list else None)
    return parsed


def save_config(config: dict, filepath: str | Path) -> Path:
    """Save a configuration dict to a JSON file.

    Parameters
    ----------
    config : dict
        Configuration dictionary (from generate_config or generate_config_with_columns).
    filepath : str or Path
        Where to save the JSON file. Will create parent directories if needed.

    Returns
    -------
    Path
        The path to the saved file.

    Example
    -------
    >>> config = generate_config_with_columns("...", api_key="...", columns=df)
    >>> save_config(config, "my_project/config.json")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return filepath

