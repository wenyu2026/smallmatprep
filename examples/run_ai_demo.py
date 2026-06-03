"""
SmallMatPrep AI 功能演示脚本
=============================

演示两大 AI 增强功能：
  1. 自然语言 → JSON 配置生成（generate_config / generate_config_with_columns）
  2. 智能数据诊断报告（diagnose）

支持多种大模型，详见下方各函数注释。

【前置条件】
  pip install -e ".[ai]"

【运行方式 — 任选一个你有的 API Key】
  # DeepSeek（推荐，国内直连）
  $env:DEEPSEEK_API_KEY='sk-你的key'
  python examples/run_ai_demo.py

  # OpenAI
  $env:OPENAI_API_KEY='sk-你的key'
  python examples/run_ai_demo.py

  # 自定义服务（Kimi / 通义千问 / Ollama …）
  $env:CUSTOM_API_KEY='你的key'
  $env:CUSTOM_BASE_URL='https://api.moonshot.cn/v1'
  $env:CUSTOM_MODEL='moonshot-v1-8k'
  python examples/run_ai_demo.py
"""

import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import smallmatprep as smp

# ══════════════════════════════════════════════════════════════════════════════
#  内部辅助函数：统一演示逻辑（消除三个 demo 函数的重复代码）
# ══════════════════════════════════════════════════════════════════════════════

_PROVIDER_TABLE = """
常见服务配置速查表：
  DeepSeek（推荐）:  base_url="https://api.deepseek.com/v1"   model="deepseek-chat"
  Kimi/Moonshot:     base_url="https://api.moonshot.cn/v1"    model="moonshot-v1-8k"
  阿里通义千问:       base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  model="qwen-plus"
  智谱 GLM:          base_url="https://open.bigmodel.cn/api/paas/v4"  model="glm-4-flash"
  本地 Ollama:        base_url="http://localhost:11434/v1"    model="qwen2.5:7b"  api_key="ollama"
  本地 vLLM:          base_url="http://localhost:8000/v1"     model="你的模型名"  api_key="not-needed"
"""


def _run_demo_for_provider(
    label: str,
    api_key: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    use_columns: bool = True,
) -> None:
    """运行一次完整的 AI 功能演示（配置生成 + 诊断报告 + 存盘）。

    Parameters
    ----------
    label : str
        显示标签（如 "DeepSeek"、"OpenAI"）。
    api_key : str
        API 密钥。
    base_url : str, optional
        服务地址，None 表示使用 OpenAI 默认。
    model : str, optional
        模型名，None 表示使用客户端默认（gpt-4o-mini）。
    use_columns : bool
        是否使用列名感知模式（传入 DataFrame 列名）。
    """
    # ── 加载真实数据 ──
    try:
        df = smp.load_csv("examples/sample_electrolyte.csv")
    except FileNotFoundError:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        df = smp.load_csv(os.path.join(script_dir, "sample_electrolyte.csv"))

    csv_cols = df.columns.tolist()
    print(f"\n数据集: {df.shape[0]} 行 × {df.shape[1]} 列, 列名: {csv_cols}")

    # ── 自然语言描述 ──
    description = (
        "我有一份电解质实验数据，列包括：sample_id（样品编号）、"
        "solvent_A和solvent_B（两种溶剂含量）、salt_S（盐浓度）、"
        "additive_C（添加剂含量，空白表示没加）、target（电导率，我要预测这个）。"
    )

    # ── 1. 生成配置 ──
    print(f"\n{'='*60}")
    print(f"【{label}】AI 配置生成")
    print(f"{'='*60}")
    try:
        if use_columns:
            config = smp.ai.generate_config_with_columns(
                description=description,
                api_key=api_key,
                columns=df,
                base_url=base_url,
                model=model,
            )
        else:
            config = smp.ai.generate_config(
                description=description,
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
        print(json.dumps(config, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ 配置生成失败: {type(e).__name__}: {e}")
        return

    # ── 2. 保存配置 ──
    try:
        saved = smp.ai.save_config(config, "examples/my_ai_config.json")
        print(f"\n✅ 配置已保存到: {saved}")
    except Exception as e:
        print(f"⚠️ 保存配置失败: {e}")

    # ── 3. 生成诊断报告 ──
    target = config.get("target", "target")
    print(f"\n{'='*60}")
    print(f"【{label}】AI 诊断报告（目标列: {target}）")
    print(f"{'='*60}")
    try:
        report = smp.ai.diagnose(
            df=df,
            target=target,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        print(report)
    except Exception as e:
        print(f"❌ 诊断报告生成失败: {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  三个厂商示例函数（共用 _run_demo_for_provider 逻辑）
# ══════════════════════════════════════════════════════════════════════════════

def demo_openai() -> None:
    """使用 OpenAI 官方 API。

    需要 OpenAI API Key: https://platform.openai.com/api_keys
    设置环境变量: $env:OPENAI_API_KEY='sk-你的key'
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("[OpenAI 示例] 跳过 — 请设置 OPENAI_API_KEY 环境变量")
        return
    _run_demo_for_provider("OpenAI", api_key=api_key)


def demo_deepseek() -> None:
    """使用 DeepSeek API（推荐国内用户，直连、便宜、中文好）。

    获取 Key: https://platform.deepseek.com/api_keys
    设置环境变量: $env:DEEPSEEK_API_KEY='sk-你的key'
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("[DeepSeek 示例] 跳过 — 请设置 DEEPSEEK_API_KEY 环境变量")
        return
    _run_demo_for_provider(
        "DeepSeek",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )


def demo_custom() -> None:
    """使用自定义 OpenAI 兼容服务（Kimi / 通义千问 / Ollama / vLLM …）。

    设置三个环境变量:
      $env:CUSTOM_API_KEY='你的key'
      $env:CUSTOM_BASE_URL='https://你的服务地址/v1'
      $env:CUSTOM_MODEL='模型名'
    """ + _PROVIDER_TABLE
    api_key = os.getenv("CUSTOM_API_KEY", "")
    if not api_key:
        print("[自定义示例] 跳过 — 请设置 CUSTOM_API_KEY / CUSTOM_BASE_URL / CUSTOM_MODEL")
        return
    base_url = os.getenv("CUSTOM_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("CUSTOM_MODEL", "deepseek-chat")
    print(f"\n当前自定义配置 → 地址: {base_url}, 模型: {model}")
    _run_demo_for_provider(
        "自定义服务",
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """依次运行所有已配置 API Key 的示例，未配置的自动跳过。"""
    print("╔════════════════════════════════════════════════════════╗")
    print("║     SmallMatPrep AI 功能演示                          ║")
    print("║     支持 OpenAI / DeepSeek / Kimi / 自定义服务        ║")
    print("╚════════════════════════════════════════════════════════╝")

    demo_openai()
    demo_deepseek()
    demo_custom()

    print(f"\n{'='*60}")
    print("演示结束。被跳过的示例请设置对应的环境变量后重试。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

