"""Unified LLM client using OpenAI-compatible protocol."""
from typing import Any


class LLMClient:
    """A thin wrapper around OpenAI SDK that supports any OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        try:
            from openai import OpenAI
        except ImportError as _err:
            raise ImportError(
                "AI features require the 'openai' package. "
                "Install with: pip install smallmatprep[ai]"
            ) from _err

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str | dict[str, Any]:
        """Send a chat completion request and return the assistant's content."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if json_mode and isinstance(content, str):
            import json
            return json.loads(content)
        return content
