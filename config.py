from __future__ import annotations

import os
from dataclasses import dataclass, replace

from dotenv import load_dotenv

load_dotenv()


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(slots=True)
class Settings:
    api_key: str
    base_url: str
    provider: str
    model: str
    output_dir: str
    enable_mllm: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
            provider=os.getenv("UI_EVAL_PROVIDER", "openai_responses").strip(),
            model=os.getenv("UI_EVAL_MODEL", "gpt-4.1-mini").strip(),
            output_dir=os.getenv("UI_EVAL_OUTPUT_DIR", "outputs").strip(),
            enable_mllm=_read_bool("UI_HIERARCHY_ENABLE_MLLM", True),
        )

    def with_cli_overrides(self, *, skip_llm: bool) -> "Settings":
        return replace(self, enable_mllm=self.enable_mllm and not skip_llm)
