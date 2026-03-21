from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    api_key: str
    base_url: str
    provider: str
    model: str
    output_dir: str

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("环境变量 OPENAI_API_KEY 未设置。")

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        provider = os.getenv("UI_EVAL_PROVIDER", "openai_responses").strip()
        model = os.getenv("UI_EVAL_MODEL", "gpt-4.1-mini").strip()
        output_dir = os.getenv("UI_EVAL_OUTPUT_DIR", "outputs").strip()

        return cls(
            api_key=api_key,
            base_url=base_url,
            provider=provider,
            model=model,
            output_dir=output_dir,
        )