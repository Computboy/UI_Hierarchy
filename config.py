from __future__ import annotations

import os
from dataclasses import dataclass, replace
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _looks_like_openai_host(base_url: str) -> bool:
    host = urlparse(base_url).netloc.lower()
    return host == "api.openai.com" or host.endswith(".openai.com")


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
            provider=os.getenv("UI_EVAL_PROVIDER", "auto").strip(),
            model=os.getenv("UI_EVAL_MODEL", "gpt-4.1-mini").strip(),
            output_dir=os.getenv("UI_EVAL_OUTPUT_DIR", "outputs").strip(),
            enable_mllm=_read_bool("UI_HIERARCHY_ENABLE_MLLM", True),
        )

    def with_cli_overrides(
        self,
        *,
        skip_llm: bool,
        base_url: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> "Settings":
        updated = replace(self, enable_mllm=self.enable_mllm and not skip_llm)

        if base_url and base_url.strip():
            updated = replace(updated, base_url=base_url.strip())
        if provider and provider.strip():
            updated = replace(updated, provider=provider.strip())
        if model and model.strip():
            updated = replace(updated, model=model.strip())

        return updated

    def is_official_openai_endpoint(self) -> bool:
        return _looks_like_openai_host(self.base_url)

    def resolved_provider(self) -> str:
        provider = (self.provider or "auto").strip().lower()
        aliases = {
            "auto": "auto",
            "responses": "openai_responses",
            "openai": "openai_responses",
            "openai_responses": "openai_responses",
            "chat": "openai_compatible_chat",
            "compatible_chat": "openai_compatible_chat",
            "openai_compatible_chat": "openai_compatible_chat",
        }
        provider = aliases.get(provider, provider)

        if provider == "auto":
            if self.is_official_openai_endpoint():
                return "openai_responses"
            return "openai_compatible_chat"

        return provider

    def llm_runtime_summary(self) -> str:
        return (
            f"provider={self.resolved_provider()}, "
            f"model={self.model}, "
            f"base_url={self.base_url}"
        )
