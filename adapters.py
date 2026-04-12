from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Protocol

from openai import OpenAI

from config import Settings
from prompts import SYSTEM_PROMPT, build_font_hierarchy_user_prompt
from schemas import get_font_hierarchy_schema_dict


JSON_SCHEMA = get_font_hierarchy_schema_dict()


class FontHierarchyEvaluator(Protocol):
    transport_name: str

    def evaluate_font_hierarchy(self, image_path: str) -> str:
        ...


def encode_image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/png"

    image_bytes = path.read_bytes()
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{base64_str}"


def _require_text_output(raw_text: str | None, provider_name: str) -> str:
    if raw_text and raw_text.strip():
        return raw_text
    raise ValueError(f"{provider_name} did not return a text payload.")


def _validate_model_endpoint_compatibility(settings: Settings) -> None:
    model_name = settings.model.strip()
    if not model_name:
        raise ValueError("UI_EVAL_MODEL is empty.")

    if settings.is_official_openai_endpoint() and model_name.lower().startswith("claude"):
        raise ValueError(
            f"Model '{settings.model}' looks like an Anthropic model, "
            "but OPENAI_BASE_URL points to the official OpenAI endpoint. "
            "Use an official GPT model name, or switch OPENAI_BASE_URL back to your compatible router."
        )


def _client_supports_responses_api(settings: Settings) -> bool:
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    return hasattr(client, "responses")


class OpenAIResponsesFontHierarchyAdapter:
    transport_name = "openai_responses"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def evaluate_font_hierarchy(self, image_path: str) -> str:
        image_name = Path(image_path).name
        data_url = encode_image_to_data_url(image_path)

        response = self.client.responses.create(
            model=self.settings.model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": build_font_hierarchy_user_prompt(image_name)},
                        {"type": "input_image", "image_url": data_url},
                    ],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": JSON_SCHEMA["name"],
                    "schema": JSON_SCHEMA["schema"],
                    "strict": True,
                }
            },
        )
        return _require_text_output(getattr(response, "output_text", None), "Responses API")


class OpenAICompatibleChatFontHierarchyAdapter:
    transport_name = "openai_compatible_chat"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def evaluate_font_hierarchy(self, image_path: str) -> str:
        image_name = Path(image_path).name
        data_url = encode_image_to_data_url(image_path)

        completion = self.client.chat.completions.create(
            model=self.settings.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_font_hierarchy_user_prompt(image_name)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )

        content = completion.choices[0].message.content
        return _require_text_output(content, "Chat Completions API")


def build_font_hierarchy_evaluator(settings: Settings) -> FontHierarchyEvaluator | None:
    if not settings.enable_mllm or not settings.api_key:
        return None

    _validate_model_endpoint_compatibility(settings)

    provider = settings.resolved_provider()
    if provider == "openai_responses":
        if not _client_supports_responses_api(settings):
            return OpenAICompatibleChatFontHierarchyAdapter(settings)
        return OpenAIResponsesFontHierarchyAdapter(settings)
    if provider == "openai_compatible_chat":
        return OpenAICompatibleChatFontHierarchyAdapter(settings)
    raise ValueError(
        f"Unsupported provider: {settings.provider}. "
        "Expected one of: auto | openai_responses | openai_compatible_chat"
    )
