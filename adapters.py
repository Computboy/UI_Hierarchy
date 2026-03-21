from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Protocol

from openai import OpenAI

from config import Settings
from prompts import SYSTEM_PROMPT, build_user_prompt
from schemas import get_json_schema_dict


class MultimodalEvaluator(Protocol):
    def evaluate_image(self, image_path: str) -> str:
        ...


def encode_image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/png"

    image_bytes = path.read_bytes()
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{base64_str}"


class OpenAIResponsesAdapter:
    """
    适用于 OpenAI 官方 Responses API。
    很多 OpenAI 兼容平台未必支持 text.format/json_schema，
    所以兼容性最强的仍然是官方接口。
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

    def evaluate_image(self, image_path: str) -> str:
        image_name = Path(image_path).name
        data_url = encode_image_to_data_url(image_path)

        resp = self.client.responses.create(
            model=self.settings.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": SYSTEM_PROMPT}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": build_user_prompt(image_name)},
                        {"type": "input_image", "image_url": data_url},
                    ],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ui_hierarchy_evaluation",
                    "schema": get_json_schema_dict()["schema"],
                    "strict": True,
                }
            },
        )

        # 对于多数官方 SDK，这里可以直接取 output_text
        return resp.output_text


class OpenAICompatibleChatAdapter:
    """
    适用于许多 OpenAI-compatible 的 /chat/completions 多模态服务。
    这类接口一般不一定支持严格 json_schema，因此用 prompt 约束 + 后解析校验。
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

    def evaluate_image(self, image_path: str) -> str:
        image_name = Path(image_path).name
        data_url = encode_image_to_data_url(image_path)

        completion = self.client.chat.completions.create(
            model=self.settings.model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_user_prompt(image_name)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )

        return completion.choices[0].message.content


def build_evaluator(settings: Settings) -> MultimodalEvaluator:
    provider = settings.provider.lower()
    if provider == "openai_responses":
        return OpenAIResponsesAdapter(settings)
    elif provider == "openai_compatible_chat":
        return OpenAICompatibleChatAdapter(settings)
    else:
        raise ValueError(
            f"不支持的 provider: {settings.provider}，"
            f"可选: openai_responses | openai_compatible_chat"
        )