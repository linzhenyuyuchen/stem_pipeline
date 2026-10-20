#!/usr/bin/env python3
"""Call an OpenAI-compatible VLM with one local image and a prompt."""

from __future__ import annotations

import argparse
import json
import base64
import mimetypes
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_PLACEHOLDER = "<image 1>"
DEFAULT_CONFIG = Path(__file__).with_name("vlm_api.yaml")
# 该端点接受的 maxOutputTokens 上限为 65536（131072 会被 400 硬拒绝：
# "supported range is from 1 to 65537 (exclusive)"）；隐藏思考 token 计入
# completion，长图转录很容易撞顶，因此直接取到上限。
DEFAULT_MAX_TOKENS = 65536


@dataclass(frozen=True)
class VLMConfig:
    api_base: str
    api_key: str
    model: str


@dataclass(frozen=True)
class VLMResponse:
    prediction: str
    finish_reason: str | None
    usage: dict[str, Any] | None


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> VLMConfig:
    """Load and validate the three required VLM settings from YAML."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("缺少 PyYAML；请安装：pip install pyyaml") from exc

    path = Path(config_path)
    try:
        config: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"找不到配置文件：{path}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"配置文件不是有效 YAML：{path}") from exc

    if not isinstance(config, dict):
        raise RuntimeError("配置文件必须是一个 YAML 映射。")

    keys = ("VLM_API_BASE", "VLM_API_KEY", "VLM_MODEL")
    missing = [
        key
        for key in keys
        if not isinstance(config.get(key), str) or not config[key].strip()
    ]
    if missing:
        raise RuntimeError(f"配置文件缺少非空字段：{', '.join(missing)}")

    values = {key: config[key].strip() for key in keys}
    placeholder_values = ("your-", "replace-with-")
    unset = [
        key for key, value in values.items() if any(marker in value for marker in placeholder_values)
    ]
    if unset:
        raise RuntimeError(f"请先替换配置文件中的占位值：{', '.join(unset)}")

    return VLMConfig(
        api_base=values["VLM_API_BASE"],
        api_key=values["VLM_API_KEY"],
        model=values["VLM_MODEL"],
    )


def image_to_data_url(image_path: str | Path) -> str:
    """Encode a local image as a base64 data URL."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到图片：{path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"无法识别图片 MIME 类型：{path}")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_multimodal_content(
    prompt: str,
    image_path: str | Path,
    *,
    placeholder: str = IMAGE_PLACEHOLDER,
    image_detail: str = "auto",
) -> list[dict[str, Any]]:
    """Replace each textual image placeholder with an OpenAI image content part."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt 必须是非空字符串。")
    if placeholder not in prompt:
        raise ValueError(f"prompt 中缺少图片占位符 {placeholder!r}")
    if image_detail not in {"auto", "low", "high"}:
        raise ValueError("image_detail 必须是 auto、low 或 high。")

    data_url = image_to_data_url(image_path)
    content: list[dict[str, Any]] = []
    parts = prompt.split(placeholder)
    for index, text in enumerate(parts):
        if text:
            content.append({"type": "text", "text": text})
        if index < len(parts) - 1:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": image_detail},
                }
            )
    return content


def create_client(
    config: VLMConfig, *, timeout: float = 120, max_retries: int = 2
) -> Any:
    """Create a configured OpenAI SDK client."""
    if timeout <= 0:
        raise ValueError("timeout 必须大于 0。")
    if max_retries < 0:
        raise ValueError("max_retries 不能小于 0。")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai；请安装：pip install openai") from exc

    return OpenAI(
        api_key=config.api_key,
        base_url=config.api_base,
        timeout=timeout,
        max_retries=max_retries,
    )


def generate_response(
    *,
    prompt: str,
    image_path: str | Path,
    config: VLMConfig,
    client: Any | None = None,
    timeout: float = 120,
    max_retries: int = 2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0,
    image_detail: str = "auto",
) -> VLMResponse:
    """Send one image/prompt request and return text plus completion metadata."""
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须大于 0。")
    if temperature < 0:
        raise ValueError("temperature 不能小于 0。")

    api_client = client or create_client(
        config, timeout=timeout, max_retries=max_retries
    )
    completion = api_client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "user",
                "content": build_multimodal_content(
                    prompt, image_path, image_detail=image_detail
                ),
            }
        ],
        temperature=temperature,
        # reasoning_effort=os.environ.get("VLM_REASONING_EFFORT", "high"),
        max_tokens=max_tokens,
        extra_headers={"Idempotency-Key": str(uuid.uuid4())},
    )
    

    # Normalize standard ChatCompletion objects and gateway string responses.
    # Some gateways return the complete HTTP response as a string.
    if isinstance(completion, str):
        raw = completion.strip()
        if raw.startswith("HTTP/"):
            separators = ("\r\n\r\n", "\\r\\n\\r\\n")
            for separator in separators:
                if separator in raw:
                    raw = raw.split(separator, 1)[1].strip()
                    break
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        completion = parsed if isinstance(parsed, (dict, list)) else raw

    if isinstance(completion, str):
        content = completion
        finish_reason = None
        usage = None
    else:
        if isinstance(completion, dict):
            choices = completion.get("choices")
            usage = completion.get("usage")
        else:
            choices = getattr(completion, "choices", None)
            usage = getattr(completion, "usage", None)
        if not choices:
            raise RuntimeError(
                "API 返回格式不兼容：既没有字符串内容，也没有 choices。"
                f" 实际类型为 {type(completion).__name__}。"
            )
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message", {})
            content = message.get("content") if isinstance(message, dict) else None
            finish_reason = choice.get("finish_reason")
        else:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            finish_reason = getattr(choice, "finish_reason", None)

    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("API 返回了空 prediction。")


    if usage is None:
        usage_data = None
    elif hasattr(usage, "model_dump"):
        usage_data = usage.model_dump(mode="json")
    elif isinstance(usage, dict):
        usage_data = dict(usage)
    else:
        raise RuntimeError(f"无法序列化 API usage：{type(usage).__name__}")

    return VLMResponse(
        prediction=content.strip(),
        finish_reason=finish_reason,
        usage=usage_data,
    )


def generate_prediction(
    *,
    prompt: str,
    image_path: str | Path,
    config: VLMConfig,
    client: Any | None = None,
    timeout: float = 120,
    max_retries: int = 2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0,
    image_detail: str = "auto",
) -> str:
    """Backward-compatible helper returning only the prediction text."""
    return generate_response(
        prompt=prompt,
        image_path=image_path,
        config=config,
        client=client,
        timeout=timeout,
        max_retries=max_retries,
        max_tokens=max_tokens,
        temperature=temperature,
        image_detail=image_detail,
    ).prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Local image path.")
    parser.add_argument("--prompt", required=True, help="Prompt containing <image 1>.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        prediction = generate_prediction(
            prompt=args.prompt,
            image_path=args.image,
            config=load_config(args.config),
            timeout=args.timeout,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            image_detail=args.image_detail,
        )
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
