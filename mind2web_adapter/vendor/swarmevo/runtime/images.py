"""Recoverable image evidence and provider-independent typed user content."""
from __future__ import annotations

import base64
import hashlib
import io
from copy import deepcopy
from pathlib import Path

IMAGE_POLICY = {"version": 1, "encoding": "original-bytes-base64", "detail": "high",
                "resize": False, "crop": False, "annotation": False}


def image_bytes(asset: dict) -> bytes:
    blob = Path(asset["path"]).read_bytes()
    if (hashlib.sha256(blob).hexdigest() != asset["sha256"]
            or len(blob) != asset["bytes"]):
        raise ValueError("image evidence content changed")
    return blob


def validate_image(asset: dict) -> dict:
    from PIL import Image

    blob = image_bytes(asset)
    with Image.open(io.BytesIO(blob)) as image:
        mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}.get(image.format)
        if mime is None:
            raise ValueError(f"unsupported screenshot format: {image.format}")
        width, height = image.size
        image.verify()
    with Image.open(io.BytesIO(blob)) as image:
        image.load()
    return {**asset, "mime_type": mime, "width": width, "height": height,
            "policy": deepcopy(asset.get("policy", IMAGE_POLICY))}


def user_content(text: str, images: list[dict]) -> str | list[dict]:
    if not images:
        return text
    return [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {
            "url": f"data:{asset['mime_type']};base64," + base64.b64encode(image_bytes(asset)).decode(),
            "detail": asset["policy"]["detail"]}} for asset in images]


def content_text(content: str | list[dict]) -> str:
    if isinstance(content, str):
        return content
    return "".join(part["text"] for part in content if part.get("type") == "text")


def image_usage(messages: list[dict]) -> dict:
    sizes = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            url = part["image_url"]["url"]
            header, data = url.split(",", 1)
            blob = base64.b64decode(data, validate=True)
            from PIL import Image

            with Image.open(io.BytesIO(blob)) as image:
                width, height = image.size
            sizes.append({"width": width, "height": height, "bytes": len(blob),
                          "encoded_chars": len(data), "sha256": hashlib.sha256(blob).hexdigest(),
                          "mime_type": header.removeprefix("data:").split(";")[0],
                          "detail": part["image_url"].get("detail")})
    return {"image_count": len(sizes), "image_bytes": sum(s["bytes"] for s in sizes),
            "images": sizes}


def anthropic_messages(messages: list[dict]) -> list[dict]:
    converted = []
    for message in messages:
        if message["role"] == "system":
            continue
        content = message["content"]
        if isinstance(content, list):
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append(dict(part))
                elif part["type"] == "image_url":
                    header, data = part["image_url"]["url"].split(",", 1)
                    parts.append({"type": "image", "source": {"type": "base64",
                        "media_type": header.removeprefix("data:").split(";")[0], "data": data}})
                else:
                    raise ValueError("unsupported typed content for Anthropic")
            content = parts
        converted.append({**message, "content": content})
    return converted
