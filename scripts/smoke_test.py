#!/usr/bin/env python3
"""Quick API smoke test against a running local server."""

from __future__ import annotations

import json
import sys
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent.parent


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}") as resp:
        return json.loads(resp.read().decode())


def multipart(fields: dict[str, tuple[str | None, bytes, str | None]]) -> tuple[bytes, str]:
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    parts: list[bytes] = []
    for name, (filename, data, content_type) in fields.items():
        if filename is not None:
            header = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type or 'application/octet-stream'}\r\n\r\n"
            ).encode()
            parts.append(header + data + b"\r\n")
        else:
            parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                    f"{data.decode()}\r\n"
                ).encode()
            )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def post_multipart(path: str, fields: dict) -> dict:
    body, ctype = multipart(fields)
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        headers={"Content-Type": ctype},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def delete(path: str) -> dict:
    req = urllib.request.Request(f"{BASE}{path}", method="DELETE")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    health = get("/api/health")
    print("health:", health)
    assert health["status"] == "ok"
    assert health["model_loaded"]

    stats = get("/api/stats")
    print("stats:", stats)
    assert stats["active_images"] >= 1

    img_path = ROOT / "data" / "images" / "red_a.jpg"
    search = post_multipart(
        "/api/search",
        {
            "file": ("red_a.jpg", img_path.read_bytes(), "image/jpeg"),
            "top_k": (None, b"5", None),
        },
    )
    print("search:", json.dumps(search, ensure_ascii=False, indent=2))
    assert search["results"], "expected search hits"
    top = search["results"][0]
    assert top["filename"] in ("red_a.jpg", "red_b.jpg"), top
    assert top["score"] > 0.9, top

    buf = BytesIO()
    im = Image.new("RGB", (224, 224), (210, 35, 35))
    ImageDraw.Draw(im).rectangle([45, 45, 175, 175], fill=(250, 90, 90))
    im.save(buf, format="JPEG")
    added = post_multipart(
        "/api/index/add",
        {"file": ("red_new.jpg", buf.getvalue(), "image/jpeg")},
    )
    print("add:", added)
    assert "id" in added

    deleted = delete(f"/api/index/{added['id']}")
    print("delete:", deleted)
    assert deleted["deleted"] is True

    print("SMOKE OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print("SMOKE FAILED:", exc, file=sys.stderr)
        raise
