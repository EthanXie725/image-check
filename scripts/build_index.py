#!/usr/bin/env python3
"""Scan an image directory and build FAISS + SQLite index from scratch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as `python scripts/build_index.py` from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config
from app.clip_encoder import ClipEncoder
from app.faiss_store import FaissStore
from app.metadata_db import MetadataDB


def collect_images(image_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(image_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS:
            files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLIP + FAISS image index")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=config.IMAGES_DIR,
        help="Directory of images to index",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.DEFAULT_BATCH_SIZE,
        help="CLIP batch size",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=True,
        help="Clear existing index/meta and rebuild from scratch (default: always)",
    )
    args = parser.parse_args()

    image_dir = args.image_dir.resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {image_dir}")

    config.ensure_dirs()
    images = collect_images(image_dir)
    if not images:
        raise SystemExit(f"No images found under {image_dir}")

    print(f"Found {len(images)} images in {image_dir}")
    print(f"Loading CLIP ({config.CLIP_MODEL_NAME}/{config.CLIP_PRETRAINED})...")
    encoder = ClipEncoder()
    print(f"Device: {encoder.device}")

    if config.META_DB_PATH.exists():
        config.META_DB_PATH.unlink()
    if config.FAISS_INDEX_PATH.exists():
        config.FAISS_INDEX_PATH.unlink()

    meta = MetadataDB(config.META_DB_PATH)
    store = FaissStore(dim=config.CLIP_EMBED_DIM)

    batch_size = max(1, args.batch_size)
    total = len(images)

    for start in range(0, total, batch_size):
        batch_paths = images[start : start + batch_size]
        try:
            vectors = encoder.encode_images(batch_paths)
        except Exception as exc:  # noqa: BLE001
            print(f"Batch encode failed at {start}: {exc}; falling back to per-image")
            vectors_list = []
            ok_paths = []
            for p in batch_paths:
                try:
                    vectors_list.append(encoder.encode_image(p)[0])
                    ok_paths.append(p)
                except Exception as e:  # noqa: BLE001
                    print(f"  skip {p}: {e}")
            if not ok_paths:
                continue
            batch_paths = ok_paths
            vectors = np.stack(vectors_list, axis=0)

        ids = []
        for p in batch_paths:
            try:
                rel = str(p.resolve().relative_to(config.ROOT_DIR)).replace("\\", "/")
            except ValueError:
                rel = str(p.resolve()).replace("\\", "/")
            image_id = meta.insert(path=rel, filename=p.name)
            ids.append(image_id)

        store.add(np.asarray(ids, dtype=np.int64), vectors)
        done = min(start + batch_size, total)
        print(f"Indexed {done}/{total}")

    store.save(config.FAISS_INDEX_PATH)
    meta.close()
    print(f"Saved FAISS index -> {config.FAISS_INDEX_PATH}")
    print(f"Saved metadata   -> {config.META_DB_PATH}")
    print(f"Total vectors: {store.ntotal}")


if __name__ == "__main__":
    main()
