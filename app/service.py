from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from app import config
from app.clip_encoder import ClipEncoder
from app.faiss_store import FaissStore
from app.metadata_db import MetadataDB
from app.schemas import SimilarImage


class ImageSearchService:
    def __init__(self) -> None:
        config.ensure_dirs()
        self.encoder: Optional[ClipEncoder] = None
        self.store: Optional[FaissStore] = None
        self.meta: Optional[MetadataDB] = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def device(self) -> str:
        return self.encoder.device if self.encoder else "n/a"

    def startup(self) -> None:
        config.ensure_dirs()
        self.encoder = ClipEncoder()
        self.meta = MetadataDB(config.META_DB_PATH)
        self.store = FaissStore.load_or_create(
            config.FAISS_INDEX_PATH, dim=config.CLIP_EMBED_DIM
        )
        self._ready = True

    def shutdown(self) -> None:
        if self.meta is not None:
            self.meta.close()
        self._ready = False

    def _require_ready(self) -> None:
        if not self._ready or self.encoder is None or self.store is None or self.meta is None:
            raise RuntimeError("Service not initialized")

    def search_bytes(
        self,
        data: bytes,
        filename: str,
        top_k: int = config.DEFAULT_TOP_K,
        threshold: Optional[float] = None,
    ) -> list[SimilarImage]:
        self._require_ready()
        assert self.encoder and self.store and self.meta

        # Over-fetch to compensate for soft-deleted hits
        fetch_k = min(max(top_k * 3, top_k), max(self.store.ntotal, top_k))
        query = self.encoder.encode_image(data)
        scores, ids = self.store.search(query, fetch_k)

        results: list[SimilarImage] = []
        candidate_ids = [int(i) for i in ids[0] if int(i) >= 0]
        records = self.meta.get_many(candidate_ids)

        for score, image_id in zip(scores[0], ids[0]):
            image_id = int(image_id)
            if image_id < 0:
                continue
            rec = records.get(image_id)
            if rec is None or rec.deleted:
                continue
            s = float(score)
            if threshold is not None and s < threshold:
                continue
            results.append(
                SimilarImage(
                    id=rec.id,
                    path=rec.path,
                    filename=rec.filename,
                    score=round(s, 6),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def add_image_bytes(self, data: bytes, original_filename: str) -> tuple[int, str, str]:
        self._require_ready()
        assert self.encoder and self.store and self.meta

        suffix = Path(original_filename).suffix.lower() or ".jpg"
        if suffix not in config.IMAGE_EXTENSIONS:
            suffix = ".jpg"
        safe_name = f"{uuid.uuid4().hex}{suffix}"
        dest = config.IMAGES_DIR / safe_name
        dest.write_bytes(data)

        rel_path = str(dest.relative_to(config.ROOT_DIR)).replace("\\", "/")
        image_id = self.meta.insert(path=rel_path, filename=original_filename)
        vector = self.encoder.encode_image(data)
        self.store.add(
            np.asarray([image_id], dtype=np.int64),
            vector,
        )
        self.store.save(config.FAISS_INDEX_PATH)
        return image_id, rel_path, original_filename

    def add_image_file(self, file_path: Path, store_copy: bool = False) -> int:
        """Index an existing file on disk (used by build script / add path)."""
        self._require_ready()
        assert self.encoder and self.store and self.meta

        file_path = Path(file_path).resolve()
        if store_copy:
            suffix = file_path.suffix.lower() or ".jpg"
            dest = config.IMAGES_DIR / f"{uuid.uuid4().hex}{suffix}"
            shutil.copy2(file_path, dest)
            target = dest
            filename = file_path.name
        else:
            target = file_path
            filename = file_path.name

        try:
            rel_path = str(target.relative_to(config.ROOT_DIR)).replace("\\", "/")
        except ValueError:
            rel_path = str(target).replace("\\", "/")

        if self.meta.path_exists(rel_path):
            raise ValueError(f"Already indexed: {rel_path}")

        image_id = self.meta.insert(path=rel_path, filename=filename)
        vector = self.encoder.encode_image(target)
        self.store.add(
            np.asarray([image_id], dtype=np.int64),
            vector,
        )
        return image_id

    def soft_delete(self, image_id: int) -> bool:
        self._require_ready()
        assert self.store and self.meta
        ok = self.meta.soft_delete(image_id)
        if ok:
            self.store.remove_ids([image_id])
            self.store.save(config.FAISS_INDEX_PATH)
        return ok

    def save_index(self) -> None:
        self._require_ready()
        assert self.store
        self.store.save(config.FAISS_INDEX_PATH)

    def stats(self) -> dict:
        self._require_ready()
        assert self.store and self.meta
        return {
            "total_vectors": self.store.ntotal,
            "active_images": self.meta.count_active(),
            "deleted_images": self.meta.count_deleted(),
            "embed_dim": self.store.dim,
            "index_path": str(config.FAISS_INDEX_PATH),
            "meta_db_path": str(config.META_DB_PATH),
        }


# Singleton used by FastAPI
service = ImageSearchService()
