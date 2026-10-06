from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class ImageRecord:
    id: int
    path: str
    filename: str
    created_at: str
    deleted: int = 0


class MetadataDB:
    """SQLite metadata store keyed by the same ids used in FAISS IndexIDMap2."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def insert(self, path: str, filename: str) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO images (path, filename, created_at, deleted)
                VALUES (?, ?, ?, 0)
                """,
                (path, filename, created_at),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get(self, image_id: int) -> Optional[ImageRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, path, filename, created_at, deleted FROM images WHERE id = ?",
                (image_id,),
            ).fetchone()
        if row is None:
            return None
        return ImageRecord(**dict(row))

    def get_many(self, image_ids: list[int]) -> dict[int, ImageRecord]:
        if not image_ids:
            return {}
        placeholders = ",".join("?" for _ in image_ids)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, path, filename, created_at, deleted
                FROM images
                WHERE id IN ({placeholders})
                """,
                image_ids,
            ).fetchall()
        return {int(r["id"]): ImageRecord(**dict(r)) for r in rows}

    def soft_delete(self, image_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE images SET deleted = 1 WHERE id = ? AND deleted = 0",
                (image_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def count_active(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM images WHERE deleted = 0"
            ).fetchone()
        return int(row["c"])

    def count_deleted(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM images WHERE deleted = 1"
            ).fetchone()
        return int(row["c"])

    def list_active_paths(self) -> list[tuple[int, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, path FROM images WHERE deleted = 0 ORDER BY id"
            ).fetchall()
        return [(int(r["id"]), str(r["path"])) for r in rows]

    def clear_all(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM images")
            self._conn.execute(
                "DELETE FROM sqlite_sequence WHERE name = 'images'"
            )
            self._conn.commit()

    def path_exists(self, path: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM images WHERE path = ? AND deleted = 0 LIMIT 1",
                (path,),
            ).fetchone()
        return row is not None
