from pathlib import Path

# Project root: image-check/
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
INDEX_DIR = DATA_DIR / "index"
UPLOADS_DIR = DATA_DIR / "uploads"

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
META_DB_PATH = INDEX_DIR / "meta.db"

# open_clip model
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_EMBED_DIM = 512

# Defaults
DEFAULT_TOP_K = 10
DEFAULT_BATCH_SIZE = 32

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
