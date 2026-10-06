from typing import Optional

from pydantic import BaseModel, Field


class SimilarImage(BaseModel):
    id: int
    path: str
    filename: str
    score: float = Field(..., description="Cosine similarity in [-1, 1], closer to 1 is more similar")


class SearchResponse(BaseModel):
    query_filename: str
    top_k: int
    threshold: Optional[float] = None
    results: list[SimilarImage]


class AddImageResponse(BaseModel):
    id: int
    path: str
    filename: str
    message: str = "indexed"


class DeleteResponse(BaseModel):
    id: int
    deleted: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_loaded: bool
    device: str
    detail: Optional[str] = None


class StatsResponse(BaseModel):
    total_vectors: int
    active_images: int
    deleted_images: int
    embed_dim: int
    index_path: str
    meta_db_path: str
