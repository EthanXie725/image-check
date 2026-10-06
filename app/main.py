from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app import config
from app.schemas import (
    AddImageResponse,
    DeleteResponse,
    HealthResponse,
    SearchResponse,
    StatsResponse,
)
from app.service import service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.ensure_dirs()
    service.startup()
    yield
    service.shutdown()


app = FastAPI(
    title="CLIP + FAISS Similar Image Search",
    description="Upload an image to find similar images in the FAISS index.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    index_loaded = (
        service.ready
        and service.store is not None
        and (config.FAISS_INDEX_PATH.exists() or service.store.ntotal >= 0)
    )
    detail = None
    if service.ready and service.store is not None and service.store.ntotal == 0:
        detail = "Index empty. Run scripts/build_index.py or POST /api/index/add first."
    return HealthResponse(
        status="ok" if service.ready else "not_ready",
        model_loaded=service.ready and service.encoder is not None,
        index_loaded=bool(index_loaded),
        device=service.device,
        detail=detail,
    )


@app.get("/api/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    if not service.ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    return StatsResponse(**service.stats())


@app.post("/api/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(...),
    top_k: int = Form(config.DEFAULT_TOP_K),
    threshold: Optional[float] = Form(None),
) -> SearchResponse:
    if not service.ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    if top_k < 1 or top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
    if threshold is not None and not (-1.0 <= threshold <= 1.0):
        raise HTTPException(status_code=400, detail="threshold must be in [-1, 1]")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = file.filename or "query.jpg"
    results = service.search_bytes(
        data, filename=filename, top_k=top_k, threshold=threshold
    )
    return SearchResponse(
        query_filename=filename,
        top_k=top_k,
        threshold=threshold,
        results=results,
    )


@app.post("/api/index/add", response_model=AddImageResponse)
async def add_image(file: UploadFile = File(...)) -> AddImageResponse:
    if not service.ready:
        raise HTTPException(status_code=503, detail="Service not ready")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    filename = file.filename or "upload.jpg"
    try:
        image_id, path, name = service.add_image_bytes(data, filename)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AddImageResponse(id=image_id, path=path, filename=name)


@app.delete("/api/index/{image_id}", response_model=DeleteResponse)
def delete_image(image_id: int) -> DeleteResponse:
    if not service.ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    ok = service.soft_delete(image_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Image id {image_id} not found or already deleted")
    return DeleteResponse(id=image_id, deleted=True, message="soft-deleted and removed from FAISS")
