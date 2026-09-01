"""FastAPI app.

Routes
------
GET  /health                  liveness
GET  /regions                 preset regions (4 training blocks + extras)
GET  /domain                  domain extent, training windows, guard-rail caps
GET  /model-card              in-domain + out-of-training-set metrics (live from JSON)
POST /jobs                    create a job (validates domain); returns {id, status}
GET  /jobs/{id}               job status / progress / result
GET  /jobs/{id}/mask.png      prediction overlay PNG (404 until ready)

Domain rules are enforced in serve.domain and return HTTP 422 with an
explanation - they are not dismissible warnings.
"""

from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import (CHECKPOINT, CLOUD_FLAG_PCT, JOB_TIMEOUT_S, JOBS_DIR,
                     MAX_AREA_KM2, MAX_RADIUS_KM, MIN_SCENES, SMALL_AREA_KM2,
                     TILE_KM)
from .domain import (DomainError, derive_bbox_from_point, domain_extent,
                     metric_case_for_bbox, preset_regions, resolve_request,
                     training_windows)
from .geocode import GeocodeError, geocode
from .jobs import JobStore
from .modelcard import build_model_card
from .pipeline import run_pipeline

app = FastAPI(title="Forest-loss live inference", version="0.9.0")

_origins = os.environ.get(
    "SERVE_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

store = JobStore()
_running: set[asyncio.Task] = set()   # keep task refs so they are not GC'd


class JobRequest(BaseModel):
    region_id: str | None = Field(None, description="preset region id")
    center: list[float] | None = Field(
        None, min_length=2, max_length=2,
        description="[lat, lon] centre point for a point-and-radius AOI")
    radius_km: float | None = Field(
        None, description="AOI radius in km (5/10/20 in the UI); side = 2*radius "
        "snapped to whole 2.56 km tiles; capped at the domain max")
    bbox_wsen: list[float] | None = Field(
        None, min_length=4, max_length=4,
        description="advanced / fallback: [W,S,E,N] lat/lon inside the domain extent")
    window_t: list[str] = Field(..., min_length=2, max_length=2,
                                description="[start, end] ISO dates, Jan-Apr")
    window_t1: list[str] = Field(..., min_length=2, max_length=2,
                                 description="[start, end] ISO dates, Jan-Apr, later year")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "checkpoint_present": CHECKPOINT.exists()}


@app.get("/regions")
def regions() -> dict:
    return {"regions": preset_regions()}


@app.get("/domain")
def domain() -> dict:
    return {
        "domain_extent_wsen": domain_extent(),
        "training_windows": training_windows(),
        "accepted_months": [1, 2, 3, 4],
        "caps": {
            "max_area_km2": MAX_AREA_KM2,
            "max_radius_km": MAX_RADIUS_KM,
            "min_tile_km": TILE_KM,
            "min_radius_km": round(TILE_KM / 2, 2),
            "small_area_km2": SMALL_AREA_KM2,
            "job_timeout_s": JOB_TIMEOUT_S,
            "cloud_flag_pct": CLOUD_FLAG_PCT,
            "min_scenes": MIN_SCENES,
        },
        "radius_presets_km": [5, 10, 20],
        "note": "The model was trained on Western Ghats moist forest, Jan-Apr "
                "composites, 2019 vs 2021. Requests outside the extent or the "
                "Jan-Apr window are refused. One model tile is 2.56 km "
                "(256 px x 10 m); AOIs smaller than a tile are refused.",
    }


@app.get("/geocode")
def geocode_place(q: str) -> dict:
    try:
        results = geocode(q)
    except GeocodeError as exc:
        raise HTTPException(status_code=502, detail={"error": "geocode_failed",
                                                     "message": str(exc)})
    return {"query": q, "count": len(results), "results": results}


@app.get("/derive-bbox")
def derive_bbox(lat: float, lon: float, radius_km: float) -> dict:
    """Authoritative snapped bbox + metric case for a point-and-radius AOI, so
    the map can draw exactly what will be processed before submit."""
    try:
        d = derive_bbox_from_point(lat, lon, radius_km)
    except DomainError as exc:
        raise HTTPException(status_code=422, detail={"error": "out_of_domain",
                                                     "message": str(exc)})
    case, case_region = metric_case_for_bbox(d["bbox_wsen"])
    ext = domain_extent()
    w, s, e, n = d["bbox_wsen"]
    inside = w >= ext[0] and s >= ext[1] and e <= ext[2] and n <= ext[3]
    return {**d, "metric_case": case, "metric_case_region": case_region,
            "inside_domain_extent": inside}


@app.get("/model-card")
def model_card() -> dict:
    return build_model_card()


@app.post("/jobs", status_code=202)
async def create_job(req: JobRequest) -> dict:
    try:
        spec = resolve_request(req.region_id, req.bbox_wsen, req.window_t,
                               req.window_t1, center=req.center,
                               radius_km=req.radius_km)
    except DomainError as exc:
        raise HTTPException(status_code=422, detail={"error": "out_of_domain",
                                                    "message": str(exc)})
    job = await store.create(spec)
    task = asyncio.create_task(store.run(job.id, run_pipeline))
    _running.add(task)
    task.add_done_callback(_running.discard)
    return {"id": job.id, "status": job.status, "spec": spec}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    return job.public()


@app.get("/jobs/{job_id}/mask.png")
def get_mask(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    path = JOBS_DIR / job_id / "mask.png"
    if not path.is_file():
        raise HTTPException(status_code=409,
                            detail=f"mask not ready (job status: {job.status})")
    return FileResponse(path, media_type="image/png")
