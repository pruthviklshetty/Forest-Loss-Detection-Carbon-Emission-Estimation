# Live forest-loss inference service (Phase 9 backend)

FastAPI service that serves the Phase 8 carry-forward checkpoint
(`results/checkpoints/p8_pooled_unet_s44_best.pt`). A request names a Western
Ghats region (or a custom bbox inside the domain extent) and two January-April
date windows; the backend pulls the two Sentinel-2 composites from Earth Engine,
runs the tiled model, and returns the predicted-loss mask, cleared hectares and
committed aboveground CO2.

## Domain constraint (enforced, not advisory)

The model was trained only on Western Ghats moist forest, Jan-Apr composites,
2019 vs 2021. `POST /jobs` returns **HTTP 422** with an explanation when:

- a custom bbox is not fully inside `domain_extent_wsen`
  (`configs/region.yaml`, currently `[74.8, 8.2, 77.9, 16.5]`);
- either date window falls outside January-April, or the two windows are not in
  increasing calendar years;
- the requested area exceeds `SERVE_MAX_AREA_KM2` (default 900 km2).

Preset regions are the four Phase 8 training blocks plus three more inside the
extent (`agumbe`, `silent_valley`, `periyar`) that are **not** in the training
set; `GET /regions` marks each with `in_training_set`. The model card
(`GET /model-card`) carries the leave-one-region-out numbers so the results page
can state that performance outside the training set is measurably lower.

## Earth Engine authentication - service account only

This service never uses `earthengine authenticate`. It needs a **service-account
JSON key**:

1. In the Google Cloud project that has Earth Engine enabled, create a service
   account and grant it the *Earth Engine Resource Viewer* role (plus
   *Service Usage Consumer* if your org requires it).
2. Create a JSON key for that account and download it.
3. Register the service account for Earth Engine at
   <https://code.earthengine.google.com/register> (or via
   `earthengine set_project` from an admin account).
4. Put the key file somewhere **outside version control**. `.gitignore` already
   blocks `backend/*.json`, `service_account*.json`, `*-ee-key.json`,
   `secrets/`; keeping it in `secrets/` is recommended.
5. Point the service at it:

   ```bash
   export EE_SERVICE_ACCOUNT_KEY=/abs/path/to/secrets/ee-service-account.json
   # optional: export EE_PROJECT=your-ee-project        # else key project_id / region.yaml
   ```

   As a fallback the path can be set in `configs/region.yaml` under
   `earth_engine.service_account_key`, but it must still point at a git-ignored
   file.

The key is read once at first job (`serve/eepull.py:init_ee`), used to build
`ee.ServiceAccountCredentials`, and never logged.

## Run locally

```bash
python -m venv .venv && . .venv/Scripts/activate      # or .venv/bin/activate
pip install -r requirements.txt          # project deps
pip install -r backend/requirements.txt  # fastapi, uvicorn, pydantic

export EE_SERVICE_ACCOUNT_KEY=/abs/path/to/secrets/ee-service-account.json
cd backend
uvicorn serve.main:app --reload --port 8000
```

`GET http://localhost:8000/health` should return `{"ok": true, "checkpoint_present": true}`.

## Configuration (environment variables)

| var | default | meaning |
|---|---|---|
| `EE_SERVICE_ACCOUNT_KEY` | – (required) | path to the service-account JSON key |
| `EE_PROJECT` | key `project_id` / region.yaml | Earth Engine cloud project |
| `SERVE_MAX_AREA_KM2` | `900` | reject requests larger than this |
| `SERVE_JOB_TIMEOUT_S` | `300` | fail a job that runs longer |
| `SERVE_CLOUD_FLAG_PCT` | `35` | flag a composite with more cloud/no-data than this |
| `SERVE_MIN_SCENES` | `8` | flag a window with fewer clear scenes |
| `SERVE_JOBS_DIR` | `backend/_jobs` | per-job scratch + served masks |
| `SERVE_CORS_ORIGINS` | `http://localhost:5173,...` | allowed frontend origins |

## API

| method | path | purpose |
|---|---|---|
| GET | `/health` | liveness + checkpoint presence |
| GET | `/regions` | preset regions (`in_training_set` flag) |
| GET | `/domain` | domain extent, training windows, guard-rail caps |
| GET | `/model-card` | in-domain and leave-one-region-out metrics, read live from `results/**.json` |
| POST | `/jobs` | body `{region_id \| bbox_wsen, window_t:[s,e], window_t1:[s,e]}` → `{id, status}` (422 if out of domain) |
| GET | `/jobs/{id}` | `{status, progress, message, result, error}`; status ∈ queued/fetching/inferring/estimating/done/failed |
| GET | `/jobs/{id}/mask.png` | prediction overlay PNG (409 until ready) |

Jobs run one at a time in a worker thread; state is in memory and lost on
restart (single-instance demo, not a durable queue). A GEE pull is ~30-90 s, so
the frontend polls `/jobs/{id}`.

## Deploying to Render

The `torch` (CPU) + `rasterio` + `earthengine-api` stack needs well over Render's
512 MB free tier; use at least a 2 GB instance, or lower `SERVE_MAX_AREA_KM2` so
each job's raster stays small.

- **Build:** `pip install -r requirements.txt -r backend/requirements.txt`
- **Start:** `uvicorn serve.main:app --host 0.0.0.0 --port $PORT` (working dir `backend/`)
- **Env:** set `EE_SERVICE_ACCOUNT_KEY` via a Render *secret file* and point the
  var at its mount path; set `SERVE_CORS_ORIGINS` to the deployed frontend URL.
- Install `torch` from the CPU wheel index to keep the image small:
  `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu`.
