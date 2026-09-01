# Live forest-loss inference service (Phase 9 backend)

FastAPI service that serves a pooled multi-region carry-forward checkpoint -
by default the 2021 -> 2023 model (`results/checkpoints/p10_pooled_unet_s43_best.pt`);
set `SERVE_CHECKPOINT_STEM=p8_pooled_unet_s44` for the 2019 -> 2021 model. A
request selects an area of
interest three ways - a **centre point + radius** (primary), a **preset region**,
or a raw **bounding box** (advanced/fallback) - plus two January-April date
windows; the backend pulls the two Sentinel-2 composites from Earth Engine, runs
the tiled model, and returns the predicted-loss mask, cleared hectares and
committed aboveground CO2. It fetches imagery from Earth Engine for the chosen
coordinates; it does not and cannot accept uploaded photos or map screenshots
(the model needs bands B3/B4/B8/B11 at two dates).

## Area of interest - point and radius

`POST /jobs` and `GET /derive-bbox` accept `center: [lat, lon]` + `radius_km`.
The derived AOI is a square of side `2 * radius_km`, **snapped up** to a whole
number of 2.56 km model tiles (256 px x 10 m), centred on the point. The UI
offers radius buttons 5 / 10 / 20 km; the API takes any positive value up to the
cap.

- **Minimum:** an AOI smaller than one tile (2.56 km/side, i.e. radius < 1.28 km)
  is **refused with an explanation** - not silently padded up. Hansen GFC labels
  are 30 m, so there is no meaningful signal below a tile.
- **Maximum:** `radius_km` is capped at `SERVE_MAX_RADIUS_KM` (default 20 km).
- The `SERVE_MAX_AREA_KM2` cap applies to the raw-bbox path only; the radius cap
  bounds the point path.

`GET /derive-bbox?lat=&lon=&radius_km=` returns the authoritative snapped bbox,
tile count, `metric_case`, and `inside_domain_extent` so the map can draw exactly
what will run before submit.

## Geocoding

`GET /geocode?q=<place>` proxies Nominatim (OpenStreetMap, no API key) with an
identifying User-Agent, an in-process LRU cache, and a >= 1.1 s interval between
upstream calls. It returns **every candidate**; the frontend chooses - nothing is
auto-selected. A failed lookup returns HTTP 502 with a message.

## Domain constraint (enforced, not advisory)

The model was trained only on Western Ghats moist forest, Jan-Apr composites,
2019 vs 2021. `POST /jobs` returns **HTTP 422** with an explanation when:

- the AOI (derived bbox or raw bbox) is not fully inside `domain_extent_wsen`
  (`configs/region.yaml`, currently `[74.8, 8.2, 77.9, 16.5]`);
- either date window falls outside January-April, or the two windows are not in
  increasing calendar years;
- the AOI is smaller than one 2.56 km tile;
- (raw-bbox path) the area exceeds `SERVE_MAX_AREA_KM2` (default 900 km2).

## Which metric applies (`metric_case`)

Every job result carries `metric_case`: **`pooled`** if the AOI sits entirely
inside one of the four training regions (in-domain, strict IoU ~0.176 +/- 0.026),
otherwise **`loro`** - the leave-one-region-out regime (mean strict IoU ~0.092,
about half). Point-and-radius and custom-bbox queries are almost always `loro`;
the results page headlines the applicable case and never silently shows the
pooled number for a `loro` AOI. Preset regions are the four training blocks plus
three in-extent non-training blocks (`agumbe`, `silent_valley`, `periyar`);
`GET /regions` marks each with `in_training_set`.

Results for an AOI below `SERVE_SMALL_AREA_KM2` (default 25 km2) also carry
`small_area: true`; at ~0.3% loss prevalence the expected true-loss pixel count
is very low there, so a zero result is expected and a non-zero result is
provisional. The raw predicted pixel count is returned alongside hectares.

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
| `EE_SERVICE_ACCOUNT_KEY` | – (required) | path to the service-account JSON key (`GEE_KEY_PATH` also accepted) |
| `EE_PROJECT` | key `project_id` / region.yaml | Earth Engine cloud project |
| `SERVE_CHECKPOINT_STEM` | `p10_pooled_unet_s43` | which trained model to serve: `p10_pooled_unet_s43` = 2021→2023 (more recent data, default), `p8_pooled_unet_s44` = 2019→2021 (paper basis). The results page, model card and `/domain` state the served model's training window; leave-one-region-out was only measured for 2019→2021 and is carried with that label. |
| `SERVE_MAX_AREA_KM2` | `900` | raw-bbox path: reject areas larger than this |
| `SERVE_MAX_RADIUS_KM` | `20` | point path: max radius |
| `SERVE_SMALL_AREA_KM2` | `25` | flag results below this with the low-signal caveat |
| `SERVE_JOB_TIMEOUT_S` | `1800` | fail a job that runs longer |
| `SERVE_CLOUD_FLAG_PCT` | `35` | flag a composite with more cloud/no-data than this |
| `SERVE_MIN_SCENES` | `8` | flag a window with fewer clear scenes |
| `SERVE_NOMINATIM_URL` | OSM public | geocoding endpoint |
| `SERVE_GEOCODE_UA` | app string | User-Agent sent to Nominatim (set a contact) |
| `SERVE_GEOCODE_MIN_INTERVAL_S` | `1.1` | min seconds between Nominatim calls |
| `SERVE_JOBS_DIR` | `backend/_jobs` | per-job scratch + served masks |
| `SERVE_CORS_ORIGINS` | `http://localhost:5173,...` | allowed frontend origins |

## API

| method | path | purpose |
|---|---|---|
| GET | `/health` | liveness + checkpoint presence |
| GET | `/regions` | preset regions (`in_training_set` flag) |
| GET | `/domain` | domain extent, training windows, caps, radius presets |
| GET | `/geocode?q=` | Nominatim candidates for a place name (frontend picks) |
| GET | `/derive-bbox?lat=&lon=&radius_km=` | authoritative snapped bbox + `metric_case` + `inside_domain_extent` |
| GET | `/model-card` | in-domain and leave-one-region-out metrics, read live from `results/**.json` |
| POST | `/jobs` | body `{center:[lat,lon]+radius_km \| region_id \| bbox_wsen, window_t:[s,e], window_t1:[s,e]}` → `{id, status, spec}` (422 if out of domain / below one tile) |
| GET | `/jobs/{id}` | `{status, progress, message, result, error}`; status ∈ queued/fetching/inferring/estimating/done/failed. `result` carries `metric_case`, `small_area`, `area_carbon.predicted_loss_pixels` |
| GET | `/jobs/{id}/mask.png` | prediction overlay PNG (409 until ready) |

Jobs run one at a time in a worker thread; state is in memory and lost on
restart (single-instance demo, not a durable queue). Timing measured locally
(GPU inference): a ~120 km² bbox takes ~3 min end to end (two Earth Engine
composite pulls dominate); a full ~850 km² preset region is ~15-25 min. The
frontend polls `/jobs/{id}` every 2 s. For quick iteration use a sub-bbox.

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
