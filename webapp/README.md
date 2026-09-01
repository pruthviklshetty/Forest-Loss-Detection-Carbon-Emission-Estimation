# Live inference UI (Phase 9 frontend)

Vite + React + plain CSS + Recharts + Leaflet. Talks to the FastAPI service in
[`../backend`](../backend). Separate from [`../frontend`](../frontend), which is
the static results dashboard.

## Flow

1. **Area of interest** — three modes:
   - **Point & radius** (default) — drop a centre point by clicking a Leaflet/OSM
     map or by searching a place name (Nominatim, via the backend; ambiguous
     queries show all candidates and you pick). Choose a radius: 5, 10 or 20 km.
     The map draws the *authoritative* snapped bbox (from `/derive-bbox`) — a
     whole number of 2.56 km model tiles — and states whether it's in a training
     region (`pooled` metrics) or not (`loro` metrics), before you submit. An AOI
     smaller than one tile, or outside the domain extent, is blocked.
   - **Preset region** — the four Phase 8 training blocks + three in-domain
     non-training blocks.
   - **Advanced: bounding box** — raw `[W,S,E,N]`, fallback path.
2. **Date windows** — two Jan–Apr windows, defaulting to the training composites
   (2019 vs 2021).
3. **Run** — `POST /jobs`; the domain gate rejects out-of-extent areas, sub-tile
   areas, non-Jan–Apr windows and oversized boxes with the backend's message.
4. **Poll** — `GET /jobs/{id}` every 2 s with a progress bar and step list
   (queued → fetching → inferring → estimating → done).
5. **Results** — prediction overlay PNG; the **raw predicted pixel count** next to
   predicted hectares (so you can see how thin the signal is); the hectares
   flagged as *not calibrated* with the measured pred/GFC ratio; committed CO₂
   (exponential regression primary + 3-bin baseline, aboveground/CO₂-only scope
   stated); cloud / no-data cover + scene count for **both** composites with a
   prominent flag when either is poor; and the model card headlining the
   **applicable metric case** — for a `loro` AOI (the common point-and-radius
   case) the leave-one-region-out figure (~0.092), not the pooled one, with a
   note that the area was not in the training set. Any AOI under ~25 km² also
   carries a "zero is expected, non-zero is provisional" caveat.

The app fetches Sentinel-2 from Earth Engine for the chosen coordinates. It does
**not** accept uploaded photos, screenshots or map images — the model needs
bands B3/B4/B8/B11 at two dates, which an RGB image does not contain.

## Ground rules honoured

- No metric is hard-coded in a component. Everything on the model card and in
  the results comes from the backend, which reads `results/**.json` live.
- A missing field renders as `—` (see `format.js` `PENDING`), never a
  plausible-looking placeholder.
- The domain notice, the cloud-quality flag and the metric uncertainty are
  always visible and cannot be dismissed.

## Run

```bash
cp .env.example .env.local          # set VITE_API_URL if backend isn't on 127.0.0.1:8000
npm install
npm run dev                         # http://localhost:5173
```

The backend must be running first (`cd ../backend && uvicorn serve.main:app --port 8000`)
with `GEE_KEY_PATH` set — see [`../backend/README.md`](../backend/README.md).
With no env vars, the frontend talks to `http://127.0.0.1:8000`.

## Build / deploy

```bash
VITE_API_URL=https://<backend-host> npm run build   # -> dist/
```

`base: './'` keeps asset paths relative, so `dist/` can be served from any
static host (Railway static service, Netlify, GitHub Pages, …). `VITE_API_URL`
is baked in at build time; add the frontend's origin to the backend's
`ALLOWED_ORIGINS`.
