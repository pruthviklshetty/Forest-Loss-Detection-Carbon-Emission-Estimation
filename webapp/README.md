# Live inference UI (Phase 9 frontend)

Vite + React + plain CSS + Recharts. Talks to the FastAPI service in
[`../backend`](../backend). Separate from [`../frontend`](../frontend), which is
the static results dashboard.

## Flow

1. **Region** — a preset (four Phase 8 training blocks + three in-domain
   non-training blocks) or a custom bounding box.
2. **Date windows** — two Jan–Apr windows, defaulting to the training composites
   (2019 vs 2021).
3. **Run** — `POST /jobs`; the domain gate rejects out-of-extent bboxes,
   non-Jan–Apr windows and oversized areas with the backend's message.
4. **Poll** — `GET /jobs/{id}` every 2 s with a visible progress bar and step
   list (queued → fetching → inferring → estimating → done).
5. **Results** — prediction overlay PNG, predicted cleared hectares, committed
   CO₂ (exponential regression primary + 3-bin baseline, with the
   aboveground/CO₂-only scope stated), cloud / no-data cover and scene count for
   **both** composites with a prominent flag when either is poor, and the model
   card: in-domain test metrics as mean ± sd plus the leave-one-region-out
   numbers showing out-of-training-set accuracy is roughly half.

## Ground rules honoured

- No metric is hard-coded in a component. Everything on the model card and in
  the results comes from the backend, which reads `results/**.json` live.
- A missing field renders as `—` (see `format.js` `PENDING`), never a
  plausible-looking placeholder.
- The domain notice, the cloud-quality flag and the metric uncertainty are
  always visible and cannot be dismissed.

## Run

```bash
cp .env.example .env.local          # set VITE_API_BASE if backend isn't on :8000
npm install
npm run dev                         # http://localhost:5173
```

The backend must be running first (`cd ../backend && uvicorn serve.main:app --port 8000`)
with `EE_SERVICE_ACCOUNT_KEY` set — see [`../backend/README.md`](../backend/README.md).

## Build / deploy

```bash
npm run build        # -> dist/
```

`base: './'` keeps asset paths relative, so `dist/` can be served from any
static host (Render static site, Netlify, GitHub Pages). Set `VITE_API_BASE` at
build time to the deployed backend URL, and add that frontend origin to the
backend's `SERVE_CORS_ORIGINS`.
