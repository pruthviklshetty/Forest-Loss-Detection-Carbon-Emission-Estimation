import { Field } from './ui.jsx'

// Region selection: a preset (4 training blocks + 3 in-extent extras) or a
// custom bbox. Presets that are not in the training set are labelled so, and
// the results page repeats the out-of-training-set performance caveat.
export default function RegionPicker({
  regions,
  mode,
  setMode,
  regionId,
  setRegionId,
  bbox,
  setBbox,
}) {
  const selected = regions?.find((r) => r.id === regionId)

  return (
    <div className="picker">
      <div className="picker__tabs">
        <button
          type="button"
          className={mode === 'preset' ? 'active' : ''}
          onClick={() => setMode('preset')}
        >
          Preset region
        </button>
        <button
          type="button"
          className={mode === 'bbox' ? 'active' : ''}
          onClick={() => setMode('bbox')}
        >
          Custom bounding box
        </button>
      </div>

      {mode === 'preset' && (
        <>
          <Field label="Region">
            <select value={regionId} onChange={(e) => setRegionId(e.target.value)}>
              <option value="">Select a region…</option>
              {(regions || []).map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id} — {r.in_training_set ? 'training region' : 'in domain, not in training set'}
                </option>
              ))}
            </select>
          </Field>
          {selected && (
            <div className="picker__meta">
              <div>{selected.name}</div>
              <div className="muted">
                bbox [{selected.bbox_wsen.map((x) => x.toFixed(2)).join(', ')}] · ≈
                {selected.area_km2} km²
                {selected.gfc_loss_ha_2019_20 != null && (
                  <> · Hansen 2019–20 loss {selected.gfc_loss_ha_2019_20} ha</>
                )}
              </div>
              {!selected.in_training_set && (
                <div className="picker__warn">
                  Not a training region — expect measurably lower accuracy (see the
                  model card below).
                </div>
              )}
              {selected.admin_context && (
                <p className="muted small">{selected.admin_context}</p>
              )}
            </div>
          )}
        </>
      )}

      {mode === 'bbox' && (
        <div className="bbox-grid">
          {['W', 'S', 'E', 'N'].map((k, i) => (
            <Field key={k} label={`${k} (lon/lat)`}>
              <input
                type="number"
                step="0.01"
                value={bbox[i]}
                onChange={(e) => {
                  const next = bbox.slice()
                  next[i] = e.target.value
                  setBbox(next)
                }}
              />
            </Field>
          ))}
          <p className="muted small span-all">
            Must be fully inside the Western Ghats domain extent. The backend
            re-checks and rejects anything outside it.
          </p>
        </div>
      )}
    </div>
  )
}
