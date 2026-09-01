import { Field } from './ui.jsx'
import MapPicker from './MapPicker.jsx'

// Area-of-interest picker. Three modes:
//   point  (default) - centre point (map click or place search) + radius button
//   preset           - one of the four training blocks or three in-domain extras
//   bbox             - advanced / fallback: raw [W,S,E,N]
export default function RegionPicker({
  regions,
  domain,
  radiusPresets,
  mode,
  setMode,
  regionId,
  setRegionId,
  bbox,
  setBbox,
  point,
  setPoint,
}) {
  const selected = regions?.find((r) => r.id === regionId)

  return (
    <div className="picker">
      <div className="picker__tabs">
        <button type="button" className={mode === 'point' ? 'active' : ''} onClick={() => setMode('point')}>
          Point &amp; radius
        </button>
        <button type="button" className={mode === 'preset' ? 'active' : ''} onClick={() => setMode('preset')}>
          Preset region
        </button>
        <button type="button" className={mode === 'bbox' ? 'active' : ''} onClick={() => setMode('bbox')}>
          Advanced: bounding box
        </button>
      </div>

      {mode === 'point' && (
        <MapPicker
          domain={domain}
          radiusPresets={radiusPresets}
          value={point}
          onChange={setPoint}
        />
      )}

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
                  Not a training region — leave-one-region-out metrics apply (see the
                  model card below).
                </div>
              )}
              {selected.admin_context && <p className="muted small">{selected.admin_context}</p>}
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
            Fallback path. Must be fully inside the Western Ghats domain extent and
            at least one 2.56 km model tile on each side. The backend re-checks and
            rejects anything outside those bounds.
          </p>
        </div>
      )}
    </div>
  )
}
