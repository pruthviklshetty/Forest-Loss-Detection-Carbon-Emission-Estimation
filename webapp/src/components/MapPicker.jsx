import { useEffect, useRef, useState } from 'react'
import { MapContainer, TileLayer, Marker, Rectangle, useMapEvents, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import markerIcon from 'leaflet/dist/images/marker-icon.png'
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png'
import markerShadow from 'leaflet/dist/images/marker-shadow.png'
import { geocodePlace, deriveBbox } from '../api.js'
import { Field } from './ui.jsx'
import { num } from '../format.js'

// leaflet's default marker asset paths break under bundlers - point them at the
// imported URLs.
L.Icon.Default.mergeOptions({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
})

function ClickToSet({ onPick }) {
  useMapEvents({ click: (e) => onPick(e.latlng.lat, e.latlng.lng) })
  return null
}

function Recenter({ center }) {
  const map = useMap()
  useEffect(() => {
    if (center) map.setView(center, Math.max(map.getZoom(), 10))
  }, [center, map])
  return null
}

// wsen -> leaflet bounds [[s,w],[n,e]]
const toBounds = (wsen) => [
  [wsen[1], wsen[0]],
  [wsen[3], wsen[2]],
]

export default function MapPicker({ domain, radiusPresets, value, onChange }) {
  const [center, setCenter] = useState(value?.center || null) // [lat, lon]
  const [radiusKm, setRadiusKm] = useState(value?.radius_km || 10)
  const [derived, setDerived] = useState(null)
  const [deriveErr, setDeriveErr] = useState(null)

  const [q, setQ] = useState('')
  const [cands, setCands] = useState(null)
  const [geoBusy, setGeoBusy] = useState(false)
  const [geoErr, setGeoErr] = useState(null)
  const debRef = useRef(null)

  const ext = domain?.domain_extent_wsen
  const presets = radiusPresets || [5, 10, 20]

  // re-derive the authoritative snapped bbox whenever centre / radius change
  useEffect(() => {
    if (!center) {
      setDerived(null)
      onChange?.(null)
      return
    }
    clearTimeout(debRef.current)
    debRef.current = setTimeout(async () => {
      try {
        const d = await deriveBbox(center[0], center[1], radiusKm)
        setDerived(d)
        setDeriveErr(null)
        onChange?.({ center, radius_km: radiusKm, derived: d })
      } catch (e) {
        setDerived(null)
        setDeriveErr(e.message)
        onChange?.({ center, radius_km: radiusKm, derived: null, error: e.message })
      }
    }, 250)
    return () => clearTimeout(debRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [center, radiusKm])

  async function runGeocode(e) {
    e.preventDefault()
    if (!q.trim()) return
    setGeoBusy(true)
    setGeoErr(null)
    setCands(null)
    try {
      const res = await geocodePlace(q)
      if (!res.results.length) setGeoErr(`No match for "${q}".`)
      else setCands(res.results)
    } catch (err) {
      setGeoErr(err.message)
    } finally {
      setGeoBusy(false)
    }
  }

  const pick = (lat, lon) => {
    setCenter([Number(lat), Number(lon)])
    setCands(null)
  }

  return (
    <div className="mappick">
      <form className="geo" onSubmit={runGeocode}>
        <Field label="Search a place">
          <div className="geo__row">
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="e.g. Moodbidri, Karnataka"
            />
            <button type="submit" disabled={geoBusy}>
              {geoBusy ? '…' : 'Search'}
            </button>
          </div>
        </Field>
        {geoErr && <p className="error-text">{geoErr}</p>}
        {cands && (
          <ul className="geo__cands">
            {cands.map((c, i) => (
              <li key={i}>
                <button type="button" onClick={() => pick(c.lat, c.lon)}>
                  <span className="geo__name">{c.display_name}</span>
                  <span className="muted small">
                    {c.type ? `${c.type} · ` : ''}
                    {num(c.lat, 4)}, {num(c.lon, 4)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
        <p className="muted small">…or click the map to drop a centre point.</p>
      </form>

      <div className="mappick__map">
        <MapContainer center={center || [13.0, 76.0]} zoom={7} scrollWheelZoom>
          <TileLayer
            attribution='&copy; OpenStreetMap contributors'
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <ClickToSet onPick={pick} />
          {center && <Recenter center={center} />}
          {center && <Marker position={center} />}
          {ext && (
            <Rectangle
              bounds={toBounds(ext)}
              pathOptions={{ color: '#2563eb', weight: 1, dashArray: '5 5', fill: false }}
            />
          )}
          {derived?.bbox_wsen && (
            <Rectangle
              bounds={toBounds(derived.bbox_wsen)}
              pathOptions={{
                color: derived.inside_domain_extent ? '#1f7a4d' : '#b91c1c',
                weight: 2,
                fillOpacity: 0.08,
              }}
            />
          )}
        </MapContainer>
      </div>

      <div className="radius">
        <span className="radius__label">Radius</span>
        {presets.map((r) => (
          <button
            key={r}
            type="button"
            className={radiusKm === r ? 'active' : ''}
            onClick={() => setRadiusKm(r)}
          >
            {r} km
          </button>
        ))}
      </div>

      {deriveErr && (
        <p className="error-text">{deriveErr}</p>
      )}
      {derived && (
        <div className={`derived${derived.inside_domain_extent ? '' : ' derived--bad'}`}>
          <div>
            Centre {num(center[0], 4)}, {num(center[1], 4)} · requested{' '}
            {num(derived.requested_side_km, 2)} km square →{' '}
            <b>
              {num(derived.derived_side_km, 2)} km ({derived.n_tiles_per_side}×
              {derived.n_tiles_per_side} tiles, ≈
              {num(
                (derived.derived_side_km * derived.derived_side_km) || 0,
                0,
              )}{' '}
              km²)
            </b>
          </div>
          <div className="muted small">
            {derived.metric_case === 'pooled' ? (
              <>Inside training region <b>{derived.metric_case_region}</b> — pooled metrics apply.</>
            ) : (
              <>Not in a training region — <b>leave-one-region-out</b> metrics apply (lower).</>
            )}
          </div>
          {!derived.inside_domain_extent && (
            <div className="derived__bad">
              This area is outside the Western Ghats domain extent and will be
              refused on submit. Move the centre point inside the dashed box.
            </div>
          )}
        </div>
      )}
    </div>
  )
}
