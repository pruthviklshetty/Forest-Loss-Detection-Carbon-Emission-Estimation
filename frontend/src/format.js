// number formatting helpers. Kept tiny and pure.

export const f3 = (x) => (x == null || Number.isNaN(x) ? '—' : x.toFixed(3))
export const f2 = (x) => (x == null || Number.isNaN(x) ? '—' : x.toFixed(2))
export const f1 = (x) => (x == null || Number.isNaN(x) ? '—' : x.toFixed(1))
export const pct1 = (x) => (x == null ? '—' : `${(x * 100).toFixed(1)}%`)

// tonnes CO2 -> compact "19.9 kt" / "116.7 kt", or plain "1,234 t" under 1 kt
export const kt = (t) => {
  if (t == null) return '—'
  if (t >= 1000) return `${(t / 1000).toFixed(1)} kt`
  return `${Math.round(t).toLocaleString('en-US')} t`
}

export const int = (x) => (x == null ? '—' : Math.round(x).toLocaleString('en-US'))

// a {mean, sd} pair -> "0.158 ±0.016"
export const meanSd = (m, digits = 3) => {
  if (!m || m.mean == null) return '—'
  const d = m.mean.toFixed(digits)
  if (m.sd == null) return d
  return `${d} ±${m.sd.toFixed(digits)}`
}
