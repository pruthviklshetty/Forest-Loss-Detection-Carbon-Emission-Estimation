// Formatting helpers. Every metric shown in the UI passes through here; when a
// value is missing we return a PENDING sentinel, never a fabricated number.

export const PENDING = '—'

export function num(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return PENDING
  return Number(x).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

export function int(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return PENDING
  return Math.round(x).toLocaleString()
}

export function pct(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return PENDING
  return `${num(x, digits)}%`
}

// Render a {mean, sd} interval as "0.176 ± 0.026". If sd is missing, show mean
// alone; if mean is missing, PENDING.
export function interval(obj, digits = 3) {
  if (!obj || obj.mean === null || obj.mean === undefined) return PENDING
  const m = num(obj.mean, digits)
  if (obj.sd === null || obj.sd === undefined) return m
  return `${m} ± ${num(obj.sd, digits)}`
}

export function ratio(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return PENDING
  return `${num(x, 2)}×`
}

export function isMissing(x) {
  return x === null || x === undefined || Number.isNaN(x)
}
