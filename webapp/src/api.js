// Thin fetch wrappers around the FastAPI backend.

export const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:8000').replace(/\/$/, '')

async function j(path, opts) {
  const res = await fetch(API_BASE + path, opts)
  let body = null
  try {
    body = await res.json()
  } catch {
    body = null
  }
  if (!res.ok) {
    // FastAPI puts our domain message in body.detail.message
    const msg =
      (body && body.detail && (body.detail.message || body.detail)) ||
      (body && body.message) ||
      `HTTP ${res.status}`
    const err = new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
    err.status = res.status
    err.body = body
    throw err
  }
  return body
}

export const getRegions = () => j('/regions')
export const getDomain = () => j('/domain')
export const getModelCard = () => j('/model-card')
export const getJob = (id) => j(`/jobs/${id}`)
export const maskUrl = (id) => `${API_BASE}/jobs/${id}/mask.png`

export function createJob(payload) {
  return j('/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

// Poll GET /jobs/{id} until the status is terminal. onTick(job) each poll.
export function pollJob(id, { intervalMs = 2000, onTick } = {}) {
  let stopped = false
  const stop = () => {
    stopped = true
  }
  const run = async () => {
    while (!stopped) {
      let job
      try {
        job = await getJob(id)
      } catch (e) {
        if (onTick) onTick({ status: 'failed', error: e.message, progress: 0, message: 'poll failed' })
        return
      }
      if (onTick) onTick(job)
      if (job.status === 'done' || job.status === 'failed') return
      await new Promise((r) => setTimeout(r, intervalMs))
    }
  }
  run()
  return stop
}
