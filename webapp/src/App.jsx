import { useEffect, useRef, useState } from 'react'
import { getDomain, getRegions, getModelCard, createJob, pollJob } from './api.js'
import { Card, Notice } from './components/ui.jsx'
import DomainNotice from './components/DomainNotice.jsx'
import RegionPicker from './components/RegionPicker.jsx'
import DateWindows from './components/DateWindows.jsx'
import JobProgress from './components/JobProgress.jsx'
import Results from './components/Results.jsx'
import ModelCard from './components/ModelCard.jsx'

const DEFAULT_WINDOWS = {
  window_t: ['2019-01-01', '2019-04-15'],
  window_t1: ['2021-01-01', '2021-04-15'],
}

export default function App() {
  const [domain, setDomain] = useState(null)
  const [regions, setRegions] = useState(null)
  const [modelCard, setModelCard] = useState(null)
  const [loadError, setLoadError] = useState(null)

  const [mode, setMode] = useState('preset')
  const [regionId, setRegionId] = useState('')
  const [bbox, setBbox] = useState(['76.00', '11.55', '76.28', '11.80'])
  const [windows, setWindows] = useState(DEFAULT_WINDOWS)

  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState(null)
  const [job, setJob] = useState(null)
  const stopRef = useRef(null)

  useEffect(() => {
    Promise.all([getDomain(), getRegions(), getModelCard()])
      .then(([d, r, m]) => {
        setDomain(d)
        setRegions(r.regions)
        setModelCard(m)
        if (d?.training_windows?.T && d?.training_windows?.T_plus_1) {
          setWindows({
            window_t: [d.training_windows.T.start, d.training_windows.T.end],
            window_t1: [d.training_windows.T_plus_1.start, d.training_windows.T_plus_1.end],
          })
        }
      })
      .catch((e) => setLoadError(e.message))
    return () => stopRef.current && stopRef.current()
  }, [])

  const busy = submitting || (job && job.status !== 'done' && job.status !== 'failed')

  async function onSubmit(e) {
    e.preventDefault()
    setSubmitError(null)
    setJob(null)
    if (stopRef.current) stopRef.current()

    const payload = { ...windows }
    if (mode === 'preset') {
      if (!regionId) {
        setSubmitError('Pick a region first.')
        return
      }
      payload.region_id = regionId
    } else {
      payload.bbox_wsen = bbox.map(Number)
    }

    setSubmitting(true)
    try {
      const created = await createJob(payload)
      setJob({ id: created.id, status: created.status, progress: 0, message: 'queued' })
      stopRef.current = pollJob(created.id, {
        onTick: (j) => setJob((prev) => ({ ...prev, ...j })),
      })
    } catch (err) {
      setSubmitError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  if (loadError) {
    return (
      <div className="wrap">
        <Notice kind="danger" title="Cannot reach the backend">
          {loadError}. Start the FastAPI service (see <code>backend/README.md</code>) and
          set <code>VITE_API_BASE</code> if it is not on <code>localhost:8000</code>.
        </Notice>
      </div>
    )
  }

  return (
    <div className="wrap">
      <header className="head">
        <h1>Western Ghats forest-loss — live inference</h1>
        <p className="muted">
          Pick a region and two January–April windows; the backend pulls Sentinel-2
          from Earth Engine, runs the Phase 8 model, and returns the loss mask,
          hectares and committed CO₂.
        </p>
      </header>

      <DomainNotice domain={domain} />

      <form onSubmit={onSubmit}>
        <Card title="1 · Region">
          <RegionPicker
            regions={regions}
            mode={mode}
            setMode={setMode}
            regionId={regionId}
            setRegionId={setRegionId}
            bbox={bbox}
            setBbox={setBbox}
          />
        </Card>

        <Card title="2 · Date windows">
          <DateWindows
            windows={windows}
            setWindows={setWindows}
            training={domain?.training_windows}
          />
        </Card>

        <div className="actions">
          <button type="submit" className="primary" disabled={busy}>
            {busy ? 'Running…' : 'Run inference'}
          </button>
          {submitError && <span className="error-text">{submitError}</span>}
        </div>
      </form>

      {job && <JobProgress job={job} />}
      {job && job.status === 'done' && <Results job={job} />}
      {!job && modelCard && <ModelCard card={modelCard} />}

      <footer className="foot muted small">
        Served checkpoint: {modelCard?.checkpoint || '—'}. Metrics on this page are
        read live from the project’s result JSON; a missing value shows as “—”, never
        a placeholder.
      </footer>
    </div>
  )
}
