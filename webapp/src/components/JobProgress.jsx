import { Card } from './ui.jsx'

const STEPS = [
  ['queued', 'Queued'],
  ['fetching', 'Fetching Sentinel-2 from Earth Engine'],
  ['inferring', 'Running the model'],
  ['estimating', 'Area + CO₂'],
  ['done', 'Done'],
]

export default function JobProgress({ job }) {
  if (!job) return null
  const pctVal = Math.round((job.progress ?? 0) * 100)
  const curIdx = STEPS.findIndex((s) => s[0] === job.status)

  return (
    <Card title="Job status" tone={job.status === 'failed' ? 'danger' : undefined}>
      <div className="progress">
        <div className="progress__bar" style={{ width: `${pctVal}%` }} />
      </div>
      <div className="progress__meta">
        <span className="badge">{job.status}</span>
        <span>{pctVal}%</span>
        <span className="muted">{job.message}</span>
      </div>
      <ol className="steps">
        {STEPS.map(([key, label], i) => (
          <li
            key={key}
            className={
              job.status === 'failed'
                ? i <= curIdx
                  ? 'step step--failed'
                  : 'step'
                : i < curIdx || job.status === 'done'
                  ? 'step step--done'
                  : i === curIdx
                    ? 'step step--active'
                    : 'step'
            }
          >
            {label}
          </li>
        ))}
      </ol>
      {job.status === 'failed' && (
        <p className="error-text">{job.error || 'Job failed.'}</p>
      )}
      {job.status !== 'failed' && job.status !== 'done' && (
        <p className="muted small">
          A small bbox (~100–200 km²) finishes in 2–4 min; a full preset region
          (~850 km²) is a 15–25 min job — two Earth Engine pulls plus tiled
          inference. This page polls every 2 s.
        </p>
      )}
    </Card>
  )
}
