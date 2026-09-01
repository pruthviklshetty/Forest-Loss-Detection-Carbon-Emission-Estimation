import { Field } from './ui.jsx'

// Two date windows. Defaults come from the backend's training_windows so the
// first run matches the training composites exactly. Inputs are limited to
// Jan-Apr; the backend enforces this again.
export default function DateWindows({ windows, setWindows, training }) {
  const set = (key, idx, val) => {
    const next = { ...windows, [key]: windows[key].slice() }
    next[key][idx] = val
    setWindows(next)
  }

  const row = (key, label) => (
    <div className="win-row">
      <span className="win-row__label">{label}</span>
      <Field label="start">
        <input type="date" value={windows[key][0]} onChange={(e) => set(key, 0, e.target.value)} />
      </Field>
      <Field label="end">
        <input type="date" value={windows[key][1]} onChange={(e) => set(key, 1, e.target.value)} />
      </Field>
    </div>
  )

  return (
    <div className="windows">
      {row('window_t', 'Earlier (T)')}
      {row('window_t1', 'Later (T+1)')}
      <p className="muted small">
        January–April only. Training used{' '}
        {training?.T ? `${training.T.start} … ${training.T.end}` : 'Jan–mid-Apr'} and{' '}
        {training?.T_plus_1
          ? `${training.T_plus_1.start} … ${training.T_plus_1.end}`
          : 'the same window two years later'}
        .
      </p>
    </div>
  )
}
