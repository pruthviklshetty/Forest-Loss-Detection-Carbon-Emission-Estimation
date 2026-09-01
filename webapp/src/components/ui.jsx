// Small shared presentational pieces.

export function Card({ title, subtitle, children, tone }) {
  return (
    <section className={`card${tone ? ` card--${tone}` : ''}`}>
      {title && <h2 className="card__title">{title}</h2>}
      {subtitle && <p className="card__subtitle">{subtitle}</p>}
      {children}
    </section>
  )
}

export function Stat({ label, value, hint, pending }) {
  return (
    <div className={`stat${pending ? ' stat--pending' : ''}`}>
      <div className="stat__value">{value}</div>
      <div className="stat__label">{label}</div>
      {hint && <div className="stat__hint">{hint}</div>}
    </div>
  )
}

export function Notice({ kind = 'info', title, children }) {
  return (
    <div className={`notice notice--${kind}`}>
      {title && <strong className="notice__title">{title}</strong>}
      <div className="notice__body">{children}</div>
    </div>
  )
}

export function Field({ label, children, hint }) {
  return (
    <label className="field">
      <span className="field__label">{label}</span>
      {children}
      {hint && <span className="field__hint">{hint}</span>}
    </label>
  )
}
