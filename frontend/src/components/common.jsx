// Shared layout bits used by every section.

export const FIG_BASE = 'figures/'

export function Section({ n, title, sub, children }) {
  return (
    <section id={`s${n}`}>
      <div className="sec-head">
        <span className="sec-num">{n}</span>
        <h2>{title}</h2>
      </div>
      {sub && <p className="sec-sub">{sub}</p>}
      {children}
    </section>
  )
}

export function Figure({ fig }) {
  if (!fig) return null
  return (
    <figure>
      <img src={FIG_BASE + fig.public_name} alt={fig.caption} loading="lazy" />
      <figcaption>{fig.caption}</figcaption>
    </figure>
  )
}
