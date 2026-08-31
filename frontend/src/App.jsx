import { content, areaSummary } from './data.js'
import { f1, f2, int } from './format.js'
import { Section, Figure } from './components/common.jsx'
import Segmentation from './components/Segmentation.jsx'
import Corrections from './components/Corrections.jsx'
import Carbon from './components/Carbon.jsx'

function Header() {
  const h = content.header
  return (
    <header className="site-head">
      <h1>{h.title}</h1>
      <div className="meta">
        <span><b>Region</b> {h.region}</span>
        <span><b>Window</b> {h.window}</span>
        <span><b>AOI</b> ~{int(h.area_km2)} km² · {h.gsd_m} m GSD</span>
        <span><b>BBox</b> {h.bbox_wsen.join(', ')}</span>
      </div>
      <p className="positioning">{h.positioning}</p>
      <p className="note" style={{ marginTop: 10, fontSize: 12.5 }}>
        Static results viewer — reads pre-computed metrics and pre-generated
        figures from the project's <code>results/</code> directory. It does not
        accept uploads, run the model, or call any backend.
      </p>
    </header>
  )
}

function Qualitative() {
  const fig = content.figures.find((f) => f.key === 'triptych')
  return (
    <Section
      n="4"
      title="Qualitative results"
      sub="Held-out test patches. Each row: Sentinel-2 T+1 false colour, Hansen GFC ground truth, predicted probability. More than five representative patches are shown."
    >
      {fig ? (
        <Figure fig={fig} />
      ) : (
        <div className="pending">Triptych figure pending — not found in results/figures/.</div>
      )}
    </Section>
  )
}

function ChangeDetection() {
  const t = areaSummary.regions.test_only
  const full = areaSummary.regions.full_region
  const mapFig = content.figures.find((f) => f.key === 'deforestation_map')
  const haFig = content.figures.find((f) => f.key === 'hectares')
  return (
    <Section n="5" title="Change detection & area" sub={content.change_detection_note}>
      <div className="compare">
        <div>
          <div className="lbl">Predicted (held-out test region)</div>
          <div className="big">{f1(t.pred_ha)} ha</div>
        </div>
        <div>
          <div className="lbl">Hansen GFC reference (same region)</div>
          <div className="big">{f1(t.gt_ha)} ha</div>
        </div>
        <div className="ratio">
          <div className="lbl">Predicted / reference</div>
          <div className="big">{f2(t.pred_over_gt_ratio)}×</div>
        </div>
      </div>
      <p className="note">
        Pixel counts converted at {areaSummary.gsd_m} m GSD → {areaSummary.ha_per_pixel} ha
        per pixel. Full-region figure (includes pixels the model trained on):
        predicted {f1(full.pred_ha)} ha vs GFC {f1(full.gt_ha)} ha
        ({f2(full.pred_over_gt_ratio)}×). The predicted hectares are not a
        validated area.
      </p>
      <div className="fig-grid">
        <Figure fig={mapFig} />
        <Figure fig={haFig} />
      </div>
    </Section>
  )
}

function Limitations() {
  return (
    <Section
      n="7"
      title="Limitations & scope"
      sub="Deliberate scope choices and the constraints they impose. Load-bearing for how the numbers above should be read."
    >
      <ul className="plain">
        {content.limitations.map((l, i) => (
          <li key={i}>{l}</li>
        ))}
      </ul>
    </Section>
  )
}

function Footer() {
  const l = content.links
  return (
    <footer className="site-foot wrap">
      <p>
        Source &amp; full write-up:{' '}
        <a href={l.repo} target="_blank" rel="noreferrer">
          {l.repo.replace('https://', '')}
        </a>
      </p>
      <p className="note" style={{ fontSize: 12 }}>{l.report_note}</p>
      <details style={{ marginTop: 12 }}>
        <summary style={{ cursor: 'pointer', fontSize: 12.5, color: 'var(--ink-3)' }}>
          References ({content.references.length}) — {content.references_note}
        </summary>
        <ol className="refs" style={{ marginTop: 8 }}>
          {content.references.map((r) => (
            <li key={r.key}>{r.cite}</li>
          ))}
        </ol>
      </details>
    </footer>
  )
}

export default function App() {
  return (
    <>
      <div className="wrap">
        <Header />
        <Segmentation />
      </div>

      <div className="corrections">
        <div className="wrap">
          <div className="wrap-inner">
            <Corrections />
          </div>
        </div>
      </div>

      <div className="wrap">
        <Qualitative />
        <ChangeDetection />
        <Carbon />
        <Limitations />
      </div>

      <Footer />
    </>
  )
}
