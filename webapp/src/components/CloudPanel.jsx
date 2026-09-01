import { Card, Notice } from './ui.jsx'
import { pct, int, PENDING } from '../format.js'

// Cloud / no-data cover for both composites, always shown. When either window
// is flagged (high cover or few scenes) the warning is prominent, not a
// footnote.
export default function CloudPanel({ cloud }) {
  if (!cloud) return null
  const win = (w, label) => {
    const d = cloud[w] || {}
    const flagged = d.high_cloud || d.few_scenes
    return (
      <div className={`cloud-win${flagged ? ' cloud-win--flag' : ''}`} key={w}>
        <div className="cloud-win__head">
          <b>{label}</b>
          <span className="muted">{(d.dates || []).join(' … ') || PENDING}</span>
        </div>
        <div className="cloud-win__row">
          <span>cloud / no-data cover</span>
          <span>{pct(d.cloud_or_nodata_cover_pct)}</span>
        </div>
        <div className="cloud-win__row">
          <span>clear Sentinel-2 scenes</span>
          <span>{int(d.n_scenes)}</span>
        </div>
        {d.high_cloud && (
          <div className="cloud-win__flag">
            High cloud / no-data cover (&gt; {pct(cloud.flag_threshold_pct, 0)}). The
            composite has gaps; the mask under them is unreliable.
          </div>
        )}
        {d.few_scenes && (
          <div className="cloud-win__flag">
            Only {int(d.n_scenes)} clear scenes (&lt; {cloud.min_scenes}). The
            median composite is thin for this window.
          </div>
        )}
      </div>
    )
  }

  return (
    <Card title="Input quality — cloud &amp; scene coverage">
      {cloud.any_flag && (
        <Notice kind="warn" title="Composite quality flag">
          At least one date window has high cloud / no-data cover or too few
          scenes. Read the hectares and CO₂ below as lower-confidence.
        </Notice>
      )}
      <div className="cloud-grid">
        {win('window_t', 'Earlier (T)')}
        {win('window_t1', 'Later (T+1)')}
      </div>
    </Card>
  )
}
