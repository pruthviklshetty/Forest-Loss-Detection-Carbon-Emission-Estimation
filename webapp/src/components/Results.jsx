import { Card, Stat, Notice } from './ui.jsx'
import CloudPanel from './CloudPanel.jsx'
import ModelCard from './ModelCard.jsx'
import { maskUrl } from '../api.js'
import { num, int, PENDING, isMissing } from '../format.js'

export default function Results({ job }) {
  const r = job?.result
  if (!r) return null
  const ac = r.area_carbon || {}
  const dom = r.domain || {}
  const card = r.model_card

  const co2Primary = ac.co2_tonnes_exponential
  const co2Bin = ac.co2_tonnes_three_bin

  return (
    <div className="results">
      <Card
        title="Prediction"
        subtitle={
          dom.region_name
            ? `${dom.region_name}`
            : `Custom bbox [${(dom.bbox_wsen || []).map((x) => Number(x).toFixed(2)).join(', ')}]`
        }
      >
        <div className="muted small">
          {(dom.window_t || []).join(' … ')} → {(dom.window_t1 || []).join(' … ')} ·{' '}
          {isMissing(dom.area_km2) ? PENDING : `${num(dom.area_km2, 1)} km²`} ·{' '}
          {(dom.raster_px || []).join(' × ')} px · operating threshold{' '}
          {isMissing(r.operating_threshold) ? PENDING : num(r.operating_threshold, 2)}
        </div>

        {dom.in_training_set === false && (
          <Notice kind="warn" title="Outside the training set">
            {dom.region_id
              ? 'This preset region is inside the model domain but was not used for training.'
              : 'Custom bounding boxes are never part of the training set.'}{' '}
            The model card below shows how much lower out-of-training-set accuracy
            is measured to be.
          </Notice>
        )}

        <div className="mask-wrap">
          {r.mask_ready ? (
            <img
              className="mask-img"
              src={maskUrl(job.id)}
              alt="Predicted forest-loss overlay (red) on the later false-colour composite"
            />
          ) : (
            <div className="mask-pending">mask not available for this job</div>
          )}
          <div className="muted small">
            Red = predicted new forest loss between the two dates, on the later
            Sentinel-2 false-colour composite.
          </div>
        </div>

        <div className="stat-row">
          <Stat
            label="Cleared area (predicted)"
            value={isMissing(ac.predicted_loss_ha) ? PENDING : `${num(ac.predicted_loss_ha, 1)} ha`}
            pending={isMissing(ac.predicted_loss_ha)}
            hint={isMissing(ac.predicted_loss_pixels) ? null : `${int(ac.predicted_loss_pixels)} px @ ${ac.gsd_m ?? 10} m`}
          />
          <Stat
            label="Committed CO₂ — regression (primary)"
            value={isMissing(co2Primary) ? PENDING : `${int(co2Primary)} t`}
            pending={isMissing(co2Primary)}
            hint={isMissing(ac.mean_agc_tC_ha) ? null : `mean AGC ${num(ac.mean_agc_tC_ha, 0)} tC/ha`}
          />
          <Stat
            label="Committed CO₂ — 3-bin baseline"
            value={isMissing(co2Bin) ? PENDING : `${int(co2Bin)} t`}
            pending={isMissing(co2Bin)}
            hint="coarse NDVI-tercile scheme, shown for comparison"
          />
        </div>
        <p className="muted small">{ac.co2_scope}</p>
      </Card>

      <CloudPanel cloud={r.cloud} />

      <ModelCard card={card} forResult />
    </div>
  )
}
