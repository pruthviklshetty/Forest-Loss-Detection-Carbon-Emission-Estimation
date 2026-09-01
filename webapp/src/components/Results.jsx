import { Card, Stat, Notice } from './ui.jsx'
import CloudPanel from './CloudPanel.jsx'
import ModelCard from './ModelCard.jsx'
import { maskUrl } from '../api.js'
import { num, int, ratio, PENDING, isMissing } from '../format.js'

export default function Results({ job }) {
  const r = job?.result
  if (!r) return null
  const ac = r.area_carbon || {}
  const dom = r.domain || {}
  const der = dom.derived
  const card = r.model_card
  const mcase = r.metric_case || 'loro'

  const co2Primary = ac.co2_tonnes_exponential
  const co2Bin = ac.co2_tonnes_three_bin
  const areaBias = card?.area?.pooled_test_pred_over_gfc

  return (
    <div className="results">
      <Card
        title="Prediction"
        subtitle={
          dom.region_name
            ? dom.region_name
            : der
              ? `Point + ${num(der.radius_km, 0)} km radius`
              : `Custom bbox [${(dom.bbox_wsen || []).map((x) => Number(x).toFixed(2)).join(', ')}]`
        }
      >
        <div className="muted small">
          {der && (
            <>
              centre {num(der.center?.[0], 4)}, {num(der.center?.[1], 4)} · snapped to{' '}
              {num(der.derived_side_km, 2)} km ({der.n_tiles_per_side}×{der.n_tiles_per_side}{' '}
              tiles) ·{' '}
            </>
          )}
          {(dom.window_t || []).join(' … ')} → {(dom.window_t1 || []).join(' … ')} ·{' '}
          {isMissing(dom.area_km2) ? PENDING : `${num(dom.area_km2, 1)} km²`} ·{' '}
          {(dom.raster_px || []).join(' × ')} px · operating threshold{' '}
          {isMissing(r.operating_threshold) ? PENDING : num(r.operating_threshold, 2)}
        </div>

        {mcase === 'loro' ? (
          <Notice kind="warn" title="Not in the training set — leave-one-region-out metrics apply">
            This area is not inside one of the four training regions, so the
            model is in its leave-one-region-out regime. Measured mean strict
            IoU there is{' '}
            <b>
              {isMissing(card?.transfer_out_of_training_set?.loro_mean_strict_iou)
                ? PENDING
                : num(card.transfer_out_of_training_set.loro_mean_strict_iou, 3)}
            </b>{' '}
            — roughly half the in-domain{' '}
            {isMissing(card?.in_domain?.strict_iou?.mean)
              ? PENDING
              : num(card.in_domain.strict_iou.mean, 3)}
            . The figures below are for that case.
          </Notice>
        ) : (
          <Notice kind="info" title={`Inside training region: ${r.metric_case_region || '—'}`}>
            This area falls inside a training region, so the pooled in-domain
            metrics apply (strict IoU{' '}
            {isMissing(card?.in_domain?.strict_iou?.mean)
              ? PENDING
              : num(card.in_domain.strict_iou.mean, 3)}
            {!isMissing(card?.in_domain?.strict_iou?.sd) && (
              <> ± {num(card.in_domain.strict_iou.sd, 3)}</>
            )}
            ).
          </Notice>
        )}

        {r.small_area && (
          <Notice kind="warn" title="Small area — result is provisional">
            At the measured ~0.3% loss prevalence, the expected number of true
            loss pixels in an area this size (
            {isMissing(dom.area_km2) ? PENDING : `${num(dom.area_km2, 0)} km²`}, below the{' '}
            {num(r.small_area_threshold_km2, 0)} km² threshold) is very low. A{' '}
            <b>zero result is the expected outcome</b>; a non-zero result should be
            treated as provisional, not a confirmed detection. Check the raw pixel
            count below to see how thin the signal is.
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
            label="Predicted loss — pixel count (10 m)"
            value={isMissing(ac.predicted_loss_pixels) ? PENDING : int(ac.predicted_loss_pixels)}
            pending={isMissing(ac.predicted_loss_pixels)}
            hint="raw count, shown so you can judge how thin the signal is"
          />
          <Stat
            label="Predicted cleared area"
            value={isMissing(ac.predicted_loss_ha) ? PENDING : `${num(ac.predicted_loss_ha, 1)} ha`}
            pending={isMissing(ac.predicted_loss_ha)}
            hint={
              isMissing(areaBias)
                ? 'not calibrated against a reference'
                : `not calibrated: measured pred/GFC ≈ ${ratio(areaBias)} on the held-out test`
            }
          />
          <Stat
            label="Committed CO₂ — regression (primary)"
            value={isMissing(co2Primary) ? PENDING : `${int(co2Primary)} t`}
            pending={isMissing(co2Primary)}
            hint={
              isMissing(ac.co2_tonnes_three_bin)
                ? null
                : `3-bin baseline: ${int(co2Bin)} t · mean AGC ${num(ac.mean_agc_tC_ha, 0)} tC/ha`
            }
          />
        </div>
        <p className="muted small">{ac.co2_scope}</p>
        {!isMissing(areaBias) && (
          <p className="muted small">
            Area calibration: on the pooled held-out test the model’s predicted
            cleared area is <b>{ratio(areaBias)}</b> the Hansen-GFC reference (this
            checkpoint over-predicts; the Phase 7 checkpoint under-predicted). The
            hectares above are not a calibrated measurement.
          </p>
        )}
      </Card>

      <CloudPanel cloud={r.cloud} />

      <ModelCard card={card} forResult metricCase={mcase} />
    </div>
  )
}
