import { Card, Notice } from './ui.jsx'
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
        title="Results"
        subtitle={
          dom.region_name
            ? dom.region_name
            : der
              ? `Point + ${num(der.radius_km, 0)} km radius`
              : `Custom bbox [${(dom.bbox_wsen || []).map((x) => Number(x).toFixed(2)).join(', ')}]`
        }
      >
        {/* headline: the two figures are the largest, highest-contrast elements */}
        <div className="headline">
          <div className="headline__stat">
            <div className="headline__value">
              {isMissing(ac.predicted_loss_ha) ? PENDING : `${num(ac.predicted_loss_ha, 1)} ha`}
            </div>
            <div className="headline__label">predicted cleared area</div>
          </div>
          <div className="headline__stat">
            <div className="headline__value">
              {isMissing(co2Primary) ? PENDING : `${int(co2Primary)} t`}
            </div>
            <div className="headline__label">committed CO₂ — regression (primary)</div>
          </div>
        </div>

        {/* prediction overlay, directly beneath the figures */}
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

        {/* --- caveats: kept in full, reduced visual weight --- */}

        {mcase === 'loro' ? (
          <Notice kind="warn" title="Not in the training set — leave-one-region-out metrics apply">
            This area is not inside one of the four training regions, so the model
            is in its leave-one-region-out regime. Measured mean strict IoU there
            is{' '}
            <b>
              {isMissing(card?.transfer_out_of_training_set?.loro_mean_strict_iou)
                ? PENDING
                : num(card.transfer_out_of_training_set.loro_mean_strict_iou, 3)}
            </b>{' '}
            — roughly half the in-domain{' '}
            {isMissing(card?.in_domain?.strict_iou?.mean)
              ? PENDING
              : num(card.in_domain.strict_iou.mean, 3)}
            .
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
            At the measured ~0.3% loss prevalence, the expected number of true loss
            pixels in an area this size (
            {isMissing(dom.area_km2) ? PENDING : `${num(dom.area_km2, 0)} km²`}, below the{' '}
            {num(r.small_area_threshold_km2, 0)} km² threshold) is very low. A{' '}
            <b>zero result is the expected outcome</b>; a non-zero result should be
            treated as provisional, not a confirmed detection.
          </Notice>
        )}

        {/* item-5: raw pixel count and the area-calibration note stay inline
            and legible - reduced weight vs the headline, not collapsed */}
        <div className="result-facts">
          <div>
            <span className="result-facts__k">Raw predicted loss pixels (10 m)</span>
            <span className="result-facts__v">
              {isMissing(ac.predicted_loss_pixels) ? PENDING : int(ac.predicted_loss_pixels)}
            </span>
            <span className="muted small">how thin the signal is, before conversion to hectares</span>
          </div>
          <div>
            <span className="result-facts__k">Area calibration</span>
            <span className="result-facts__v">
              {isMissing(areaBias) ? 'not calibrated' : `pred/GFC ≈ ${ratio(areaBias)}`}
            </span>
            <span className="muted small">
              {isMissing(areaBias)
                ? 'the predicted hectares are not calibrated against a reference.'
                : 'on the pooled held-out test the predicted cleared area is this multiple of the Hansen-GFC reference (this checkpoint over-predicts; the Phase 7 checkpoint under-predicted). The hectares above are not a calibrated measurement.'}
            </span>
          </div>
        </div>

        <dl className="secondary">
          <div>
            <dt>CO₂ — other estimate &amp; scope</dt>
            <dd className="muted small">
              3-bin baseline {isMissing(co2Bin) ? PENDING : `${int(co2Bin)} t`}
              {!isMissing(ac.mean_agc_tC_ha) && <> · mean AGC {num(ac.mean_agc_tC_ha, 0)} tC/ha</>}
              . {ac.co2_scope}
            </dd>
          </div>
          <div>
            <dt>Run</dt>
            <dd className="muted small">
              {der && (
                <>
                  centre {num(der.center?.[0], 4)}, {num(der.center?.[1], 4)} · snapped to{' '}
                  {num(der.derived_side_km, 2)} km ({der.n_tiles_per_side}×
                  {der.n_tiles_per_side} tiles) ·{' '}
                </>
              )}
              {(dom.window_t || []).join(' … ')} → {(dom.window_t1 || []).join(' … ')} ·{' '}
              {isMissing(dom.area_km2) ? PENDING : `${num(dom.area_km2, 1)} km²`} ·{' '}
              {(dom.raster_px || []).join(' × ')} px · operating threshold{' '}
              {isMissing(r.operating_threshold) ? PENDING : num(r.operating_threshold, 2)}
            </dd>
          </div>
        </dl>
      </Card>

      <CloudPanel cloud={r.cloud} />

      <ModelCard card={card} forResult metricCase={mcase} />
    </div>
  )
}
