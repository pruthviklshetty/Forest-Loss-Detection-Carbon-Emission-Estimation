import { useEffect, useRef, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ReferenceLine,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { Notice } from './ui.jsx'
import { interval, num, PENDING, isMissing } from '../format.js'

// The model's own scorecard, read live from results/**.json via the backend.
// Collapsed by default: the one-line summary carries the two headline IoUs;
// the full table, notes, chart and findings live inside the expander.
// Nothing here is hard-coded; a missing field renders as PENDING.
const fmtPeriod = (p) => (p ? p.replace('_', '→') : null)

// Finite number or null - guards Math.max()/recharts against non-finite or
// missing values producing a degenerate axis (e.g. Infinity/NaN collapsing to
// a huge bogus "nice" tick).
const finite = (x) => (typeof x === 'number' && Number.isFinite(x) ? x : null)

export default function ModelCard({ card, forResult, metricCase }) {
  // Native <details> hides its body with display:none when closed. Recharts'
  // ResponsiveContainer measures its container synchronously on mount; if that
  // happens before the browser has finished laying out the just-opened
  // <details> (the toggle click and React's re-render can outrace layout), it
  // measures 0x0 and recharts locks in a degenerate axis scale (a huge bogus
  // tick) that a later resize does not reliably correct. So: track `open` for
  // the UI, but only mount the chart on `chartReady`, set one paint later via
  // double requestAnimationFrame (the standard "wait for layout" technique) -
  // and remount it fresh (via `chartKey`) every time, so no run can inherit a
  // previous bad measurement.
  const [open, setOpen] = useState(false)
  const [chartReady, setChartReady] = useState(false)
  const [chartKey, setChartKey] = useState(0)
  const rafIds = useRef([])

  useEffect(() => () => rafIds.current.forEach((id) => cancelAnimationFrame(id)), [])

  const handleToggle = (e) => {
    const isOpen = e.currentTarget.open
    setOpen(isOpen)
    rafIds.current.forEach((id) => cancelAnimationFrame(id))
    rafIds.current = []
    if (isOpen) {
      const id1 = requestAnimationFrame(() => {
        const id2 = requestAnimationFrame(() => {
          setChartKey((k) => k + 1)
          setChartReady(true)
        })
        rafIds.current = [id2]
      })
      rafIds.current = [id1]
    } else {
      setChartReady(false)
    }
  }

  if (!card) return null
  const d = card.in_domain || {}
  const t = card.transfer_out_of_training_set || {}
  const md = card.more_data_finding // may be null - do not coerce to {}
  const loroApplies = metricCase === 'loro'

  // LORO was either measured for this checkpoint's period, or carried from the
  // 2019→2021 model and labelled as such - never a null implying a measurement.
  const loroMeasured = t.measured !== false
  const loroPeriod = fmtPeriod(t.loro_period)

  const rows = [
    ['IoU (strict, primary)', d.strict_iou],
    ['IoU (±3 px tolerance)', d.tolerance_iou],
    ['Dice', d.dice],
    ['Precision', d.precision],
    ['Recall', d.recall],
  ]

  const folds = (t.folds || []).map((f) => ({
    name: f.test_region ?? '?',
    iou: isMissing(f.strict_iou) ? 0 : f.strict_iou,
    missing: isMissing(f.strict_iou),
  }))
  const inDomainMean = finite(t.loro_in_domain_strict_iou_mean)
  const loroInterval = {
    mean: t.loro_mean_strict_iou,
    sd: t.loro_sd_strict_iou,
  }
  // Every candidate coerced to a finite number (or dropped) before Math.max,
  // so a stray null/NaN/Infinity can never produce a runaway axis top. IoU is
  // bounded [0, 1] regardless of what the data says.
  const yCandidates = [inDomainMean, ...folds.map((f) => finite(f.iou)), 0.05]
    .filter((v) => v !== null)
  const yMax = Math.min(Math.max(...yCandidates, 0.05) * 1.15, 1)

  return (
    <details className="card modelcard" onToggle={handleToggle}>
      <summary className="modelcard__summary">
        <span className="modelcard__toggle">Model performance details</span>
        <span className="modelcard__oneline">
          in-domain strict IoU <b>{interval(d.strict_iou)}</b>, out-of-region{' '}
          <b>{interval(loroInterval)}</b>
          {!loroMeasured && loroPeriod && (
            <> (measured on {loroPeriod}, not re-measured here)</>
          )}
        </span>
      </summary>

      <div className="modelcard__body">
        {forResult && (
          <p className="muted small">
            These are the model’s held-out test scores, not this run’s accuracy —
            there is no ground truth for a live request.
          </p>
        )}
        {loroApplies && forResult && (
          <p className="muted small">
            The pooled numbers in this table are <b>not</b> the applicable case for
            your result — that area is outside the training set, so the
            leave-one-region-out figures below apply.
          </p>
        )}

        <table className="metrics">
          <thead>
            <tr>
              <th>Metric (pooled 4-region held-out test)</th>
              <th>mean ± sd over {d.seeds ? d.seeds.join('/') : '3'} seeds</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, obj]) => (
              <tr key={label} className={isMissing(obj?.mean) ? 'pending' : ''}>
                <td>{label}</td>
                <td>{interval(obj)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">{d.note}</p>

        <Notice kind="warn" title="Performance outside the training set is lower">
          {loroMeasured ? (
            <p>
              Leave-one-region-out (train on 3 Western Ghats regions, test on the
              4th{loroPeriod ? `, ${loroPeriod}` : ''}): mean strict IoU{' '}
              <b>{isMissing(t.loro_mean_strict_iou) ? PENDING : num(t.loro_mean_strict_iou, 3)}</b>
              {!isMissing(t.loro_sd_strict_iou) && <> ± {num(t.loro_sd_strict_iou, 3)}</>} versus{' '}
              <b>{isMissing(inDomainMean) ? PENDING : num(inDomainMean, 3)}</b> in-domain —
              roughly half. Any result for a custom bbox or a non-training preset
              should be read as limited by this gap.
            </p>
          ) : (
            <p>{t.note}</p>
          )}
          {folds.length > 0 && !chartReady && (
            <p className="muted small">
              {open ? '(rendering chart…)' : '(chart shown once this card is expanded)'}
            </p>
          )}
          {folds.length > 0 && chartReady && (
            <div className="chart">
              <ResponsiveContainer key={chartKey} width="100%" height={180} minWidth={220} debounce={1}>
                <BarChart data={folds} margin={{ top: 12, right: 8, bottom: 4, left: 0 }}>
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} domain={[0, yMax]} allowDataOverflow />
                  <Tooltip formatter={(v, _n, p) => (p.payload.missing ? PENDING : num(v, 3))} />
                  {inDomainMean !== null && (
                    <ReferenceLine
                      y={inDomainMean}
                      stroke="#2563eb"
                      strokeWidth={1.5}
                      strokeDasharray="4 3"
                      label={{
                        value: `in-domain ${num(inDomainMean, 3)}`,
                        position: 'insideTopLeft',
                        fontSize: 10,
                        fill: '#2563eb',
                      }}
                    />
                  )}
                  <Bar dataKey="iou" radius={[3, 3, 0, 0]}>
                    {folds.map((f, i) => (
                      <Cell key={i} fill={f.missing ? '#cbd5e1' : '#d97706'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="muted small">
                Held-out-region strict IoU per fold vs the in-domain mean (blue line)
                {!loroMeasured && loroPeriod ? `, measured on the ${loroPeriod} model` : ''}.
              </div>
            </div>
          )}
        </Notice>

        <div className="muted small">
          {md && (
            <p>
              <b>
                More data did not raise the ceiling
                {md.period ? ` (${fmtPeriod(md.period)})` : ''}.
              </b>{' '}
              {isMissing(md.pooled_iou_mean)
                ? PENDING
                : `Pooled 4-region IoU ${num(md.pooled_iou_mean, 3)}`}
              {md.wayanad_only_iou && !isMissing(md.wayanad_only_iou.mean) && (
                <> vs single-region {num(md.wayanad_only_iou.mean, 3)} ± {num(md.wayanad_only_iou.sd, 3)}</>
              )}
              {!isMissing(md.delta_vs_wayanad_only) && (
                <>
                  {' '}(Δ {md.delta_vs_wayanad_only > 0 ? '+' : ''}
                  {num(md.delta_vs_wayanad_only, 3)},{' '}
                  {md.within_seed_variance ? 'within' : 'outside'} seed variance)
                </>
              )}
              .
            </p>
          )}
          <p>{card.label_resolution_note}</p>
        </div>
      </div>
    </details>
  )
}
