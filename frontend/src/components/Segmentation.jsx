import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ErrorBar, ResponsiveContainer,
} from 'recharts'
import { segBaseline, segAttention, content } from '../data.js'
import { meanSd } from '../format.js'
import { Section } from './common.jsx'

const CX = content.segmentation_context

// metric cards: strict IoU primary, tolerance IoU right next to it (secondary),
// then Dice / precision / recall.
const CARDS = [
  { key: 'test_iou', label: 'Strict pixel IoU', cls: 'primary', sub: 'primary metric' },
  { key: 'test_tolerance_iou', label: 'Tolerance IoU (±3 px)', cls: 'secondary', sub: 'secondary — GT dilated 1 GFC cell' },
  { key: 'test_dice', label: 'Dice / F1', cls: '', sub: null },
  { key: 'test_precision', label: 'Precision', cls: '', sub: null },
  { key: 'test_recall', label: 'Recall', cls: '', sub: null },
]

const CHART_ROWS = [
  ['Strict IoU', 'test_iou'],
  ['Tolerance IoU', 'test_tolerance_iou'],
  ['Dice', 'test_dice'],
  ['Precision', 'test_precision'],
  ['Recall', 'test_recall'],
].map(([name, k]) => ({ name, mean: segBaseline[k].mean, sd: segBaseline[k].sd }))

export default function Segmentation() {
  return (
    <Section
      n="2"
      title="Segmentation results — plain U-Net"
      sub={`Real measured numbers on the ${CX.test_set}. Every value is the mean ± standard deviation across 3 training seeds (42, 43, 44); the sd is shown because run-to-run spread is comparable to the quantities themselves.`}
    >
      <div className="metric-grid">
        {CARDS.map((c) => {
          const m = segBaseline[c.key]
          return (
            <div className={`metric ${c.cls}`} key={c.key}>
              <div className="k">{c.label}</div>
              <div className="v">
                {m.mean.toFixed(3)} <span className="sd">±{m.sd.toFixed(3)}</span>
              </div>
              <div className="sub">{c.sub || `seeds: ${m.values.map((v) => v.toFixed(3)).join(' / ')}`}</div>
            </div>
          )
        })}
      </div>

      <div className="chart">
        <ResponsiveContainer width="100%" height={230}>
          <BarChart
            data={CHART_ROWS}
            layout="vertical"
            margin={{ top: 4, right: 24, bottom: 4, left: 8 }}
          >
            <CartesianGrid horizontal={false} stroke="#e2ded4" />
            <XAxis
              type="number"
              domain={[0, 0.4]}
              tick={{ fontSize: 11, fill: '#726d63' }}
              tickFormatter={(v) => v.toFixed(1)}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={96}
              tick={{ fontSize: 11, fill: '#4a4740' }}
            />
            <Tooltip
              cursor={{ fill: '#f4f2ec' }}
              formatter={(v, _n, p) => [`${v.toFixed(3)} ±${p.payload.sd.toFixed(3)}`, 'mean ± sd']}
            />
            <Bar dataKey="mean" fill="#4a4740" barSize={16} isAnimationActive={false}>
              <ErrorBar dataKey="sd" width={5} strokeWidth={1.2} stroke="#9c3b1b" direction="x" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="chart-note">
          Bars: 3-seed mean. Whiskers: ± 1 standard deviation across seeds.
          Held-out test split; operating threshold tuned on validation.
        </div>
      </div>

      <p className="note">{CX.iou_note}</p>

      <div className="callout">
        <b>Model selection.</b> {CX.model_selection_rule} The carry-forward
        checkpoint is {CX.carry_forward_checkpoint}.
      </div>

      <p className="note">{CX.attention_note} For the record, its 3-seed test IoU
        is {meanSd(segAttention.test_iou)} (strict) / {meanSd(segAttention.test_tolerance_iou)} (tolerance);
        per-seed values for both models are in <code>results/metrics/seed_runs.json</code>.
      </p>
    </Section>
  )
}
