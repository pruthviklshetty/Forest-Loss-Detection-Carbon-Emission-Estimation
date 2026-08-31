import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { carbon, content } from '../data.js'
import { f1, kt, int } from '../format.js'
import { Section, Figure } from './common.jsx'

const E = carbon.estimates
const GFW = content.carbon_reference_gfw
const calFig = content.figures.find((f) => f.key === 'carbon_calibration')

const SETS = [
  ['predicted_test', 'Predicted — test region'],
  ['gfc_test', 'Hansen GFC reference — test region'],
  ['predicted_full_region', 'Predicted — full region'],
  ['gfc_full_region', 'Hansen GFC reference — full region'],
]

const chartData = [
  ['predicted_test', 'Predicted (test)'],
  ['gfc_test', 'GFC reference (test)'],
].map(([k, name]) => ({
  name,
  '3-bin baseline': Math.round(E[k].three_bin_baseline.tCO2),
  'Regression (linear)': Math.round(E[k].regression_linear.tCO2),
  'Regression (exp, primary)': Math.round(E[k].regression_exponential_primary.tCO2),
}))

export default function Carbon() {
  return (
    <Section
      n="6"
      title="Carbon estimation"
      sub="Each cleared pixel's pre-clearing NDVI is mapped to an aboveground carbon density and summed over the deforested area, then converted to CO2. Three mappings are compared, with a published external reference alongside."
    >
      <div className="callout">
        <b>Calibration.</b> {content.carbon_calibration_label}
      </div>

      <div className="tbl-scroll">
        <table>
          <thead>
            <tr>
              <th>Pixel set</th>
              <th className="num">Area (ha)</th>
              <th className="num">3-bin baseline</th>
              <th className="num">Regression — linear</th>
              <th className="num">Regression — exp (primary)</th>
            </tr>
          </thead>
          <tbody>
            {SETS.map(([k, label]) => (
              <tr key={k}>
                <td>{label}</td>
                <td className="num">{f1(E[k].area_ha)}</td>
                <td className="num">{kt(E[k].three_bin_baseline.tCO2)}</td>
                <td className="num">{kt(E[k].regression_linear.tCO2)}</td>
                <td className="num">{kt(E[k].regression_exponential_primary.tCO2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="chart">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid vertical={false} stroke="#e2ded4" />
            <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#4a4740' }} />
            <YAxis
              tick={{ fontSize: 11, fill: '#726d63' }}
              tickFormatter={(v) => `${Math.round(v / 1000)}k`}
              label={{ value: 't CO₂', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#726d63' }}
            />
            <Tooltip formatter={(v) => `${int(v)} t CO₂`} cursor={{ fill: '#f4f2ec' }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="3-bin baseline" fill="#c9c2b0" isAnimationActive={false} />
            <Bar dataKey="Regression (linear)" fill="#8a8375" isAnimationActive={false} />
            <Bar dataKey="Regression (exp, primary)" fill="#4a4740" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
        <div className="chart-note">
          The regression (primary = exponential) is the pipeline's carbon output;
          the 3-bin scheme is the coarse baseline it replaces.
        </div>
      </div>

      {calFig && <Figure fig={calFig} />}

      <h3 style={{ fontSize: 14, margin: '20px 0 6px', fontWeight: 640 }}>
        vs. published reference — Global Forest Watch, Wayanad district
      </h3>
      <div className="compare">
        <div>
          <div className="lbl">This study (GFC reference area)</div>
          <div className="big">~{int(GFW.study_emission_factor_tCO2_per_ha)} t CO₂/ha</div>
          <div className="lbl" style={{ marginTop: 6, textTransform: 'none' }}>
            aboveground carbon, CO₂ only
          </div>
        </div>
        <div>
          <div className="lbl">GFW ({GFW.period})</div>
          <div className="big">~{int(GFW.implied_emission_factor_tCO2e_per_ha)} t CO₂e/ha</div>
          <div className="lbl" style={{ marginTop: 6, textTransform: 'none' }}>
            {GFW.tree_cover_loss_kha} kha · {GFW.co2e_Mt} Mt CO₂e · all pools, all gases
          </div>
        </div>
        <div className="ratio">
          <div className="lbl">study / GFW</div>
          <div className="big">~{Math.round(GFW.study_over_gfw_ratio * 100)}%</div>
        </div>
      </div>
      <p className="note">
        <b>The ~40% difference is scope, not error.</b> {GFW.scope_difference}
      </p>
      <p className="note" style={{ fontSize: 12 }}>
        Reference: {GFW.source}. Derivation: {GFW.implied_emission_factor_derivation}
      </p>
    </Section>
  )
}
