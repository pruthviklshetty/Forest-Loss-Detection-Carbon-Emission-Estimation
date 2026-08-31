import { content, seedRuns, segBaseline, areaSummary, carbon } from '../data.js'
import { meanSd, f1, f2, f3, kt } from '../format.js'
import { Section } from './common.jsx'

const LA = content.leakage_audit
const B = LA.metrics_before

// "after" column — read live, never from content.json
const after = {
  segStrictIoU: meanSd(segBaseline.test_iou),
  segTolIoU: meanSd(segBaseline.test_tolerance_iou),
  segDice: meanSd(segBaseline.test_dice),
  areaPred: areaSummary.regions.test_only.pred_ha,
  areaRef: areaSummary.regions.test_only.gt_ha,
  areaRatio: areaSummary.regions.test_only.pred_over_gt_ratio,
  co2: carbon.estimates.predicted_test.regression_exponential_primary.tCO2,
}

function Row({ label, before, after }) {
  return (
    <tr>
      <td>{label}</td>
      <td className="num">{before}</td>
      <td className="num">{after}</td>
    </tr>
  )
}

export default function Corrections() {
  return (
    <Section
      n="3"
      title={<>Methodological corrections</>}
      sub="The credibility of this project rests here. A self-run leakage audit found and fixed contaminated splits; run-to-run variance was quantified rather than ignored. The corrected numbers are lower than the first draft — that is the point."
    >
      {/* ---- 3a. leakage audit ---- */}
      <h3>
        Data-leakage audit
        <span className="badge">50% of val &amp; test contaminated</span>
      </h3>
      <p className="note">{LA.mechanism}</p>
      <p className="note">
        <b>Found by</b> {LA.found_by}
      </p>
      <p className="note">
        <b>Fix</b> {LA.fix}
      </p>

      <div className="tbl-scroll">
        <table>
          <thead>
            <tr>
              <th>Quantity</th>
              <th className="num">Pre-audit — 1 run (c9947eb)</th>
              <th className="num">Post-audit — 3-seed mean ± sd</th>
            </tr>
          </thead>
          <tbody>
            <Row
              label="Training patches"
              before={`${LA.before.train_patches} (${LA.before.train_canonical} + ${LA.before.train_overlap_crops} overlap)`}
              after={`${LA.after.train_patches} (${LA.after.train_canonical} + ${LA.after.train_overlap_crops} overlap)`}
            />
            <Row
              label="Val / test patches containing training pixels"
              before={`${LA.before.val_patches_contaminated}/${LA.before.val_patches_total}  ·  ${LA.before.test_patches_contaminated}/${LA.before.test_patches_total}`}
              after={`${LA.after.val_patches_contaminated}/${LA.before.val_patches_total}  ·  ${LA.after.test_patches_contaminated}/${LA.before.test_patches_total}`}
            />
            <Row
              label="verify_no_leakage.py exit code"
              before={<span className="pill bad">{LA.before.verify_script_exit_code}</span>}
              after={<span className="pill ok">{LA.after.verify_script_exit_code}</span>}
            />
            <Row
              label="Segmentation test — strict IoU"
              before={f3(B.segmentation_test.strict_iou)}
              after={after.segStrictIoU}
            />
            <Row
              label="Segmentation test — Dice"
              before={f3(B.segmentation_test.dice)}
              after={after.segDice}
            />
            <Row
              label="Segmentation test — tolerance IoU"
              before={<span style={{ color: 'var(--ink-3)' }}>— (metric added post-audit)</span>}
              after={after.segTolIoU}
            />
            <Row
              label="Held-out test area — predicted / GFC reference"
              before={`${f1(B.area_test.predicted_ha)} / ${f1(B.area_test.gfc_reference_ha)} ha  (${f2(B.area_test.predicted_over_gfc)}×)`}
              after={`${f1(after.areaPred)} / ${f1(after.areaRef)} ha  (${f2(after.areaRatio)}×)`}
            />
            <Row
              label="Carbon, predicted test region (regression, primary)"
              before={kt(B.carbon_predicted_test.regression_exponential_tCO2)}
              after={kt(after.co2)}
            />
          </tbody>
        </table>
      </div>

      <div className="callout">
        <b>Read the two columns carefully.</b> {LA.run_count_caveat}
      </div>

      <p className="note" style={{ fontSize: 12.5, color: 'var(--ink-3)' }}>
        Contaminated patches each had {LA.before.contaminated_patch_area_pct_range[0]}–
        {LA.before.contaminated_patch_area_pct_range[1]}% of their area also present in
        training. The canonical grid, block-to-split assignment, normalisation
        statistics and the val/test patches are byte-identical before and after the
        fix (verified by hash) — only the overlap crops changed.
      </p>

      {/* ---- 3b. seed variance ---- */}
      <h3>Seed variance</h3>
      <p className="note">{content.seed_variance_note}</p>
      <p className="note" style={{ fontSize: 12.5 }}>
        Per-seed strict test IoU (U-Net): {segBaseline.test_iou.values.map((v) => v.toFixed(3)).join(' · ')}
        {'  '}(sd {segBaseline.test_iou.sd.toFixed(3)}).
      </p>

      {/* ---- 3c. overfitting ---- */}
      <h3>Overfitting</h3>
      <p className="note">{content.overfitting_note}</p>
      <p className="note" style={{ fontSize: 12.5 }}>
        U-Net per seed — best validation-Dice epoch{' '}
        {segBaseline.best_epoch.join(' · ')}; early stopping (patience{' '}
        {seedRuns.early_stop_patience}) halted training at epoch{' '}
        {segBaseline.stop_epoch.join(' · ')}.
      </p>
    </Section>
  )
}
