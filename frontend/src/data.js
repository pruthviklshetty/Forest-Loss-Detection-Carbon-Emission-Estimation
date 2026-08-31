// Single point where the dashboard reads its numbers. Every metric shown in the
// UI comes from one of these imports at BUILD TIME. No metric is hardcoded in a
// component. Updating a results file and rebuilding updates the page.
//
//  - seedRuns / areaSummary / carbon / phase4 : real computed results
//  - content : values not present in the result JSON (pre-audit baselines for
//    the before/after table, leakage-audit stats, the external GFW reference,
//    and static paper text). See results/dashboard/content.json "_meta".

import seedRuns from '../../results/metrics/seed_runs.json'
import areaSummary from '../../results/deforestation/baseline_unet_area_summary.json'
import carbon from '../../results/carbon_validation/carbon_estimates.json'
import phase4 from '../../results/metrics/phase4_comparison.json'
import content from '../../results/dashboard/content.json'

export { seedRuns, areaSummary, carbon, phase4, content }

// convenience: the pipeline model's per-metric {mean, sd, values}
export const segBaseline = seedRuns.summary.baseline_unet
export const segAttention = seedRuns.summary.attention_unet
