import { Notice } from './ui.jsx'

// Always-visible statement of what the model is valid for. This must not be
// collapsible or dismissible - it is a hard constraint, not a tip.
export default function DomainNotice({ domain }) {
  if (!domain) return null
  const ext = domain.domain_extent_wsen
  const tw = domain.training_windows || {}
  return (
    <Notice kind="domain" title="Model domain — enforced">
      <p>
        This model was trained only on <b>Western Ghats moist forest</b>,{' '}
        <b>January–April</b> Sentinel-2 composites,{' '}
        <b>
          {tw.T?.label ?? '2019'} vs {tw.T_plus_1?.label ?? '2021'}
        </b>
        . Requests are <b>refused</b> (not warned) when they fall outside it:
      </p>
      <ul>
        <li>
          custom bounding boxes must lie entirely inside{' '}
          {ext ? `[W ${ext[0]}, S ${ext[1]}, E ${ext[2]}, N ${ext[3]}]` : 'the domain extent'};
        </li>
        <li>both date windows must be within January–April, in increasing years;</li>
        <li>
          area is capped at{' '}
          {domain.caps ? `${domain.caps.max_area_km2} km²` : 'the configured limit'}.
        </li>
      </ul>
    </Notice>
  )
}
