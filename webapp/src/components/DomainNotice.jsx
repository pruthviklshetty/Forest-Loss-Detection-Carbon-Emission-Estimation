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
          the area (point + radius, or a custom box) must lie entirely inside{' '}
          {ext ? `[W ${ext[0]}, S ${ext[1]}, E ${ext[2]}, N ${ext[3]}]` : 'the domain extent'};
        </li>
        <li>both date windows must be within January–April, in increasing years;</li>
        <li>
          the smallest AOI is one model tile —{' '}
          {domain.caps?.min_tile_km ? `${domain.caps.min_tile_km} km` : '2.56 km'} on a
          side (256 px × 10 m); anything smaller is refused, not padded, because
          Hansen GFC labels are 30 m and there is no signal below that;
        </li>
        <li>
          the radius is capped at{' '}
          {domain.caps?.max_radius_km ? `${domain.caps.max_radius_km} km` : '20 km'}.
        </li>
      </ul>
    </Notice>
  )
}
