# Use user-selected daylight profiles in the first release

Generation briefs will let the user set counts and intensity ranges for two
daylight profiles: diverse daylight and controlled clear/overcast daylight.
Night, fog, rain, and other weather-degradation conditions are excluded from
the first release so visual QA thresholds remain comparable across every task
package.

Each sample will record its actual Blender lighting recipe: sun elevation and
relative azimuth, sun energy, world strength, exposure EV, and colour
temperature. The UI may describe a range as weak, normal, or strong, but must
not present renderer controls as physically calibrated lux.

## Considered Options

- A single controlled daylight profile would make rendering easier to compare
  but narrow the appearance domain.
- Including night and adverse weather now would broaden realism but require a
  distinct low-visibility quality and release contract.
