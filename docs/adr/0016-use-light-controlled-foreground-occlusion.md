# Use light controlled foreground occlusion

The first release will render clear, light, and moderate foreground-occlusion
bands, limited to 0–30% of the target facade. The renderer will calculate
scene-truth occlusion and task-specific visibility thresholds before a sample
can enter a trainable package; no component is labelled merely because it exists
behind an occluder.

## Considered Options

- Clean-only facades simplify labels but leave a material street-view gap.
- Heavy occlusion provides a harder robustness case but sharply reduces useful
  window instances and continuous floorline evidence in the main packages.
