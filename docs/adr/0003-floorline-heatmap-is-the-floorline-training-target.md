# Use heatmaps as the floorline training target

The floorline task dataset will train on raster heatmaps, with scene-truth
polylines retained as annotation and evaluation data rather than as a second
task dataset. Heatmaps tolerate small line-placement variation during training,
while polylines preserve precise geometry for quality checks and downstream
line extraction.

## Considered Options

- A separate polyline-prediction dataset would duplicate the same underlying
  semantics and create two independently versioned floorline contracts.
