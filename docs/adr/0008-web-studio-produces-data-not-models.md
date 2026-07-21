# Keep the Web Studio focused on training data

The Web Studio will configure generation, inspect quality, validate contracts,
and export self-describing training packages. Model training and evaluation
remain external consumers of those packages, so the rendering application's
release cycle and dependencies are not coupled to a particular ML framework.

## Considered Options

- In-app training would demonstrate an end-to-end loop but requires model,
  accelerator, experiment, and checkpoint-management concerns outside the
  dataset application's purpose.
