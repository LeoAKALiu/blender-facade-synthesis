# Derive window count from window-instance truth

The window-count task dataset will include a semantic mask, instance mask, and
per-instance spatial annotations; the count is derived from those instances.
This permits counting, detection, and instance-segmentation consumers to use
one auditable task dataset and makes every count directly verifiable against
the rendered scene.

## Considered Options

- A scalar-count-only dataset would be smaller but cannot establish which
  windows caused a count or support spatial quality assurance.
