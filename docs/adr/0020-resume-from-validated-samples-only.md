# Resume jobs from validated samples only

An interrupted job may resume from samples that were fully written and passed
per-sample validation under its immutable brief. The worker will regenerate only
missing samples; no partial job can be published until it meets the requested
count, passes full-package validation, and receives publication confirmation.

## Considered Options

- Restarting all samples is simpler but wastes local rendering time.
- Treating every pre-existing output as complete risks mixing partial or stale
  artifacts into a trainable package.
