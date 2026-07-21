# Treat mixed use as a building-use class

The building-use task dataset will classify `mixed_use` alongside
`residential`, `office`, and `commercial`. A facade with a visibly commercial
podium and a different upper program must not be forced into either component
class; per-floor multi-label use annotation is deferred from the first release.

## Considered Options

- Collapsing mixed-use facades into a primary use would hide a common facade
  pattern and introduce contradictory classification labels.
- Per-floor multi-label classification preserves more detail but changes the
  task from building-use classification to a separate structured prediction
  problem.
