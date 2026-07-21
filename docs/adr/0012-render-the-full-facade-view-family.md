# Render the full facade view family for each recipe

The first-release generation brief will cover all displayed facade angle bands:
frontal, light/medium-oblique, and strong-oblique views. Each render receives
its own projected visibility truth for windows, floors, and floorlines, while
all views derived from one building recipe remain in the same train,
validation, or test split.

## Consequences

- A component may be present in the scene yet excluded from a view's labels if
  its projected visibility does not meet the task's threshold.
- Overhead and nadir camera regimes are not implied by this facade-view family
  and require a later explicit dataset decision.
