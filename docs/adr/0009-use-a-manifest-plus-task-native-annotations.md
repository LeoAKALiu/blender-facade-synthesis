# Use a package manifest plus task-native annotations

Every training package will include a common JSONL package manifest, then use
the annotation forms natural to its task: COCO JSON and PNG masks for window
instances, PNG heatmaps plus JSON polylines for floorlines, and JSONL image
labels for visible-floor-count and building-use classification. This gives
external training code one stable entry point without forcing all targets into
one lossy annotation format.

## Considered Options

- A custom JSON-only contract would make all packages uniform but requires each
  consumer to reimplement common spatial annotation adapters.
- Framework-specific data layouts would be convenient for one trainer but make
  the dataset application dependent on that framework.
