# Issue an immutable dataset receipt with every publication

Every published training package will include a dataset receipt that binds its
generation-brief hash, code Git SHA, Blender and BlenderProc versions, asset
fingerprints and SHAs, seeds, actual render parameters, validation result, and
human publication confirmation. A package without this receipt is not
publishable.

## Considered Options

- Logs alone aid debugging but cannot reliably identify the exact inputs used
  for a later model-training run.
