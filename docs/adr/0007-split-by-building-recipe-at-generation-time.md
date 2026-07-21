# Split datasets by building recipe at generation time

Every task dataset will allocate recipes to `train`, `validation`, and `test`
using a user-confirmed ratio before rendering; 70/15/15 is the default, not a
fixed requirement. A recipe's facade grammar, parameters, and random seed
lineage may not cross splits, so a test image is not a minor camera or lighting
variation of a training facade.

## Considered Options

- Randomly splitting completed images is simpler but leaks near-duplicate
  building structure into validation and test data.
