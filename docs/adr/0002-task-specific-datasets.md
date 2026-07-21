# Generate independent task-specific datasets

Each learning target will receive its own generated and versioned task dataset,
rather than being an export view of a shared multi-task corpus. This makes a
dataset's class vocabulary, annotations, quality gates, and train/validation/
test split explicit for its training consumer, even though the scene grammar
and rendering worker may be shared.

Each task dataset also owns a task curriculum: sampling distribution, class
balance, and visibility thresholds. For example, window instances can emphasize
small dense windows and light occlusion, while floorline tasks favor readable
floor bands and building-use tasks balance their four classes.

## Considered Options

- A single canonical multi-task dataset would reduce duplicated renders but
  couples unrelated task contracts and release cadence.
- Post-generation export profiles would be convenient but leave ambiguity over
  which sample and label version trained a particular model.
