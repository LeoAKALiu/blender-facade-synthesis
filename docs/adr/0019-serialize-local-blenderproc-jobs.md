# Serialize local BlenderProc jobs

The first-release local worker will process exactly one BlenderProc generation
job at a time. It may queue jobs and cancel jobs that have not started; a
running job stops safely only after its current sample completes. This keeps
GPU/CPU and memory ownership predictable and makes progress and diagnostic logs
belong to one job.

## Considered Options

- Parallel local jobs can improve throughput on some machines but make renderer
  failures, resource contention, progress, and cancellation hard to attribute.
