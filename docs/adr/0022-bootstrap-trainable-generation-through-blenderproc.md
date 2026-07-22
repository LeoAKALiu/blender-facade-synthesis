# Bootstrap trainable generation through BlenderProc

Status: accepted

## Context

The inherited V3.1 runner imported NumPy before BlenderProc initialization. In
the local macOS installation that allowed Blender's Python to resolve Anaconda
NumPy, producing an ABI error. The old runner also allowed projection fallback,
which conflicts with the Trainable Package contract.

## Decision

Every Worker render starts from a BlenderProc entrypoint whose first executable
statement imports `blenderproc as bproc`, then calls `bproc.init()` before
adding the project source root or importing NumPy and V3.1 modules. The local
V3.1 runner rejects forced and automatic projection fallback, writes a runtime
summary, and is accepted only when every sample reports
`blenderproc_blender` with zero fallback count.

The Web Studio exposes this as BlenderProc preflight. A failed preflight is an
explicit environment-not-ready outcome; it cannot publish diagnostics or switch
to a different render backend.

## Consequences

- Local source imports must occur after BlenderProc initialization inside the
  BlenderProc process because initialization cleans inherited Python paths.
- The entrypoint intentionally keeps its BlenderProc import before normal module
  imports, with a narrow lint exemption documenting that requirement.
- Existing V3.1 fallback tests are replaced by regression tests that assert
  fallback rejection and real BlenderProc smoke evidence.
