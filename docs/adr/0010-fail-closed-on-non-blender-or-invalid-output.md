# Fail closed on non-Blender or invalid output

A trainable package may contain only samples rendered by the local
BlenderProc/Blender path and validated against the asset and annotation
contracts. If rendering falls back to projection, an asset fingerprint cannot
be read or does not match its local file, or labels do not validate, the job may
retain diagnostic previews but cannot publish a trainable package.

## Considered Options

- Allowing fallbacks keeps jobs green on machines without Blender, but would
  silently contaminate the real-render distribution promised by the package.
