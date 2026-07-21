# Use a local Web Studio with a local BlenderProc worker

The application will run on the user's machine: the Web Studio configures,
submits, inspects, and exports dataset jobs, while a local worker owns
BlenderProc invocation and dataset writes. This keeps rendering inputs and
outputs local, makes batch runs reproducible outside the browser, and avoids
requiring a remote GPU service for the first release.

## Considered Options

- A CLI-only generator would be easy to automate but would not provide the
  requested interactive sample inspection and export workflow.
- A remote GPU service would scale later but adds job, storage, identity, and
  deployment boundaries before they are needed.
