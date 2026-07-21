import blenderproc as bproc

"""BlenderProc-first bootstrap for the locally owned V3.1 seed generator."""

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preflight", action="store_true")
    known, remaining = parser.parse_known_args(argv)

    bproc.init()
    source_root = Path(__file__).resolve().parents[1]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    if known.preflight:
        import bpy
        import numpy as np

        print(
            json.dumps(
                {
                    "runtime": "blenderproc_blender",
                    "blender_version": bpy.app.version_string,
                    "numpy_version": np.__version__,
                },
                sort_keys=True,
            )
        )
        return 0

    from facade_synth.seed_v31.blenderproc_facade_v3 import main as seed_main

    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    return seed_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
