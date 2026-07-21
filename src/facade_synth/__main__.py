from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .web import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the local Blender Facade Synthesis Studio.")
    parser.add_argument("--workspace", type=Path, default=Path(".facade-synthesis"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    uvicorn.run(create_app(workspace=args.workspace), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
