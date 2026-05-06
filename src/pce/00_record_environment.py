#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import collect_environment, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Record hardware, Python, and benchmark environment metadata.")
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_path = output_root / "environment.json"
    write_json(output_path, collect_environment())
    print(f"[environment] wrote {output_path}")


if __name__ == "__main__":
    main()

