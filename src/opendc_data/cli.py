"""Dispatcher for the `opendc-data` console script.

    opendc-data build --config datasets.yaml --tokenizer hf:<model> --out data/x
    opendc-data fetch-corpus --out data/_public_src

`python -m opendc_data.build --config ...` (no subcommand) keeps working
unchanged, so existing cluster scripts are unaffected.
"""
from __future__ import annotations

import sys

_COMMANDS = {"build": "opendc_data.build", "fetch-corpus": "opendc_data.fetch"}


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help") or not argv:
        print(__doc__.strip())
        print("\ncommands: " + ", ".join(_COMMANDS))
        raise SystemExit(0 if argv else 2)

    name = argv[0] if argv[0] in _COMMANDS else "build"
    if argv[0] in _COMMANDS:
        sys.argv = [f"opendc-data {name}"] + argv[1:]
    import importlib
    importlib.import_module(_COMMANDS[name]).main()


if __name__ == "__main__":
    main()
