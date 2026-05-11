"""Entry point for 360-Video-Manager.

Run as a module::

    python -m app.main          # launch GUI
    python -m app.main --cli    # launch CLI  (all other flags forwarded)
    python -m app.main --url "https://youtu.be/..." --cli --upload

The ``--cli`` flag triggers non-interactive CLI mode; every other argument
is forwarded verbatim to the :mod:`app.cli` parser.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]

    if "--cli" in args:
        # Strip the --cli flag before forwarding to the CLI parser.
        cli_args = [a for a in args if a != "--cli"]
        from app.cli import run_cli
        sys.exit(run_cli(cli_args))
    else:
        from app.gui.gui_app import run_gui
        run_gui()


if __name__ == "__main__":
    main()
