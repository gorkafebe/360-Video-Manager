"""Entry point for 360-Video-Manager GUI."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    _ = argv
    from app.gui.gui_app import run_gui
    run_gui()


if __name__ == "__main__":
    main()
