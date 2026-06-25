# Building the desktop executable

Produces a double-clickable `GestorVideo360` executable for clinical/therapist
machines that have no Python installed. Windows is the primary target;
macOS is supported as a bonus.

## Prerequisites (on the machine doing the build)

1. Python >= 3.9 with the project's own dependencies installed:
   ```
   pip install -r requirements.txt
   ```
2. **ffmpeg installed system-wide and on PATH.** The app shells out to a bare
   `ffmpeg`/`ffprobe` command (`detector/video_io.py`) — it is *not* bundled
   into the executable, so this is a one-time setup step per machine:
   - Windows: `winget install ffmpeg` (or download a static build and add its
     `bin` folder to PATH manually).
   - macOS: `brew install ffmpeg`.
   - Verify with `ffmpeg -version` in a terminal before building or
     distributing.
3. PyInstaller (build-time only, not a runtime dependency of the app):
   ```
   pip install pyinstaller
   ```

## Regenerating the icon (optional — already committed under `packaging/`)

Only needed if you want to redesign the icon. From the repo root:
```
python packaging/icon_generator.py
```
Regenerates `packaging/icon_256.png`, `packaging/icon.ico`, and
`packaging/icon.icns`. `.icns` generation works directly from any OS with a
recent Pillow (>=9.2); if it fails on your setup, generate it on macOS
instead, or convert `icon_256.png` with `iconutil`/`png2icns`.

## Building

From the repo root, on the **target OS itself** — PyInstaller does not
cross-compile, so a Windows `.exe` must be built on Windows, and a macOS
binary must be built on macOS:

```
pyinstaller packaging/build_app.spec
```

Output:
- Windows: `dist/GestorVideo360.exe`
- macOS: `dist/GestorVideo360` (a plain onefile binary, not a `.app` bundle —
  fine for a bonus target; revisit with PyInstaller's `BUNDLE()` step later
  if proper Dock integration becomes a real need)

The spec resolves all paths relative to its own location, so it can be
invoked from anywhere (`pyinstaller packaging/build_app.spec` from the repo
root, or `cd packaging && pyinstaller build_app.spec`).

`build_app.spec` itself needs almost no manual hidden-imports/data
configuration — `customtkinter`, `googleapiclient`, `cv2`, `numpy` and `PIL`
are all covered by hooks bundled with `pyinstaller`/`pyinstaller-hooks-contrib`,
and `yt_dlp` ships its own. The one manual addition
(`PIL._tkinter_finder`) was found by actually running a test build and
fixing the crash it produced — if you add new dependencies later and hit a
similar `ModuleNotFoundError` only when frozen, that's the pattern to expect
and fix the same way.

## Verification checklist

- [ ] Double-clicking the executable opens the GUI with no console window
- [ ] The custom icon shows in the taskbar/title bar (not a Python icon)
- [ ] Settings in a `.env` placed next to the executable are picked up, and
      persist across runs (downloads/logs/jobs land in a `data/` folder next
      to the executable, not in a vanishing temp directory)
- [ ] With ffmpeg installed and on PATH, a YouTube URL can be searched,
      downloaded, detected, converted, and uploaded end-to-end without
      touching a terminal
- [ ] If ffmpeg is missing or a download/upload fails, the activity log shows
      a plain-language message, not a crash dialog
- [ ] Runs on a machine with no Python installed (test in a clean VM)

## Known limitations / explicit scope decisions

- No bundled ffmpeg — must be installed system-wide (see Prerequisites).
- No in-app MediaCMS server-config/"Test connection" UI — server URL and
  credentials are set once via `.env` next to the executable.
- The executable is sizeable (opencv-contrib-python alone is large); this is
  expected, not a packaging bug.
