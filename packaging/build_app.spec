# -*- mode: python ; coding: utf-8 -*-
import os
import sys

# SPECPATH (injected by PyInstaller) is this file's directory, so resolve
# paths from there regardless of the caller's current working directory.
ROOT_DIR = os.path.abspath(os.path.join(SPECPATH, os.pardir))

# macOS needs .icns for a proper Dock icon; everywhere else (Windows is the
# primary target) uses .ico. Ignored on Linux — PyInstaller doesn't support
# exe icons there.
ICON_FILE = "icon.icns" if sys.platform == "darwin" else "icon.ico"

# customtkinter, googleapiclient, cv2, numpy and PIL all ship (or are
# covered by pyinstaller-hooks-contrib with) their own PyInstaller hooks
# that auto-collect their hidden imports/data files, and yt_dlp bundles its
# own hook the same way — so datas/hiddenimports only list what the hooks miss.
#
# PIL._tkinter_finder: Pillow's ImageTk locates the Tcl/Tk image extension
# via a runtime import PyInstaller's static analysis doesn't see; without it
# the executable raises ModuleNotFoundError when a CTkImage is rendered.
a = Analysis(
    [os.path.join(ROOT_DIR, "app", "main.py")],
    pathex=[ROOT_DIR],
    binaries=[],
    datas=[],
    hiddenimports=["PIL._tkinter_finder"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="GestorVideo360",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(SPECPATH, ICON_FILE),
)
