# -*- mode: python ; coding: utf-8 -*-

import os

# PyInstaller executes .spec as Python code and provides SPECPATH global that
# points to the directory containing the spec file.
project_root = os.path.abspath(globals().get('SPECPATH') or os.path.dirname(__file__))


# We intentionally do NOT bundle config/models here.
# Recommended distribution is a one-folder app where `config.yaml`, `_main/` and
# OpenSlide binaries are placed next to the executable.
datas = []

a = Analysis(
    ['run.py'],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    # Some imports are dynamic (openslide inside a function; remote connector optional)
    hiddenimports=['openslide', 'tfs_connector'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='run',
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
)
