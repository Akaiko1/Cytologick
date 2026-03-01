#!/usr/bin/env python3
"""
Backward-compatible alias for `run_audit.py`.
"""

from __future__ import annotations

from run_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
