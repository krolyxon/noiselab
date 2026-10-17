"""Limits, paths and defaults. Override any key via create_app(test_config={...})."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

DEFAULTS = {
    "MAX_CONTENT_LENGTH": 25 * 1024 * 1024,      # 25 MB upload limit (Flask enforces → 413)
    "MAX_DURATION_S": 120.0,                     # reject longer clips
    "MIN_DURATION_S": 0.25,
    "MAX_SAMPLE_RATE": 48000,                    # resample above this, keep otherwise
    "MAX_PROFILE_DURATION_S": 30.0,
    "JOB_TTL_S": 30 * 60,                        # jobs are deleted after 30 min idle
    "MAX_JOBS": 16,                              # in-memory LRU size
    "MAX_PROFILES": 32,
    "DEMO_MAX_DURATION_S": 20.0,
    "JOB_ROOT": Path(os.environ.get("NOISELAB_JOB_ROOT", Path(tempfile.gettempdir()) / "noiselab-jobs")),
    "JSON_SORT_KEYS": False,
}
