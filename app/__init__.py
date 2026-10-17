"""NoiseLab — web extension of the Simulink spectral-subtraction model.

The Simulink model is the primary implementation; this app re-implements the same
algorithm in Python for interactive exploration and comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask

from .config import DEFAULTS
from .services.jobs import JobStore, ProfileStore

__version__ = "0.2.0"


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(DEFAULTS)
    if test_config:
        app.config.update(test_config)

    root = Path(app.config["JOB_ROOT"])
    app.extensions["jobs"] = JobStore(root, app.config["MAX_JOBS"], app.config["JOB_TTL_S"])
    app.extensions["profiles"] = ProfileStore(app.config["MAX_PROFILES"], app.config["JOB_TTL_S"])

    from .routes import bp
    app.register_blueprint(bp)
    return app
