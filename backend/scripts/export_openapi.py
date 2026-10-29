"""Writes backend/openapi.json (the contract the frontend's TS types are generated from). Run: uv run python scripts/export_openapi.py"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("APP_ENV", "dev")
from app.main import app

out = Path(__file__).resolve().parent.parent / "openapi.json"
out.write_text(json.dumps(app.openapi(), indent=2, sort_keys=False) + "\n")
print(f"wrote {out} ({len(app.openapi()['paths'])} paths)")
