from __future__ import annotations

import os
import json
from typing import Any, Dict

import pandas as pd
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_json_event(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"ts_utc": utc_now_iso(), **payload}
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)
    