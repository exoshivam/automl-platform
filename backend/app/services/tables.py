"""CSV/XLSX/Parquet/JSONL tables (DAT-2): read, infer column types, apply the user's mapping."""

import io
from typing import Any

import pandas as pd

from app.errors import ApiError
from app.services import storage

ROLES = ("target", "feature", "text", "id", "timestamp", "group", "ignore")
TYPES = ("categorical", "numeric", "text", "datetime", "boolean")
FORMATS = ("csv", "xlsx", "parquet", "jsonl")


def read_table(key: str, fmt: str, nrows: int | None = None) -> pd.DataFrame:
    data = storage.get_bytes(key)
    try:
        if fmt == "csv":
            return pd.read_csv(io.BytesIO(data), nrows=nrows, sep=None, engine="python", encoding="utf-8-sig")
        if fmt == "xlsx":
            return pd.read_excel(io.BytesIO(data), nrows=nrows, engine="openpyxl")
        if fmt == "parquet":
            df = pd.read_parquet(io.BytesIO(data))
            return df.head(nrows) if nrows else df
        if fmt == "jsonl":
            return pd.read_json(io.BytesIO(data), lines=True, nrows=nrows)
    except Exception as e:
        raise ApiError(422, "unreadable_table", f"Couldn't read the {fmt} file: {str(e)[:200]}") from e
    raise ApiError(422, "unsupported_format", f"Unsupported table format '{fmt}'")


def infer_type(s: pd.Series) -> str:
    v = s.dropna()
    if v.empty:
        return "categorical"
    if pd.api.types.is_bool_dtype(v) or set(map(str, v.unique())) <= {"0", "1", "True", "False", "true", "false",
                                                                     "yes", "no", "Yes", "No"}:
        return "boolean"
    if pd.api.types.is_numeric_dtype(v):
        return "numeric"
    sv = v.astype(str)
    if sv.str.len().mean() > 30:
        return "text"
    try:
        parsed = pd.to_datetime(sv, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.9 and sv.str.contains(r"\d{4}|\d{1,2}[/-]\d{1,2}").mean() > 0.9:
            return "datetime"
    except (ValueError, TypeError):
        pass
    return "categorical"


def preview(df: pd.DataFrame, total_rows: int) -> dict[str, Any]:
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({"name": str(c), "inferred_type": infer_type(s),
                     "sample_values": [None if pd.isna(x) else str(x)[:100] for x in s.head(5)],
                     "unique_count": int(s.nunique(dropna=True)),
                     "missing_pct": round(float(s.isna().mean() * 100), 2)})
    return {"columns": cols, "row_count": total_rows}


def check_mapping(df: pd.DataFrame, mapping: dict[str, Any], task_type: str) -> list[dict[str, Any]]:
    """Returns a normalised mapping; raises 422 on unknown columns, bad roles, or not exactly one target."""
    names = [str(c) for c in df.columns]
    cols = mapping.get("columns")
    if not isinstance(cols, list) or not cols:
        raise ApiError(422, "invalid_mapping", "mapping.columns must be a non-empty list")
    errors, seen, out = [], set(), []
    for i, c in enumerate(cols):
        name, role, typ = c.get("name"), c.get("role", "feature"), c.get("type")
        if name not in names:
            errors.append({"loc": f"/mapping/columns/{i}/name", "msg": f"No column '{name}' in the file"})
        if role not in ROLES:
            errors.append({"loc": f"/mapping/columns/{i}/role", "msg": f"role must be one of {list(ROLES)}"})
        if typ is not None and typ not in TYPES:
            errors.append({"loc": f"/mapping/columns/{i}/type", "msg": f"type must be one of {list(TYPES)}"})
        if name in seen:
            errors.append({"loc": f"/mapping/columns/{i}/name", "msg": "Column listed twice"})
        seen.add(name)
        out.append({"name": name, "role": role, "type": typ})
    targets = [c for c in out if c["role"] == "target"]
    if len(targets) != 1:
        errors.append({"loc": "/mapping/columns", "msg": f"Exactly one target column is required ({len(targets)} given)"})
    if task_type == "text_classification" and not any(c["role"] == "text" for c in out):
        errors.append({"loc": "/mapping/columns", "msg": "Text classification needs one column with role 'text'"})
    if errors:
        raise ApiError(422, "invalid_mapping", "The column mapping is invalid", errors=errors)
    # columns not listed are features with inferred types
    listed = {c["name"] for c in out}
    out += [{"name": n, "role": "feature", "type": None} for n in names if n not in listed]
    for c in out:
        c["inferred_type"] = infer_type(df[c["name"]])
        c["type"] = c["type"] or c["inferred_type"]
    return out


def to_parquet(df: pd.DataFrame, columns: list[dict[str, Any]]) -> bytes:
    df = df.copy()
    for c in columns:
        if c["type"] in ("categorical", "text"):
            df[c["name"]] = df[c["name"]].map(lambda x: None if pd.isna(x) else str(x))
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()
