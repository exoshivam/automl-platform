"""Cursor pagination over UUIDv7 ids (time-ordered): newest first, cursor = last id of the previous page."""

import uuid

from pydantic import BaseModel
from sqlalchemy import Select
from sqlalchemy.orm import Session

from app.errors import ApiError


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None


def paginate(db: Session, stmt: Select, id_col, cursor: str | None, limit: int, newest_first: bool = True):
    limit = max(1, min(limit, 200))
    if cursor:
        try:
            cur = uuid.UUID(cursor)
        except ValueError as e:
            raise ApiError(422, "invalid_cursor", "Malformed cursor") from e
        stmt = stmt.where(id_col < cur if newest_first else id_col > cur)
    stmt = stmt.order_by(id_col.desc() if newest_first else id_col.asc()).limit(limit + 1)
    rows = list(db.scalars(stmt).all())
    more = len(rows) > limit
    rows = rows[:limit]
    return rows, (str(rows[-1].id) if more else None)
