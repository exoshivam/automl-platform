from functools import lru_cache

import redis

from app.settings import get_settings


@lru_cache
def client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def login_blocked(email: str, ip: str) -> int | None:
    """Seconds to wait if this email+IP already used up its failures in the window, else None."""
    s = get_settings()
    key = f"login_fail:{email.lower()}:{ip}"
    if int(client().get(key) or 0) >= s.login_rate_limit:
        return max(int(client().ttl(key)), 1)
    return None


def record_login_failure(email: str, ip: str) -> None:
    s = get_settings()
    key = f"login_fail:{email.lower()}:{ip}"
    pipe = client().pipeline()
    pipe.incr(key)
    pipe.expire(key, s.login_rate_window_seconds, nx=True)
    pipe.execute()


def clear_login_failures(email: str, ip: str) -> None:
    client().delete(f"login_fail:{email.lower()}:{ip}")
