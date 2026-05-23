"""emboss — On-Disk Input-keyed Cache.

Disk-backed memoization built on `diskcache`, with auto-detection of
pydantic v2 `BaseModel` return types (encoded via `model_dump`, decoded
via `model_validate`) so models defined in `__main__` round-trip across
script invocations.

Usage::

    import diskcache
    from emboss import cached

    cache = diskcache.Cache("/tmp/my-cache")

    @cached(cache)
    def expensive(url: str) -> dict:
        return requests.get(url).json()

    # pydantic BaseModel returns are auto-encoded / decoded
    from pydantic import BaseModel

    class User(BaseModel):
        name: str

    @cached(cache)
    def get_user(uid: int) -> User | None:
        return User.model_validate(requests.get(f"/users/{uid}").json())

See README.md for the full feature list.
"""

from emboss._cached import cached, safe_jsonable_encoder

__version__ = "0.1.0"
__all__ = ["cached", "safe_jsonable_encoder"]
