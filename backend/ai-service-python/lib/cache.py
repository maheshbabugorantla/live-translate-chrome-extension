"""
lib/cache.py — cache: memory + Redis (optional, shared) + SQLite
=====================================================================
Why these tiers?
  - MEMORY (dict): instant, but per-process and lost on restart.
  - REDIS (optional): shared across every replica of this service — a
    translation cached by one instance is a hit for all of them. Purely a
    throughput/cost win once you run more than one replica; on a single
    local process it's a no-op with a bit of added latency. Fully OPTIONAL:
    if REDIS_URL isn't set, or Redis is unreachable, this tier is skipped
    and the cache behaves exactly as it did before — a Redis outage must
    never fail a translation.
  - SQLite (disk): survives restarts, and is where you can inspect what
    this service has learned. Also what seeds Redis when a fresh replica
    starts and asks for something this process already knows from disk.

Lookup order: memory -> Redis -> SQLite -> miss (fall through to the LLM).
Writes fan out to every tier that's available.

The cache key must be deterministic for the same (text, target). Hashing the
input with sha256 gives you a compact, collision-safe key.
"""
import hashlib

import aiosqlite

try:
    import redis.asyncio as redis_asyncio
except ImportError:  # pragma: no cover - redis is an optional dependency
    redis_asyncio = None

from lib.logger import get_logger

log = get_logger("cache")

REDIS_KEY_PREFIX = "xlate:"
REDIS_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days — bounds unbounded growth


def normalize_text(text: str) -> str:
    """Normalize text so trivial variants (extra whitespace, different
    casing) collapse to one identity — used both for the cache KEY (below)
    and, in app.py, for single-flight dedup, so the two layers agree on
    what counts as "the same text." The raw `text` is still stored in the
    `source` column — normalization never changes what's displayed or
    returned, only what's treated as equivalent.
    """
    return " ".join(text.split()).casefold()


def _key(text: str, target: str) -> str:
    return hashlib.sha256(f"{target}::{normalize_text(text)}".encode("utf-8")).hexdigest()


def _redact_redis_url(url: str) -> str:
    """Strip credentials before logging a Redis URL."""
    scheme_sep = url.find("://")
    if scheme_sep != -1 and "@" in url[scheme_sep:]:
        return url[: scheme_sep + 3] + "***@" + url.split("@", 1)[1]
    return url


class TwoTierCache:
    def __init__(self, db_path: str, redis_url: str | None = None):
        self.db_path = db_path
        self.redis_url = redis_url
        self._mem: dict[str, str] = {}
        self._redis = None  # set in init() if redis_url is configured and reachable
        self._redis_status = "disabled"  # "disabled" | "down" | "ok"
        self._stats = {"requests": 0, "memory_hits": 0, "redis_hits": 0, "db_hits": 0, "misses": 0}

    async def init(self) -> None:
        """Create the translations table, then (optionally) connect to Redis."""
        async with aiosqlite.connect(self.db_path) as db_cnxn:
            await db_cnxn.execute("""
                CREATE TABLE IF NOT EXISTS translations(
                key TEXT PRIMARY KEY, source TEXT, target TEXT, translated TEXT,
                model TEXT, access_count INTEGER DEFAULT 1, created_at TIMESTAMP)
            """)
            await db_cnxn.commit()

        if not self.redis_url:
            self._redis_status = "disabled"
            return
        if redis_asyncio is None:
            log.warning("redis_not_installed", extra={"hint": "pip install redis"})
            self._redis_status = "down"
            return
        try:
            client = redis_asyncio.from_url(self.redis_url, decode_responses=True)
            await client.ping()
            self._redis = client
            self._redis_status = "ok"
            log.info("redis_connected", extra={"url": _redact_redis_url(self.redis_url)})
        except Exception as e:
            log.warning("redis_connect_failed", extra={"error": str(e)})
            self._redis = None
            self._redis_status = "down"

    async def close(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception as e:
                log.warning("redis_close_failed", extra={"error": str(e)})

    def redis_status(self) -> str:
        """'disabled' (no REDIS_URL), 'down' (configured but unreachable), or 'ok'."""
        return self._redis_status

    async def _redis_get(self, key_hash: str) -> str | None:
        if self._redis is None:
            return None
        try:
            value = await self._redis.get(REDIS_KEY_PREFIX + key_hash)
            self._redis_status = "ok"
            return value
        except Exception as e:
            log.warning("redis_get_failed", extra={"error": str(e)})
            self._redis_status = "down"
            return None

    async def _redis_set(self, key_hash: str, translated: str) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.set(REDIS_KEY_PREFIX + key_hash, translated, ex=REDIS_TTL_SECONDS)
            self._redis_status = "ok"
        except Exception as e:
            log.warning("redis_set_failed", extra={"error": str(e)})
            self._redis_status = "down"

    async def _redis_mget(self, key_hashes: list[str]) -> dict[str, str]:
        """Returns {key_hash: translated} for whichever keys Redis has."""
        if self._redis is None or not key_hashes:
            return {}
        try:
            values = await self._redis.mget([REDIS_KEY_PREFIX + k for k in key_hashes])
            self._redis_status = "ok"
            return {k: v for k, v in zip(key_hashes, values) if v is not None}
        except Exception as e:
            log.warning("redis_mget_failed", extra={"error": str(e)})
            self._redis_status = "down"
            return {}

    async def _redis_mset(self, mapping: dict[str, str]) -> None:
        """mapping: {key_hash: translated}. Warms Redis from a SQLite hit or a
        fresh LLM translation so the next replica sees it via the shared tier."""
        if self._redis is None or not mapping:
            return
        try:
            pipe = self._redis.pipeline()
            for key_hash, translated in mapping.items():
                pipe.set(REDIS_KEY_PREFIX + key_hash, translated, ex=REDIS_TTL_SECONDS)
            await pipe.execute()
            self._redis_status = "ok"
        except Exception as e:
            log.warning("redis_mset_failed", extra={"error": str(e)})
            self._redis_status = "down"

    async def get(self, text: str, target: str) -> str | None:
        """Return a cached translation or None. Check memory, then Redis, then SQLite."""
        self._stats["requests"] += 1
        k = _key(text, target)

        # 1) memory tier
        if k in self._mem:
            self._stats["memory_hits"] += 1
            return self._mem[k]

        # 2) Redis tier (shared across replicas; no-op if not configured/reachable)
        redis_value = await self._redis_get(k)
        if redis_value is not None:
            self._mem[k] = redis_value  # warm the memory tier
            self._stats["redis_hits"] += 1
            return redis_value

        # 3) SQLite tier
        async with aiosqlite.connect(self.db_path) as db_cnxn:
            query = "SELECT translated FROM translations WHERE key = ?"
            async with db_cnxn.execute(query, (k,)) as cursor:
                row = await cursor.fetchone()
            if row:
                self._mem[k] = row[0]  # warm the cache
                self._stats["db_hits"] += 1
                update_access_count_query = "UPDATE translations SET access_count = access_count + 1 WHERE key = ?"
                await db_cnxn.execute(update_access_count_query, (k,))
                await db_cnxn.commit()
                await self._redis_set(k, row[0])  # seed the shared tier from disk
                return row[0]
        self._stats["misses"] += 1

    async def set(self, text: str, target: str, translated: str, model: str) -> None:
        """Store a translation in every available tier."""
        k = _key(text, target)
        self._mem[k] = translated
        await self._redis_set(k, translated)
        upsert_query = """
            INSERT INTO translations (key, source, target, translated, model, created_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (key)
            DO UPDATE SET
                translated = excluded.translated,
                model = excluded.model,
                access_count = access_count + 1;
        """
        async with aiosqlite.connect(self.db_path) as db_cnxn:
            await db_cnxn.execute(upsert_query, (k, text, target, translated, model))
            await db_cnxn.commit()

    async def get_many(self, pairs: list[tuple[str, str]]) -> dict[str, str]:
        """Batched lookup for (text, target) pairs. Returns {text: translated}
        for hits only — callers treat a missing text as a cache miss.

        Checks memory for every pair first (no I/O), then a single Redis
        MGET for the remainder, then one SQLite round-trip
        (`WHERE key IN (...)`) for whatever's still missing.
        """
        hits: dict[str, str] = {}
        remaining: dict[str, str] = {}  # key -> original text, for Redis/DB lookups
        for text, target in pairs:
            self._stats["requests"] += 1
            k = _key(text, target)
            if k in self._mem:
                self._stats["memory_hits"] += 1
                hits[text] = self._mem[k]
            else:
                remaining[k] = text

        if not remaining:
            return hits

        redis_hits = await self._redis_mget(list(remaining.keys()))
        for k, translated in redis_hits.items():
            self._mem[k] = translated  # warm the memory tier
            self._stats["redis_hits"] += 1
            hits[remaining[k]] = translated
            del remaining[k]

        if not remaining:
            return hits

        keys = list(remaining.keys())
        placeholders = ",".join("?" * len(keys))
        sqlite_hits: dict[str, str] = {}
        async with aiosqlite.connect(self.db_path) as db_cnxn:
            query = f"SELECT key, translated FROM translations WHERE key IN ({placeholders})"
            async with db_cnxn.execute(query, keys) as cursor:
                rows = await cursor.fetchall()
            if rows:
                update_query = "UPDATE translations SET access_count = access_count + 1 WHERE key = ?"
                await db_cnxn.executemany(update_query, [(k,) for k, _ in rows])
                await db_cnxn.commit()

            hit_keys = {k for k, _ in rows}
            for k, translated in rows:
                self._mem[k] = translated  # warm the cache
                self._stats["db_hits"] += 1
                hits[remaining[k]] = translated
                sqlite_hits[k] = translated

            self._stats["misses"] += len(keys) - len(hit_keys)

        await self._redis_mset(sqlite_hits)  # seed the shared tier from disk
        return hits

    async def set_many(self, entries: list[tuple[str, str, str, str]]) -> None:
        """Batched store for (text, target, translated, model) tuples — one
        connection, one executemany, instead of one connect+INSERT per entry.
        """
        if not entries:
            return
        upsert_query = """
            INSERT INTO translations (key, source, target, translated, model, created_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (key)
            DO UPDATE SET
                translated = excluded.translated,
                model = excluded.model,
                access_count = access_count + 1;
        """
        rows = []
        redis_mapping: dict[str, str] = {}
        for text, target, translated, model in entries:
            k = _key(text, target)
            self._mem[k] = translated
            rows.append((k, text, target, translated, model))
            redis_mapping[k] = translated

        await self._redis_mset(redis_mapping)

        async with aiosqlite.connect(self.db_path) as db_cnxn:
            await db_cnxn.executemany(upsert_query, rows)
            await db_cnxn.commit()

    async def size(self) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM translations") as cur:
                row = await cur.fetchone()
                return row[0] if row else 0

    async def stats(self) -> dict:
        total = self._stats["memory_hits"] + self._stats["redis_hits"] + self._stats["db_hits"] + self._stats["misses"]
        hits = self._stats["memory_hits"] + self._stats["redis_hits"] + self._stats["db_hits"]
        hit_rate = round(100 * hits / total, 1) if total else 0.0
        return {**self._stats, "hit_rate_pct": hit_rate, "memory_entries": len(self._mem), "redis_status": self._redis_status}
