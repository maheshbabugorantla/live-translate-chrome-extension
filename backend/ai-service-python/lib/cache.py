"""
lib/cache.py — two-tier cache: memory + SQLite
=====================================================================
Why two tiers?
  - MEMORY (dict): instant, but lost on restart.
  - SQLite (disk): survives restarts, and is where you can inspect what your
    service has learned. Check memory first, then disk, then LLM.

The cache key must be deterministic for the same (text, target). Hashing the
input with sha256 gives you a compact, collision-safe key.
"""
import hashlib

import aiosqlite


def _key(text: str, target: str) -> str:
    return hashlib.sha256(f"{target}::{text}".encode("utf-8")).hexdigest()


class TwoTierCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._mem: dict[str, str] = {}
        self._stats = {"requests": 0, "memory_hits": 0, "db_hits": 0, "misses": 0}

    async def init(self) -> None:
        """Create the translations table if it doesn't exist."""
        async with aiosqlite.connect(self.db_path) as db_cnxn:
            await db_cnxn.execute("""
                CREATE TABLE IF NOT EXISTS translations(
                key TEXT PRIMARY KEY, source TEXT, target TEXT, translated TEXT,
                model TEXT, access_count INTEGER DEFAULT 1, created_at TIMESTAMP)
            """)
            await db_cnxn.commit()

    async def get(self, text: str, target: str) -> str | None:
        """Return a cached translation or None. Check memory, then SQLite."""
        self._stats["requests"] += 1
        k = _key(text, target)

        # 1) memory tier
        if k in self._mem:
            self._stats["memory_hits"] += 1
            return self._mem[k]

        # 2) SQLite tier
        async with aiosqlite.connect(self.db_path) as db_cnxn:
            query = "SELECT translated FROM translations WHERE key = ?"
            async with db_cnxn.execute(query, (k,)) as cursor:
                row = await cursor.fetchone()
            if row:
                self._mem[k] = row[0] # Warm the cache
                self._stats["db_hits"] += 1
                update_access_count_query = "UPDATE translations SET access_count = access_count + 1 WHERE key = ?"
                await db_cnxn.execute(update_access_count_query, (k,))
                await db_cnxn.commit()
                return row[0]
        self._stats["misses"] += 1

    async def set(self, text: str, target: str, translated: str, model: str) -> None:
        """Store a translation in both tiers."""
        k = _key(text, target)
        self._mem[k] = translated
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

    async def size(self) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM translations") as cur:
                row = await cur.fetchone()
                return row[0] if row else 0

    async def stats(self) -> dict:
        total = self._stats["memory_hits"] + self._stats["db_hits"] + self._stats["misses"]
        hits = self._stats["memory_hits"] + self._stats["db_hits"]
        hit_rate = round(100 * hits / total, 1) if total else 0.0
        return {**self._stats, "hit_rate_pct": hit_rate, "memory_entries": len(self._mem)}
