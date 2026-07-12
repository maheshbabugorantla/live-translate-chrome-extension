"""
FDE · Assignment 1 · Python AI Service  (this is the real assignment)
=====================================================================
A small FastAPI service that translates English → Mexican Spanish with:
  - an LLM call            (lib/llm.py)
  - a two-tier cache       (lib/cache.py)  — memory + SQLite
  - structured logging     (lib/logger.py) — provided, wired for you

The Node gateway forwards the browser's requests here. You implement the
TODOs so the widget lights up. Run:

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env          # then add your API key
    uvicorn app:app --reload --port 8000
"""
import asyncio
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel

from lib.cache import TwoTierCache, normalize_text
from lib.constants import MODEL
from lib.llm import translate_text
from lib.logger import get_logger

load_dotenv()

DB_PATH = os.getenv("TRANSLATION_DB_PATH", "translations.db")
REDIS_URL = os.getenv("REDIS_URL")  # optional — shared cache tier across replicas

app = FastAPI(title="FDE Live Translate — AI Service")
log = get_logger("ai-service")
cache = TwoTierCache(DB_PATH, redis_url=REDIS_URL)


# request/response shapes ----------------------------------------------------
class TranslateIn(BaseModel):
    text: str
    target: str = "es-MX"


class BatchIn(BaseModel):
    texts: list[str]
    target: str = "es-MX"


@app.on_event("startup")
async def startup():
    await cache.init()
    log.info("ai_service_started", extra={"model": MODEL, "db": DB_PATH, "redis": cache.redis_status()})


@app.on_event("shutdown")
async def shutdown():
    await cache.close()


# --- core: translate one string --------------------------------------------
async def translate_one(text: str, target: str) -> dict:
    """Translate a single string, using the cache first.

    Returns a dict shaped exactly like the widget expects:
        {"translated": str, "cached": bool, "latencyMs": int, "model": str}
    """
    text = (text or "").strip()
    if not text:
        return {"translated": "", "cached": False, "latencyMs": 0, "model": MODEL}

    t0 = time.perf_counter()

    cached_value = await cache.get(text, target)
    if cached_value is not None:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {"translated": cached_value, "cached": True, "latencyMs": latency_ms, "model": MODEL}

    translated = await translate_text(text, target, model=MODEL)
    await cache.set(text, target, translated, model=MODEL)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {"translated": translated, "cached": False, "latencyMs": latency_ms, "model": MODEL}


@app.post("/translate")
async def translate(body: TranslateIn, request: Request):
    result = await translate_one(body.text, body.target)
    log.info(
        "translate",
        extra={
            "requestId": request.headers.get("x-request-id"),
            "cached": result["cached"],
            "latencyMs": result["latencyMs"],
            "chars": len(body.text),
        },
    )
    return result


# --- single-flight coalescing for batch cache misses ------------------------
# Keyed by (target, normalize_text(text)): concurrent requests translating the
# same text — including casing/whitespace variants of each other — share one
# in-flight Future instead of each firing their own LLM call. Using the same
# normalize_text() as the cache key means this layer and the cache agree on
# what counts as "the same text": a batch of 5 casing variants of one novel
# phrase now costs exactly 1 LLM call, not up to 5. This also closes a
# contract gap — "identical (text, target) must never hit the LLM twice" —
# that broke once /translate/batch went concurrent.
_inflight: dict[tuple[str, str], "asyncio.Future[tuple[dict, bool]]"] = {}


async def _translate_miss(text: str, target: str, semaphore: asyncio.Semaphore, request: Request) -> tuple[dict, bool]:
    """Translate a confirmed cache miss: retry once, then fall back to the
    original text. Returns (result, should_cache) — should_cache is False for
    the fallback path so callers never persist untranslated text."""
    async with semaphore:
        t0 = time.perf_counter()
        try:
            translated = await translate_text(text, target, model=MODEL)
            should_cache = True
        except Exception as e:
            log.warning(
                "translate_item_failed",
                extra={"requestId": request.headers.get("x-request-id"), "error": str(e)},
            )
            try:
                translated = await translate_text(text, target, model=MODEL)
                should_cache = True
            except Exception as e2:
                log.warning(
                    "translate_item_retry_failed",
                    extra={"requestId": request.headers.get("x-request-id"), "error": str(e2)},
                )
                translated, should_cache = text, False
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {"translated": translated, "cached": False, "latencyMs": latency_ms, "model": MODEL}, should_cache


async def _translate_coalesced(text: str, target: str, semaphore: asyncio.Semaphore, request: Request) -> tuple[dict, bool]:
    key = (target, normalize_text(text))
    existing = _inflight.get(key)
    if existing is not None:
        return await existing

    future: "asyncio.Future[tuple[dict, bool]]" = asyncio.get_event_loop().create_future()
    _inflight[key] = future
    try:
        result_pair = await _translate_miss(text, target, semaphore, request)
    except Exception as exc:
        future.set_exception(exc)
        raise
    else:
        future.set_result(result_pair)
        return result_pair
    finally:
        _inflight.pop(key, None)


@app.post("/translate/batch")
async def translate_batch(body: BatchIn, request: Request):
    t0 = time.perf_counter()
    semaphore = asyncio.Semaphore(8)

    stripped = [(t or "").strip() for t in body.texts]
    non_empty = [t for t in stripped if t]

    # one batched cache lookup instead of one connect+SELECT per item
    cache_hits = await cache.get_many([(t, body.target) for t in non_empty]) if non_empty else {}

    misses = [t for t in non_empty if t not in cache_hits]
    # gather() calls _translate_coalesced once per occurrence, but duplicate
    # misses (same string twice in this batch, or a concurrent request
    # translating it right now) resolve via the shared future — one LLM call
    # per unique string, not per occurrence.
    miss_results = await asyncio.gather(
        *(_translate_coalesced(t, body.target, semaphore, request) for t in misses)
    ) if misses else []

    miss_lookup: dict[str, dict] = {}
    to_persist = []
    for t, (result, should_cache) in zip(misses, miss_results):
        miss_lookup[t] = result
        if should_cache:
            to_persist.append((t, body.target, result["translated"], result["model"]))

    # one batched write instead of one connect+UPSERT per item; fallbacks
    # (should_cache=False) are excluded so they can never poison the cache
    await cache.set_many(to_persist)

    results = []
    for t in stripped:
        if not t:
            results.append({"translated": "", "cached": False})
        elif t in cache_hits:
            results.append({"translated": cache_hits[t], "cached": True})
        else:
            results.append({"translated": miss_lookup[t]["translated"], "cached": False})

    latency = int((time.perf_counter() - t0) * 1000)
    hits = sum(1 for r in results if r["cached"])
    log.info(
        "translate_batch",
        extra={"requestId": request.headers.get("x-request-id"), "count": len(results), "hits": hits, "latencyMs": latency},
    )
    # widget expects {results: [{translated, cached}], latencyMs}
    return {"results": results, "latencyMs": latency}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "cacheSize": await cache.size(), "redis": cache.redis_status()}


@app.get("/stats")
async def stats():
    return await cache.stats()
