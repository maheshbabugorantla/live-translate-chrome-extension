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
from lib.pricing import PricingStore

load_dotenv()

DB_PATH = os.getenv("TRANSLATION_DB_PATH", "translations.db")
REDIS_URL = os.getenv("REDIS_URL")  # optional — shared cache tier across replicas

app = FastAPI(title="FDE Live Translate — AI Service")
log = get_logger("ai-service")
cache = TwoTierCache(DB_PATH, redis_url=REDIS_URL)
pricing = PricingStore(redis_url=REDIS_URL)


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
    pricing.start()
    log.info("ai_service_started", extra={"model": MODEL, "db": DB_PATH, "redis": cache.redis_status(), "pricing": pricing.status()})


@app.on_event("shutdown")
async def shutdown():
    await cache.close()
    pricing.stop()


# --- core: translate one string --------------------------------------------
async def translate_one(text: str, target: str) -> dict:
    """Translate a single string, using the cache first.

    Returns a dict shaped exactly like the widget expects, plus two additive
    fields for real cost accounting:
        {"translated": str, "cached": bool, "latencyMs": int, "model": str,
         "costUsd": float, "savingsUsd": float}
    `model` now reports the REAL model that produced this translation (e.g.
    "gemini-2.5-flash"), not the static MODEL env var — needed for accurate
    per-model pricing, and incidentally more honest than before.
    """
    text = (text or "").strip()
    if not text:
        return {"translated": "", "cached": False, "latencyMs": 0, "model": MODEL, "costUsd": 0.0, "savingsUsd": 0.0}

    t0 = time.perf_counter()

    cached_entry = await cache.get(text, target)
    if cached_entry is not None:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        savings_usd = pricing.cost_usd(cached_entry.model_id, cached_entry.input_tokens, cached_entry.output_tokens)
        pricing.record(cost_usd=0.0, savings_usd=savings_usd)
        return {
            "translated": cached_entry.translated,
            "cached": True,
            "latencyMs": latency_ms,
            "model": cached_entry.model_id or MODEL,
            "costUsd": 0.0,
            "savingsUsd": savings_usd,
        }

    result = await translate_text(text, target, model=MODEL)
    cost_usd = pricing.cost_usd(result.model_id, result.input_tokens, result.output_tokens)
    pricing.record(cost_usd=cost_usd, savings_usd=0.0)
    await cache.set(
        text, target, result.translated, model=result.model_id,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "translated": result.translated,
        "cached": False,
        "latencyMs": latency_ms,
        "model": result.model_id,
        "costUsd": cost_usd,
        "savingsUsd": 0.0,
    }


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
            "costUsd": result["costUsd"],
            "savingsUsd": result["savingsUsd"],
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
    the fallback path so callers never persist untranslated text.

    Note: a failed attempt's real token spend (if any was billed before the
    provider returned an empty/error response) isn't captured here — the
    RuntimeError is raised before a Completion/TranslationResult exists. A
    known, rare edge case; not worth the complexity to chase for a retry path.
    """
    async with semaphore:
        t0 = time.perf_counter()
        model_id, input_tokens, output_tokens = "", 0, 0
        try:
            result = await translate_text(text, target, model=MODEL)
            translated, model_id, input_tokens, output_tokens = result.translated, result.model_id, result.input_tokens, result.output_tokens
            should_cache = True
        except Exception as e:
            log.warning(
                "translate_item_failed",
                extra={"requestId": request.headers.get("x-request-id"), "error": str(e)},
            )
            try:
                result = await translate_text(text, target, model=MODEL)
                translated, model_id, input_tokens, output_tokens = result.translated, result.model_id, result.input_tokens, result.output_tokens
                should_cache = True
            except Exception as e2:
                log.warning(
                    "translate_item_retry_failed",
                    extra={"requestId": request.headers.get("x-request-id"), "error": str(e2)},
                )
                translated, should_cache = text, False
        latency_ms = int((time.perf_counter() - t0) * 1000)
        cost_usd = pricing.cost_usd(model_id, input_tokens, output_tokens) if should_cache else 0.0
        if should_cache:
            pricing.record(cost_usd=cost_usd, savings_usd=0.0)
        return {
            "translated": translated,
            "cached": False,
            "latencyMs": latency_ms,
            "model": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "costUsd": cost_usd,
            "savingsUsd": 0.0,
        }, should_cache


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
            to_persist.append((t, body.target, result["translated"], result["model"], result["input_tokens"], result["output_tokens"]))

    # one batched write instead of one connect+UPSERT per item; fallbacks
    # (should_cache=False) are excluded so they can never poison the cache
    await cache.set_many(to_persist)

    results = []
    total_cost_usd = 0.0
    total_savings_usd = 0.0
    # Duplicate misses in this batch share ONE real LLM call — via single-
    # flight coalescing keyed on (target, normalize_text(text)), same as
    # _inflight above, which means CASING/WHITESPACE variants of one phrase
    # collapse too, not just byte-identical repeats. Each occurrence still
    # gets its own result entry carrying that shared call's costUsd, so
    # summing costUsd per OCCURRENCE would overcount a cost that was only
    # actually incurred once. Dedupe on the SAME key single-flight uses, so
    # each unique underlying LLM call is counted exactly once toward the
    # total; per-item `costUsd` still shows the true cost for anyone
    # inspecting that occurrence. Cache hits don't need this: every hit is a
    # genuinely separate avoided LLM call, so summing savings per occurrence
    # is correct.
    counted_cost_keys: set[tuple[str, str]] = set()
    for t in stripped:
        if not t:
            results.append({"translated": "", "cached": False, "costUsd": 0.0, "savingsUsd": 0.0})
        elif t in cache_hits:
            entry = cache_hits[t]
            savings_usd = pricing.cost_usd(entry.model_id, entry.input_tokens, entry.output_tokens)
            pricing.record(cost_usd=0.0, savings_usd=savings_usd)
            total_savings_usd += savings_usd
            results.append({"translated": entry.translated, "cached": True, "costUsd": 0.0, "savingsUsd": savings_usd})
        else:
            r = miss_lookup[t]
            cost_key = (body.target, normalize_text(t))
            if cost_key not in counted_cost_keys:
                total_cost_usd += r["costUsd"]
                counted_cost_keys.add(cost_key)
            results.append({"translated": r["translated"], "cached": False, "costUsd": r["costUsd"], "savingsUsd": 0.0})

    latency = int((time.perf_counter() - t0) * 1000)
    hits = sum(1 for r in results if r["cached"])
    log.info(
        "translate_batch",
        extra={
            "requestId": request.headers.get("x-request-id"),
            "count": len(results),
            "hits": hits,
            "latencyMs": latency,
            "totalCostUsd": total_cost_usd,
            "totalSavingsUsd": total_savings_usd,
        },
    )
    # widget expects {results: [{translated, cached, costUsd, savingsUsd}], latencyMs, totalCostUsd, totalSavingsUsd}
    return {"results": results, "latencyMs": latency, "totalCostUsd": total_cost_usd, "totalSavingsUsd": total_savings_usd}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "cacheSize": await cache.size(),
        "redis": cache.redis_status(),
        "pricing": pricing.status(),
    }


@app.get("/stats")
async def stats():
    return {**await cache.stats(), **pricing.totals()}
