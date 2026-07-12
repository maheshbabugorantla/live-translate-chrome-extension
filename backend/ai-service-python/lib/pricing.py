"""
lib/pricing.py — live per-token pricing for real cost/savings accounting
=====================================================================
Pulls current USD-per-token prices for our routed models from litellm's
community-maintained pricing catalog, caches them in Redis (shared across
replicas, like lib/cache.py's translation cache), and refreshes them on a
background thread that only rewrites Redis when the prices actually change
(sha256 checksum of the extracted subset — not the whole 1.6MB upstream
file, so unrelated provider churn never triggers a spurious update).

Fully fail-open, matching lib/cache.py's philosophy: if Redis is unset or
unreachable, or the upstream fetch fails, cost math keeps working from the
static fallback below or the last successfully fetched prices — a pricing
outage must never break translation.

Why a background THREAD and not an asyncio task: refreshing means an HTTP
fetch of a multi-MB JSON file plus parsing it — blocking I/O/CPU work that
would stall the event loop for every in-flight request if it ran as a
coroutine. Same principle docs/DESIGN.md §5.4 states for CPU-bound work:
keep it off the loop, in a thread.
"""
import hashlib
import json
import logging
import threading
import time
import urllib.request

try:
    import redis as redis_sync  # plain (non-asyncio) client — usable from a thread
except ImportError:  # pragma: no cover - redis is an optional dependency
    redis_sync = None

from lib.constants import MODELS
from lib.logger import get_logger

log = get_logger("pricing")

LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/litellm_internal_staging/"
    "model_prices_and_context_window.json"
)
REDIS_PRICING_KEY = "xlate:pricing:models"
REDIS_CHECKSUM_KEY = "xlate:pricing:checksum"
DEFAULT_REFRESH_INTERVAL_SECONDS = 6 * 60 * 60  # 6 hours — prices don't change often
FETCH_TIMEOUT_SECONDS = 30

# Snapshotted from litellm's catalog and confirmed against
# benchmark/sla.gemini.json ($0.30 / $2.50 per Mtok == 3e-07 / 2.5e-06 per
# token). Used whenever Redis is unavailable and no live fetch has succeeded.
STATIC_FALLBACK_PRICES = {
    "gemini-2.5-flash": {"input_cost_per_token": 3e-07, "output_cost_per_token": 2.5e-06},
    "gemini-2.5-pro": {"input_cost_per_token": 1.25e-06, "output_cost_per_token": 1e-05},
}

# The only model ids we ever actually need pricing for — every model id
# referenced by lib/constants.MODELS, so this stays correct if more routes
# are added later without hardcoding model names here.
KNOWN_MODEL_IDS = sorted({model_id for _, model_id in MODELS.values()})


def _canonical_subset_json(prices: dict) -> str:
    """Deterministic serialization so the checksum only reflects OUR models'
    actual prices, not upstream key ordering or unrelated provider entries."""
    return json.dumps(prices, sort_keys=True)


def _extract_known_prices(full_catalog: dict) -> dict:
    extracted = {}
    for model_id in KNOWN_MODEL_IDS:
        entry = full_catalog.get(model_id)
        if not entry:
            continue
        input_cost = entry.get("input_cost_per_token")
        output_cost = entry.get("output_cost_per_token")
        if input_cost is None or output_cost is None:
            continue
        extracted[model_id] = {"input_cost_per_token": input_cost, "output_cost_per_token": output_cost}
    return extracted


class PricingStore:
    def __init__(self, redis_url: str | None = None, refresh_interval_seconds: int = DEFAULT_REFRESH_INTERVAL_SECONDS):
        self.redis_url = redis_url
        self.refresh_interval_seconds = refresh_interval_seconds
        self._prices: dict[str, dict] = dict(STATIC_FALLBACK_PRICES)
        self._lock = threading.Lock()
        self._source = "fallback"  # "fallback" | "redis" | "live-fetch"
        self._last_updated: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._totals = {"total_cost_usd": 0.0, "total_savings_usd": 0.0, "priced_requests": 0}
        self._totals_lock = threading.Lock()

        # Best-effort initial load from Redis so a fresh process benefits
        # from whatever another replica (or this process's prior run)
        # already fetched, without waiting for the first refresh tick.
        self._seed_from_redis()

    def _sync_redis_client(self):
        if not self.redis_url or redis_sync is None:
            return None
        try:
            client = redis_sync.from_url(self.redis_url, decode_responses=True, socket_timeout=5)
            client.ping()
            return client
        except Exception as e:
            log.warning("pricing_redis_connect_failed", extra={"error": str(e)})
            return None

    def _seed_from_redis(self) -> None:
        client = self._sync_redis_client()
        if client is None:
            return
        try:
            raw = client.get(REDIS_PRICING_KEY)
            if raw:
                with self._lock:
                    self._prices.update(json.loads(raw))
                    self._source = "redis"
                    self._last_updated = time.time()
                log.info("pricing_seeded_from_redis", extra={"models": list(json.loads(raw).keys())})
        except Exception as e:
            log.warning("pricing_seed_failed", extra={"error": str(e)})
        finally:
            try:
                client.close()
            except Exception:
                pass

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._refresh_loop, name="pricing-refresh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _refresh_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._fetch_and_maybe_update()
            except Exception as e:
                log.warning("pricing_refresh_failed", extra={"error": str(e)})
            # interruptible sleep so stop() doesn't wait a full interval
            self._stop_event.wait(self.refresh_interval_seconds)

    def _fetch_and_maybe_update(self) -> None:
        with urllib.request.urlopen(LITELLM_PRICING_URL, timeout=FETCH_TIMEOUT_SECONDS) as resp:
            full_catalog = json.loads(resp.read())

        extracted = _extract_known_prices(full_catalog)
        if not extracted:
            log.warning("pricing_fetch_no_known_models", extra={"looked_for": KNOWN_MODEL_IDS})
            return

        canonical = _canonical_subset_json(extracted)
        checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        client = self._sync_redis_client()
        if client is not None:
            try:
                existing_checksum = client.get(REDIS_CHECKSUM_KEY)
                if existing_checksum == checksum:
                    log.info("pricing_unchanged", extra={"checksum": checksum[:12]})
                    with self._lock:
                        self._prices.update(extracted)
                        self._source = "redis"
                        self._last_updated = time.time()
                    return
                pipe = client.pipeline()
                pipe.set(REDIS_PRICING_KEY, canonical)
                pipe.set(REDIS_CHECKSUM_KEY, checksum)
                pipe.execute()
                log.info("pricing_updated", extra={"checksum": checksum[:12], "models": list(extracted.keys())})
            except Exception as e:
                log.warning("pricing_redis_persist_failed", extra={"error": str(e)})
            finally:
                try:
                    client.close()
                except Exception:
                    pass

        with self._lock:
            self._prices.update(extracted)
            self._source = "live-fetch"
            self._last_updated = time.time()

    def get_price(self, model_id: str) -> dict:
        with self._lock:
            return self._prices.get(model_id) or STATIC_FALLBACK_PRICES.get(
                model_id, {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
            )

    def cost_usd(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        price = self.get_price(model_id)
        return input_tokens * price["input_cost_per_token"] + output_tokens * price["output_cost_per_token"]

    def record(self, cost_usd: float, savings_usd: float) -> None:
        with self._totals_lock:
            self._totals["total_cost_usd"] += cost_usd
            self._totals["total_savings_usd"] += savings_usd
            self._totals["priced_requests"] += 1

    def totals(self) -> dict:
        with self._totals_lock:
            return dict(self._totals)

    def status(self) -> dict:
        with self._lock:
            return {"source": self._source, "lastUpdated": self._last_updated}
