"""
lib/llm.py — the LLM translation call
======================================
One job: turn an English string into Mexican Spanish using an LLM.

Structure:
  - BaseLlmApiClient   — provider-agnostic interface. Add Anthropic/OpenAI
                          later by subclassing; nothing else changes.
  - GeminiAPIClient     — the only concrete client today.
  - Route / SelectionStrategy / LLMRouter
                        — traffic routing across models. Only Gemini models
                          are registered right now, split by a simple
                          weighted-random strategy (weights come from
                          lib.constants.ROUTE_WEIGHTS). Quality-based
                          rebalancing is a real discipline (needs a defined
                          scoring methodology, not a stub) and isn't
                          attempted here — SelectionStrategy is the seam
                          where that would plug in later.

FAIL LOUD: do NOT wrap the call in a try/except that returns `text` on error.
If the provider fails, let the exception propagate so the caller returns a 502.
Silently returning the untranslated input is an automatic fail on this
assignment (and a real production bug — it ships English while looking healthy).
"""
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from google import genai
from google.genai import types

from lib.constants import GEMINI, MODEL, MODELS, ROUTE_WEIGHTS, TRANSLATION_SYSTEM_PROMPT, translation_user_prompt
from lib.logger import get_logger

log = get_logger("llm")


@dataclass
class Completion:
    """One LLM call's result plus its REAL billable token counts (for cost
    accounting — see lib/pricing.py). `output_tokens` is deliberately
    `total_token_count - prompt_token_count`, not just the visible
    candidate text's token count: Gemini 2.5 models spend internal
    "thinking" tokens that are billed but not exposed as their own field —
    they only show up as the gap between total and prompt+candidates.
    Folding them into output_tokens avoids undercounting real cost."""

    text: str
    input_tokens: int
    output_tokens: int


class BaseLlmApiClient(ABC):
    """Provider-agnostic interface. One method: give it a system/user prompt
    and a model id, get back clean text plus real token usage."""

    provider: str

    @abstractmethod
    async def complete(self, *, system: str, user: str, model_id: str, temperature: float = 0.2) -> Completion:
        ...

    @staticmethod
    def _clean(raw: str) -> str:
        """Strip whitespace and a wrapping pair of quotes/backticks the model
        may add despite being told not to (Gemini does this more than most)."""
        text = raw.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
            text = text[1:-1].strip()
        return text


class GeminiAPIClient(BaseLlmApiClient):
    provider = GEMINI

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set — cannot construct GeminiAPIClient")
        self._client = genai.Client(api_key=api_key)

    async def complete(self, *, system: str, user: str, model_id: str, temperature: float = 0.2) -> Completion:
        response = await self._client.aio.models.generate_content(
            model=model_id,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
            ),
        )
        if not response.text:
            raise RuntimeError(f"Gemini returned an empty response for model {model_id}")

        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        total_tokens = (usage.total_token_count or 0) if usage else 0
        # Everything that isn't prompt tokens — visible output plus any
        # hidden reasoning tokens — billed at the output rate.
        output_tokens = max(total_tokens - input_tokens, 0)

        return Completion(text=self._clean(response.text), input_tokens=input_tokens, output_tokens=output_tokens)


@dataclass
class Route:
    """One routable (client, model) pair plus its traffic weight, which is
    what WeightedRandomStrategy reads to pick a route."""

    client: BaseLlmApiClient
    alias: str
    model_id: str
    weight: float = 1.0


class SelectionStrategy(ABC):
    """How the router picks a Route for a given request. Swap this out to
    change routing behavior without touching LLMRouter or translate_text()."""

    @abstractmethod
    def choose(self, routes: list[Route]) -> Route:
        ...


class WeightedRandomStrategy(SelectionStrategy):
    """Pick a route at random, proportional to its `weight`."""

    def choose(self, routes: list[Route]) -> Route:
        if not routes:
            raise ValueError("WeightedRandomStrategy.choose() called with no routes")
        weights = [max(r.weight, 0.0) for r in routes]
        if sum(weights) <= 0:
            # All weights zero/negative (e.g. misconfigured ROUTE_WEIGHTS) —
            # fail safe with a uniform pick rather than taking translation
            # down entirely.
            weights = [1.0] * len(routes)
        return random.choices(routes, weights=weights, k=1)[0]


class LLMRouter:
    def __init__(self, routes: list[Route], strategy: SelectionStrategy | None = None):
        if not routes:
            raise ValueError("LLMRouter needs at least one route")
        self.routes = routes
        self.strategy = strategy or WeightedRandomStrategy()

    @classmethod
    def default(cls) -> "LLMRouter":
        """Build a router from every Gemini entry in the model registry,
        sharing one GeminiAPIClient, with starting weights from
        lib.constants.ROUTE_WEIGHTS (default 1.0 for any alias not listed)."""
        gemini_client = GeminiAPIClient()
        routes = [
            Route(client=gemini_client, alias=alias, model_id=model_id, weight=ROUTE_WEIGHTS.get(alias, 1.0))
            for alias, (provider, model_id) in MODELS.items()
            if provider == GEMINI
        ]
        if not routes:
            raise ValueError("No Gemini models registered in lib.constants.MODELS")
        return cls(routes)

    async def complete(self, *, system: str, user: str) -> tuple[Completion, str, str]:
        """Returns (completion, alias, model_id) — model_id (e.g.
        "gemini-2.5-flash") is what lib.pricing.PricingStore keys on;
        alias (e.g. "gemini-flash") is what's logged/reported today."""
        route = self.strategy.choose(self.routes)
        log.info("llm_route_selected", extra={"model": route.alias})
        completion = await route.client.complete(system=system, user=user, model_id=route.model_id)
        return completion, route.alias, route.model_id


_router: LLMRouter | None = None


def _get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter.default()
    return _router


@dataclass
class TranslationResult:
    """What translate_text() hands back to app.py — translated text plus
    everything lib.pricing.PricingStore.cost_usd() needs (model_id, real
    token counts) to compute actual dollar cost for this call."""

    translated: str
    model_id: str
    input_tokens: int
    output_tokens: int


async def translate_text(text: str, target: str = "es-MX", model: str = MODEL) -> TranslationResult:
    """Return `text` translated into `target` (Mexican Spanish by default),
    plus the model id and real token counts actually used for this call.

    Routing (which Gemini model actually handles the request) is decided by
    LLMRouter, not by the `model` argument — `model` is accepted for API
    compatibility with callers/tests but the router owns model selection.
    """
    completion, used_alias, model_id = await _get_router().complete(
        system=TRANSLATION_SYSTEM_PROMPT,
        user=translation_user_prompt(text),
    )
    log.info(
        "translate_text",
        extra={
            "model": used_alias,
            "chars": len(text),
            "inputTokens": completion.input_tokens,
            "outputTokens": completion.output_tokens,
        },
    )
    return TranslationResult(
        translated=completion.text,
        model_id=model_id,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )
