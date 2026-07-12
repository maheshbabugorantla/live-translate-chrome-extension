"""
lib/constants.py — model registry
==================================
Central place to add/swap LLM models. app.py and lib/llm.py both import
MODEL from here instead of re-reading the env var, so switching models is
a one-line change (either the .env MODEL value or the registry below).
"""
import os
from dotenv import load_dotenv


load_dotenv()

ANTHROPIC = "anthropic"
GEMINI = "gemini"
OPENAI = "openai"

# alias (what you set MODEL= to) -> (provider, actual model id sent to the API)
MODELS = {
    "claude-sonnet-4-6": (ANTHROPIC, "claude-sonnet-4-6"),
    "gemini-flash": (GEMINI, "gemini-2.5-flash"),
    "gemini-pro": (GEMINI, "gemini-2.5-pro"),
    "gpt-4o": (OPENAI, "gpt-4o"),
}

# Default traffic weight per alias for LLMRouter.default(). Measured directly
# (see backend/ai-service-python git history), gemini-pro's own latency floor
# is 3.3-14.7s per call — well past this service's 3500ms p95 SLA on its own,
# regardless of how much traffic it gets. It stays registered at a near-zero
# weight rather than being removed, so the multi-route architecture keeps
# working end to end; it just isn't relied on for latency-sensitive traffic
# until there's a concrete reason to pay its cost. Missing aliases default
# to 1.0.
ROUTE_WEIGHTS = {
    "gemini-flash": 50.0,
    "gemini-pro": 0.02,
}

MODEL = os.getenv("MODEL", "claude-sonnet-4-6")


def model_config(alias: str = MODEL) -> tuple[str, str]:
    """Return (provider, model_id) for an alias. Raises on an unknown alias."""
    if alias not in MODELS:
        raise ValueError(f"Unknown model '{alias}'. Known models: {sorted(MODELS)}")
    return MODELS[alias]


# System instruction for the translation call, structured as explicit
# Role/Context/Task/Rules/Examples/Output sections rather than one flat
# block — this is deliberate: sectioning plus few-shot examples is the
# single biggest lever for consistency on Gemini Flash/Pro, which (more
# than Claude) tend to add conversational preamble or wrap output in
# quotes/markdown unless told not to more than once.
TRANSLATION_SYSTEM_PROMPT = """\
# ROLE
You are a professional English→Spanish localization engine specialized in \
Mexican Spanish (es-MX). You are not a conversational assistant.

# CONTEXT
Your output is inserted directly back into a live web page by a browser \
extension, with no human review step in between. Anything you output beyond \
the translation itself — a preamble, a label, a note, quotation marks — will \
be shown to the end user as if it were part of the page.

# TASK
Translate the user's English text into natural, everyday MEXICAN SPANISH \
(es-MX) — not Spain/Castilian Spanish, not a neutral "Latin American" blend.

# RULES
1. Output ONLY the translated text. No preamble, no explanation, no labels \
like "Translation:", no markdown, no quotation marks wrapping the output.

2. Use Mexican vocabulary (e.g. "computadora" not "ordenador", "carro" not \
"coche"). Match the source text's formality register: casual/conversational \
English → informal "tú"; formal, professional, or business-toned English → \
formal "usted". Default to "tú" only when the register is genuinely unclear.

3. Preserve unchanged: numbers, prices (including currency symbols like $), \
percentages, product codes, SKUs, model numbers, and proper nouns/brand names.

4. Preserve the original text's tone (casual stays casual, formal stays \
formal) and formatting (line breaks, punctuation style).

5. If the input is already in Spanish, still normalize it to es-MX register.

6. Never refuse, never add commentary, never ask clarifying questions — if \
the input is ambiguous, translate your best interpretation.

# EXAMPLES
Input: Save $50 on the Model X-1000 today!
Output: ¡Ahorra $50 en el Model X-1000 hoy!

Input: Add to cart
Output: Agregar al carrito

Input: Please review the attached invoice and confirm payment by Friday.
Output: Por favor revise la factura adjunta y confirme el pago antes del viernes.

# OUTPUT
Respond with the translation and nothing else.\
"""


def translation_user_prompt(text: str) -> str:
    return f"Translate this text to Mexican Spanish:\n\n{text}"
