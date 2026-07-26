"""Teacher-model labeling for the qualitative half of the extraction schema.

Numeric fields come free and exact from XBRL (src/data/xbrl_facts.py). Only risk-factor
categories and summaries need a model, so this is the one place in the pipeline that costs money.

Claude Haiku 4.5 is the teacher: the job is high-volume, short-output classification over a fixed
taxonomy, where per-call cost dominates and the task sits well within a small model's range.
Structured outputs (`messages.parse`) constrain the response to the schema, so there is no
JSON-repair path to maintain.

The `Teacher` protocol keeps the provider swappable -- src/labels/build.py depends on the protocol,
not on Anthropic specifically.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from src.labels.schema import RISK_CATEGORIES, RiskFactor, RiskFactorList

logger = logging.getLogger(__name__)

TEACHER_MODEL = "claude-haiku-4-5"

# Only the risk-factor excerpt is sent: Item 1A is self-contained, and sending the whole filing
# would pay for tokens the task doesn't use.
SYSTEM_PROMPT = f"""You label risk factors disclosed in SEC 10-K filings.

Given the "Item 1A. Risk Factors" text of a filing, identify each distinct risk the company \
discloses and return it with a short title, one category, and a brief summary.

Rules:
- Use only these categories: {", ".join(RISK_CATEGORIES)}.
- Pick the single best-fitting category. If a risk spans several, choose the one the filing \
emphasizes most.
- Only include risks actually stated in the provided text. Do not infer risks the filing does not \
mention, and do not include risks from your background knowledge of the company.
- The text may be truncated mid-sentence. Ignore any trailing partial risk rather than guessing \
how it ends.
- Titles are under 12 words. Summaries are one or two sentences, specific to this company rather \
than generic boilerplate.
- Merge near-duplicate risks into one entry.
"""


class TeacherError(Exception):
    """Raised when labeling fails after retries. Caller should log and skip the filing."""


class TeacherUnavailableError(TeacherError):
    """Raised when the failure is permanent and affects every call -- exhausted API credit, an
    invalid key, an unknown model. The caller should abort the run rather than attempt the
    remaining filings, each of which would fail identically."""


class Teacher(Protocol):
    """Anything that can turn risk-factor text into structured risk factors."""

    def label_risk_factors(self, risk_text: str) -> list[RiskFactor]: ...


class AnthropicTeacher:
    """Claude-backed implementation of `Teacher`."""

    def __init__(self, model: str = TEACHER_MODEL, max_retries: int = 3) -> None:
        import anthropic

        # A bare client resolves credentials in order: ANTHROPIC_API_KEY, then ANTHROPIC_AUTH_TOKEN,
        # then an `ant auth login` profile on disk. An unset API key is therefore not an error --
        # only a total absence of credentials is, and the SDK reports that itself.
        # The SDK retries 429/5xx; `max_retries` here also covers schema-validation failures.
        try:
            self._client = anthropic.Anthropic()
        except Exception as e:
            raise SystemExit(
                f"Could not construct an Anthropic client ({e}). Set ANTHROPIC_API_KEY in .env, "
                "or run `ant auth login` to store a profile."
            ) from e
        self._model = model
        self._max_retries = max_retries

    def label_risk_factors(self, risk_text: str) -> list[RiskFactor]:
        import anthropic

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.parse(
                    model=self._model,
                    max_tokens=8_000,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": risk_text}],
                    output_format=RiskFactorList,
                )
                if response.stop_reason == "max_tokens":
                    # Truncated output would be a partial risk list masquerading as complete.
                    raise TeacherError("teacher response hit max_tokens; risk list is incomplete")
                return response.parsed_output.risk_factors
            except (
                anthropic.BadRequestError,
                anthropic.AuthenticationError,
                anthropic.PermissionDeniedError,
                anthropic.NotFoundError,
            ) as e:
                # Permanent: exhausted credit, bad key, unknown model. Retrying burns ~14s per
                # filing and cannot succeed -- surface it immediately with the API's own wording.
                raise TeacherUnavailableError(f"non-retryable API error: {e}") from e
            except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                last_exc = e
                delay = 2.0 * (2**attempt)
                logger.warning(
                    "teacher call failed (attempt %d/%d): %s -- retrying in %.1fs",
                    attempt + 1,
                    self._max_retries,
                    e,
                    delay,
                )
                time.sleep(delay)
            except Exception as e:  # schema-validation failures and anything else unexpected
                last_exc = e
                logger.warning(
                    "teacher response unusable (attempt %d/%d): %s", attempt + 1, self._max_retries, e
                )

        raise TeacherError(f"labeling failed after {self._max_retries} attempts") from last_exc
