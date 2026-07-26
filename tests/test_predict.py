"""Unit tests for src/eval/predict.py using a fake model -- no network, no GPU."""

from __future__ import annotations

import pytest

from src.eval import predict as predict_mod
from src.eval.predict import (
    PROMPT_MODES,
    Prediction,
    PredictionError,
    cache_path,
    estimate_cost,
    predict_dataset,
)
from tests.test_eval import make_example, make_target


class FakeModel:
    """Records every excerpt it is asked about, so cache hits are observable."""

    def __init__(self, response: str = "{}", stop_reason: str = "end_turn", system_prompt: str = "sys"):
        self.name = "fake-model"
        self.system_prompt = system_prompt
        self.calls: list[str] = []
        self._response = response
        self._stop_reason = stop_reason

    def predict(self, excerpt: str) -> Prediction:
        self.calls.append(excerpt)
        return Prediction(
            raw=self._response, stop_reason=self._stop_reason, input_tokens=100, output_tokens=50
        )


class FailingModel:
    name = "failing-model"
    system_prompt = "sys"

    def predict(self, excerpt: str) -> Prediction:
        raise PredictionError("simulated generation failure")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Never touch the real prediction cache from a test."""
    monkeypatch.setattr(predict_mod, "PREDICTION_CACHE", tmp_path / "predictions")


def test_predictions_are_cached_across_runs():
    target = make_target()
    examples = [make_example(target)]
    model = FakeModel(response=target.model_dump_json())

    first, first_stats = predict_dataset(examples, model)
    second, second_stats = predict_dataset(examples, model)

    assert first == second
    assert len(model.calls) == 1  # second run served entirely from cache
    assert first_stats["generated"] == 1 and first_stats["cached"] == 0
    assert second_stats["generated"] == 0 and second_stats["cached"] == 1


def test_cache_key_changes_when_the_excerpt_changes():
    """A changed prompt must miss the cache -- otherwise a model is scored on an input it never saw."""
    a = cache_path("fake-model", "0001-25-000001", "excerpt one", "sys")
    b = cache_path("fake-model", "0001-25-000001", "excerpt two", "sys")
    assert a != b


def test_cache_key_changes_with_the_system_prompt():
    """The two prompt modes exist to be compared; sharing a cache entry would conflate them."""
    a = cache_path("fake-model", "0001", "excerpt", "prose only")
    b = cache_path("fake-model", "0001", "excerpt", "prose plus schema")
    assert a != b


def test_cache_is_keyed_per_model():
    excerpt = "identical filing text"
    assert cache_path("model-a", "0001", excerpt, "sys") != cache_path("model-b", "0001", excerpt, "sys")


def test_prompt_modes_differ_and_schema_mode_names_the_fields():
    trained = PROMPT_MODES["trained"]()
    schema = PROMPT_MODES["schema"]()
    assert trained in schema and len(schema) > len(trained)
    # The zero-shot failure this mode exists to fix: inventing field names.
    for field in ("eps_diluted", "risk_factors", "category"):
        assert field in schema
        assert field not in trained


def test_no_cache_forces_regeneration():
    examples = [make_example()]
    model = FakeModel()

    predict_dataset(examples, model)
    predict_dataset(examples, model, use_cache=False)

    assert len(model.calls) == 2


def test_failed_generation_becomes_an_empty_prediction():
    """A model that cannot answer got the filing wrong -- it is scored, not skipped."""
    examples = [make_example(), make_example()]
    raws, stats = predict_dataset(examples, FailingModel())

    assert raws == ["", ""]
    assert stats["failed"] == 2
    assert len(raws) == len(examples)  # alignment with `examples` is what scoring depends on


def test_truncation_and_refusal_are_counted():
    truncated = FakeModel(stop_reason="max_tokens")
    _, stats = predict_dataset([make_example()], truncated)
    assert stats["truncated"] == 1

    refused = FakeModel(stop_reason="refusal")
    _, stats = predict_dataset([make_example()], refused, use_cache=False)
    assert stats["refused"] == 1


def test_prediction_round_trips_through_the_cache_format():
    original = Prediction(raw="{}", stop_reason="end_turn", input_tokens=10, output_tokens=5)
    assert Prediction.from_dict(original.to_dict()) == original


def test_estimate_cost():
    stats = {"input_tokens": 1_000_000, "output_tokens": 500_000}
    cost = estimate_cost(stats, {"input": 2.0, "output": 10.0})
    assert cost == pytest.approx(2.0 + 5.0)

    assert estimate_cost({"input_tokens": 0, "output_tokens": 0}) is None
