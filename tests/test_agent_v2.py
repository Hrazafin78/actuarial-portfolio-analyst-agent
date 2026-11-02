"""Tests des outils et de la boucle conversationnelle de l’agent.

Utilise des agrégats contrôlés et un faux client Responses API pour vérifier
les appels multiples et la continuité de conversation, sans requête réseau.
"""

import json
import sys
import types
from types import SimpleNamespace

import pandas as pd

# Le client est simulé : ce repli permet aussi de collecter les tests sans SDK installé.
try:
    from openai import OpenAI  # noqa: F401
except ImportError:
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **_kwargs: None
    sys.modules["openai"] = fake_openai

import agent_v2
from agent_tools_v2 import execute_tool_v2
from descriptive_analyzer import generate_descriptive_report


def _sample_report():
    """Construit un rapport agrégé de référence pour les appels d’outils simulés."""
    frame = pd.DataFrame({
        "id": ["P1", "P2", "P3", "P4"],
        "year": [2003, 2003, 2004, 2004],
        "claims": [0, 1, 0, 2],
        "expo": [0.5, 1.0, 1.0, 1.0],
        "premium": [100, 150, 100, 150],
        "cost": [0, 200, 0, 300],
        "fuel": ["diesel", "essence", "diesel", "essence"],
        "age": [20, 35, 29, 62],
    })
    return generate_descriptive_report(
        frame,
        roles={"year": "temporal", "fuel": "categorical", "age": "numeric"},
        actuarial_mapping={
            "policy_id": "id", "year": "year", "claim_count": "claims",
            "exposure": "expo", "premium": "premium", "claim_amount": "cost",
        },
    )


def test_tool_overview_returns_actual_aggregates():
    report = _sample_report()
    data = execute_tool_v2(report, "get_portfolio_overview", {})
    assert data["dataset"]["n_rows"] == 4
    assert data["actuarial_overview"]["total_claim_count"] == 3
    assert data["actuarial_overview"]["claim_frequency"] == 3 / 3.5
    json.dumps(data, allow_nan=False)


def test_tool_variable_and_segments():
    report = _sample_report()
    stats = execute_tool_v2(report, "get_variable_statistics", {"variable": "fuel"})
    assert stats["role"] == "categorical"
    assert stats["distribution"]["total_rows"] == 2
    segments = execute_tool_v2(report, "get_segment_analysis", {"variable": "fuel"})
    assert segments["total_rows"] == 2
    assert {row["segment"] for row in segments["rows"]} == {"diesel", "essence"}
    error = execute_tool_v2(report, "get_variable_statistics", {"variable": "inconnu"})
    assert error["error"] == "Variable inconnue"


def test_tool_temporal_and_whitelist():
    report = _sample_report()
    values = execute_tool_v2(report, "get_temporal_analysis", {"variable": "year"})
    assert values["total_rows"] == 2
    try:
        execute_tool_v2(report, "read_raw_records", {})
    except ValueError:
        pass
    else:
        raise AssertionError("L'outil inconnu doit être interdit")


def test_multiple_calls_and_conversation_chain(monkeypatch):
    report = _sample_report()
    monkeypatch.setenv("OPENAI_API_KEY", "test_key_fake")
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(
                    id="resp_a", output_text="",
                    output=[
                        SimpleNamespace(type="function_call", name="get_portfolio_overview",
                                        arguments="{}", call_id="call_1"),
                        SimpleNamespace(type="function_call", name="get_variable_statistics",
                                        arguments='{"variable":"fuel"}', call_id="call_2"),
                    ],
                )
            if len(calls) == 2:
                return SimpleNamespace(
                    id="resp_b", output_text="",
                    output=[SimpleNamespace(
                        type="function_call", name="get_temporal_analysis",
                        arguments='{"variable":"year"}', call_id="call_3",
                    )],
                )
            return SimpleNamespace(id="resp_c", output_text="Synthèse fondée sur les outils.", output=[])

    monkeypatch.setattr(agent_v2, "OpenAI", lambda **kwargs: SimpleNamespace(responses=FakeResponses()))

    answer, response_id = agent_v2.ask_agent_v2("Analyse le portefeuille", report)
    assert answer == "Synthèse fondée sur les outils."
    assert response_id == "resp_c"
    assert len(calls) == 3
    assert calls[1]["previous_response_id"] == "resp_a"
    assert calls[2]["previous_response_id"] == "resp_b"
    assert len(calls[1]["input"]) == 2
    assert json.loads(calls[1]["input"][0]["output"])["dataset"]["n_rows"] == 4
    assert json.loads(calls[2]["input"][0]["output"])["total_rows"] == 2
