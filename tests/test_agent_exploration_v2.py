"""Tests de consultation ciblée des résultats statistiques descriptifs.

Vérifie catalogue, associations, seuils, tri avant pagination et limites
des distributions, sans modifier les résultats ni appeler l’API.
"""
import json

import pandas as pd
import pytest

from agent_tools_v2 import TOOLS_V2, execute_tool_v2
from descriptive_analyzer import generate_descriptive_report


@pytest.fixture
def report():
    """Construit une analyse contrôlée avec associations et segments comparables."""
    return generate_descriptive_report(
        pd.DataFrame({
            "age": [20, 30, 40, 50, 60, 70],
            "power": [12, 10, 8, 6, 4, 2],
            "fuel": ["A", "A", "B", "B", "C", "C"],
            "expo": [1., 1., 2., 2., 1., 1.],
            "claims": [0, 0, 1, 1, 3, 3],
        }),
        roles={"age": "numeric", "power": "numeric", "fuel": "categorical"},
        actuarial_mapping={"exposure": "expo", "claim_count": "claims"},
    )


def test_catalog_lists_all_variables_and_limits(report):
    result = execute_tool_v2(report, "list_available_analyses", {})
    assert set(result["variables"]) == set(report["roles"])
    assert "claim_frequency" in result["segments"]["fuel"]
    assert result["limits"]
    json.dumps(result, allow_nan=False)


def test_exact_association_in_both_directions(report):
    for a, b in [("age", "power"), ("power", "age")]:
        result = execute_tool_v2(report, "get_associations", {"variable": a, "other_variable": b})
        assert result["total_rows"] == 1
        assert result["rows"][0]["coefficient"] == pytest.approx(-1)
    unavailable = execute_tool_v2(report, "get_associations", {"variable": "age", "other_variable": "expo"})
    assert unavailable["total_rows"] == 0
    assert "indisponible" in unavailable["note"]


def test_variable_associations_beyond_first_page(report):
    first = execute_tool_v2(report, "get_associations", {"variable": "age", "limit": 1})
    second = execute_tool_v2(report, "get_associations", {"variable": "age", "limit": 1, "offset": first["next_offset"]})
    assert first["truncated"]
    assert not second["truncated"]
    assert first["rows"] != second["rows"]


def test_sort_and_threshold_before_pagination_without_mutation(report):
    before = report["segments"]["fuel"].copy(deep=True)
    result = execute_tool_v2(report, "get_segment_analysis", {
        "variable": "fuel", "sort_by": "claim_frequency", "limit": 1,
    })
    assert result["rows"][0]["segment"] == "C"
    assert result["rows"][0]["claim_frequency"] == 3
    threshold = execute_tool_v2(report, "get_segment_analysis", {
        "variable": "fuel", "min_exposure": 3, "sort_by": "claim_frequency",
    })
    assert [r["segment"] for r in threshold["rows"]] == ["B"]
    assert threshold["rows"][0]["share_observations_pct"] == pytest.approx(100 / 3)
    pd.testing.assert_frame_equal(before, report["segments"]["fuel"])
    json.dumps(result, allow_nan=False)


def test_compare_specific_labels_and_empty_selection(report):
    result = execute_tool_v2(report, "get_segment_analysis", {"variable": "fuel", "segments": ["B", "C"]})
    assert {r["segment"] for r in result["rows"]} == {"B", "C"}
    empty = execute_tool_v2(report, "get_segment_analysis", {"variable": "fuel", "min_observations": 100})
    assert empty["rows"] == []
    assert empty["source_rows"] == 3


@pytest.mark.parametrize("args", [
    {"sort_by": "invented"}, {"segments": ["absent"]},
    {"min_exposure": -1}, {"min_exposure": float("inf")},
    {"min_observations": 1.5}, {"limit": 0}, {"limit": 101},
    {"offset": -1}, {"offset": True},
])
def test_reject_invalid_queries(report, args):
    with pytest.raises(ValueError):
        execute_tool_v2(report, "get_segment_analysis", {"variable": "fuel", **args})


def test_schema_strict_required_parameters():
    for tool in TOOLS_V2:
        schema = tool["parameters"]
        assert set(schema.get("required", [])) == set(schema["properties"])
        assert schema["additionalProperties"] is False


def test_distribution_pagination_signals_aggregation(report):
    result = execute_tool_v2(report, "get_variable_statistics", {"variable": "fuel", "limit": 1})
    assert result["distribution"]["next_offset"] == 1
    assert "Autres" in result["distribution_note"]
