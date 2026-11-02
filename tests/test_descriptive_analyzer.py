"""Tests du moteur statistique et actuariel sur des données contrôlées.

Vérifie rôles, mapping, ratios, segments, associations, temporalité et
sérialisation du contexte IA, sans appel à un modèle.
"""

import json
import numpy as np
import pandas as pd
import pytest

from descriptive_analyzer import (
    AnalysisOptions,
    analyze_categorical_variables,
    analyze_dataset,
    analyze_numeric_variables,
    analyze_portfolio_segments,
    analyze_relationships,
    analyze_temporal_variables,
    build_llm_context,
    detect_variable_roles,
    generate_descriptive_report,
    validate_actuarial_mapping,
)


@pytest.fixture
def portfolio():
    """Fournit un portefeuille contrôlé avec mesures, années et variables explicatives."""
    return pd.DataFrame({
        "insured_id": [1, 2, 3, 4, 5, 6],
        "year": [2023, 2023, 2023, 2024, 2024, 2024],
        "age": [20, 30, 40, 50, 60, 70],
        "years_licence": [2, 12, 22, 32, 42, 52],
        "fuel": ["D", "D", "E", "E", "D", None],
        "brand": ["A", "A", "B", "B", "A", "B"],
        "claims": [1, 0, 0, 2, 1, 0],
        "exposure": [1.0, 1.0, 0.5, 1.0, 0.5, 1.0],
        "premium": [100, 200, 100, 200, 100, 100],
        "incurred": [300, 0, 0, 200, 100, 0],
    })


@pytest.fixture
def mapping():
    """Associe les colonnes du portefeuille de test aux rôles actuariels attendus."""
    return {"claim_count": "claims", "exposure": "exposure",
            "premium": "premium", "claim_amount": "incurred", "year": "year",
            "policy_id": "insured_id"}


def test_roles_and_overrides(portfolio):
    roles = detect_variable_roles(portfolio)
    assert roles["insured_id"] == "identifier"
    assert roles["year"] == "temporal"
    assert roles["age"] == "numeric"
    assert roles["fuel"] == "categorical"
    assert detect_variable_roles(portfolio, {"age": "categorical"})["age"] == "categorical"


def test_reject_invalid_role_and_missing_column(portfolio):
    with pytest.raises(ValueError):
        detect_variable_roles(portfolio, {"age": "alien"})
    with pytest.raises(ValueError):
        detect_variable_roles(portfolio, {"missing": "numeric"})


def test_overview(portfolio):
    info = analyze_dataset(portfolio)
    assert info["n_rows"] == 6
    assert info["n_columns"] == 10
    assert info["total_missing"] == 1
    assert info["role_counts"]["identifier"] == 1


def test_numerical_stats_are_correct(portfolio):
    roles = detect_variable_roles(portfolio)
    table = analyze_numeric_variables(portfolio, roles)
    age = table.set_index("variable").loc["age"]
    assert age["mean"] == 45
    assert age["median"] == 45
    assert age["min"] == 20
    assert age["max"] == 70


def test_categorical_distributions_and_missing(portfolio):
    stats, distributions = analyze_categorical_variables(portfolio, detect_variable_roles(portfolio), 2)
    assert stats.set_index("variable").loc["fuel", "missing"] == 1
    assert distributions["fuel"]["effectif"].sum() == 6
    assert "insured_id" not in distributions


def test_mapping_validations(portfolio, mapping):
    assert validate_actuarial_mapping(portfolio, mapping)["exposure"] == "exposure"
    with pytest.raises(ValueError):
        validate_actuarial_mapping(portfolio, {"claim_count": "nonsense"})
    with pytest.raises(ValueError):
        validate_actuarial_mapping(portfolio, {"claim_count": "claims", "exposure": "claims"})
    bad = portfolio.copy(); bad.loc[0, "claims"] = -1
    with pytest.raises(ValueError):
        validate_actuarial_mapping(bad, mapping)
    bad = portfolio.copy(); bad["claims"] = bad["claims"].astype(float); bad.loc[0, "claims"] = 1.5
    with pytest.raises(ValueError):
        validate_actuarial_mapping(bad, mapping)


def test_global_metrics(portfolio, mapping):
    report = generate_descriptive_report(portfolio, actuarial_mapping=mapping)
    overview = report["actuarial_overview"]
    assert overview["total_claim_count"] == 4
    assert overview["total_exposure"] == 5
    assert overview["total_premium"] == 800
    assert overview["total_claim_amount"] == 600
    assert overview["claim_frequency"] == pytest.approx(0.8)
    assert overview["average_severity"] == pytest.approx(150)
    assert overview["loss_ratio"] == pytest.approx(0.75)
    assert overview["pure_premium"] == pytest.approx(120)


def test_segments_frequency_and_shares(portfolio, mapping):
    roles = detect_variable_roles(portfolio)
    segments, warnings = analyze_portfolio_segments(portfolio, roles, mapping)
    fuel = segments["fuel"].set_index("segment")
    assert fuel.loc["D", "total_exposure"] == pytest.approx(2.5)
    assert fuel.loc["D", "total_claim_count"] == pytest.approx(2)
    assert fuel.loc["D", "claim_frequency"] == pytest.approx(0.8)
    assert fuel.loc["(Manquant)", "observations"] == 1
    assert fuel["share_observations_pct"].sum() == pytest.approx(100)
    assert "insured_id" not in segments
    assert "claims" not in segments


def test_temporal(portfolio, mapping):
    tables, warnings = analyze_temporal_variables(portfolio, detect_variable_roles(portfolio), mapping)
    years = tables["year"].set_index("segment")
    assert years.loc["2023", "observations"] == 3
    assert years.loc["2024", "total_claim_count"] == 3
    assert years.loc["2024", "claim_frequency"] == pytest.approx(3/2.5)


def test_pearson_identifies_perfect_correlation(portfolio):
    result = analyze_relationships(portfolio, detect_variable_roles(portfolio))
    assert result["pearson"].loc["age", "years_licence"] == pytest.approx(1)
    assert "insured_id" not in result["pearson"]
    assert "year" not in result["pearson"]


def test_cramers_strong_association():
    data = pd.DataFrame({"cat_a": ["x"]*100+["y"]*100,
                         "cat_b": ["a"]*100+["b"]*100})
    result = analyze_relationships(data, detect_variable_roles(data))
    assert result["cramers_v"].loc["cat_a", "cat_b"] > 0.98


def test_eta_squared_perfect_association():
    data = pd.DataFrame({"cat": ["x"]*100+["y"]*100,
                         "value": [10.]*100+[20.]*100})
    result = analyze_relationships(data, detect_variable_roles(data))
    assert result["eta_squared"].loc["cat", "value"] == pytest.approx(1)


def test_high_cardinality_explicitly_skipped():
    data = pd.DataFrame({"cat": [f"c{i}" for i in range(100)],
                         "other": ["a", "b"]*50})
    rel = analyze_relationships(data, detect_variable_roles(data), max_categories=80)
    assert rel["cramers_v"].shape == (0, 0) or "cat" not in rel["cramers_v"].columns
    assert any("cat : 100 modalités" in warn for warn in rel["skipped"])


def test_constant_and_missing_variables():
    data = pd.DataFrame({"v": [1, 1, 1, np.nan], "w": [2, 3, 4, 5],
                         "category": ["A", "A", "A", "A"]})
    report = generate_descriptive_report(data)
    assert any("v : variable numérique constante" in msg for msg in report["warnings"])
    assert report["associations"]["pearson"].shape == (1, 1)


def test_no_exposure_does_not_invent_frequency(portfolio):
    report = generate_descriptive_report(portfolio, actuarial_mapping={"claim_count": "claims"})
    assert "claim_frequency" not in report["actuarial_overview"]
    assert any("Fréquence indisponible" in x for x in report["warnings"])


def test_zero_denominator_does_not_produce_infinite_rate(portfolio):
    df = portfolio.copy(); df["exposure"] = 0
    report = generate_descriptive_report(df, actuarial_mapping={"claim_count": "claims", "exposure": "exposure"})
    assert report["actuarial_overview"]["claim_frequency"] is None
    assert report["segments"]["fuel"]["claim_frequency"].isna().all()


def test_input_unchanged(portfolio, mapping):
    before = portfolio.copy(deep=True)
    generate_descriptive_report(portfolio, actuarial_mapping=mapping)
    pd.testing.assert_frame_equal(before, portfolio)


def test_llm_context_is_json_serializable_and_limited(portfolio, mapping):
    report = generate_descriptive_report(portfolio, actuarial_mapping=mapping)
    ctx = build_llm_context(report, 2)
    assert ctx["dataset"]["n_rows"] == 6
    assert len(ctx["numeric_summary"]) <= 2
    assert "segment_examples" in ctx
    json.dumps(ctx, ensure_ascii=False, allow_nan=False)


def test_empty_dataframe_and_numerical_only():
    df = pd.DataFrame({"value": pd.Series(dtype="float64")})
    r = generate_descriptive_report(df)
    assert r["overview"]["n_rows"] == 0
    assert "claim_frequency" not in r["actuarial_overview"]


def test_quantile_bins_for_numeric_variable(portfolio, mapping):
    result = generate_descriptive_report(portfolio, actuarial_mapping=mapping,
                                         options=AnalysisOptions(numerical_bins=3))
    age = result["segments"]["age"]
    assert age["segmentation"].eq("quantiles").all()
    assert age["observations"].sum() == 6


def test_invalid_dataframe_columns():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError):
        generate_descriptive_report(df)


def test_custom_role_code_to_categorical(portfolio, mapping):
    df = portfolio.assign(area_code=[1, 1, 1, 2, 2, 2])
    report = generate_descriptive_report(df, roles={"area_code": "categorical"}, actuarial_mapping=mapping)
    assert "area_code" in report["associations"]["cramers_v"].columns
    assert "area_code" in report["segments"]


def test_missing_numeric_mapping_is_rejected(portfolio):
    bad = portfolio.copy()
    bad.loc[0,"exposure"] = np.nan
    with pytest.raises(ValueError):
        validate_actuarial_mapping(bad, {"exposure":"exposure"})


def test_pair_numeric_numeric_sample(portfolio):
    from descriptive_analyzer import analyze_variable_pair
    detail = analyze_variable_pair(portfolio, "age", "years_licence", detect_variable_roles(portfolio), scatter_sample=3)
    assert detail["correlation"] == pytest.approx(1)
    assert len(detail["plot_data"]) == 3
    assert detail["observations"] == 6


def test_pair_categorical_categorical(portfolio):
    from descriptive_analyzer import analyze_variable_pair
    detail = analyze_variable_pair(portfolio, "fuel", "brand", detect_variable_roles(portfolio))
    assert detail["counts"].values.sum() == 6
    assert (detail["row_pct"].sum(axis=1).round(3) == 100).all()


def test_pair_mixed(portfolio):
    from descriptive_analyzer import analyze_variable_pair
    detail = analyze_variable_pair(portfolio, "fuel", "age", detect_variable_roles(portfolio))
    assert detail["statistics"].loc[detail["statistics"]["modalite"] == "D", "count"].iloc[0] == 3


def test_pair_rejects_same_variable(portfolio):
    from descriptive_analyzer import analyze_variable_pair
    with pytest.raises(ValueError):
        analyze_variable_pair(portfolio, "age", "age", detect_variable_roles(portfolio))
