"""Tests du diagnostic de qualité sur de petits DataFrames contrôlés.

Vérifie les sections du rapport, les anomalies attendues et les cas limites,
sans correction des données ni accès réseau.
"""

import pandas as pd
import pytest
import json

from quality_checker import (
    check_structure,
    check_missing_values,
    check_duplicates,
    check_data_types,
    check_outliers,
    check_actuarial_consistency,
    check_statistical_issues,
    generate_quality_report
)


def create_test_dataframe():
    """
    Crée une petite base fictive contenant
    volontairement des anomalies.
    """

    return pd.DataFrame({
        "insured_id": [1, 2, 3, 3],
        "year": [2023, 2023, 2024, 2024],
        "driver_age": [25, 40, None, None],
        "total_premium": [500, 700, 600, 600],
        "total_claims": [0, 1, 0, 0]
    })


def test_structure():
    df = create_test_dataframe()

    result = check_structure(df)

    assert result["n_rows"] == 4
    assert result["n_columns"] == 5
    assert result["empty_dataset"] is False


def test_missing_values():
    df = create_test_dataframe()

    result = check_missing_values(df)

    age_result = result[
        result["variable"] == "driver_age"
    ].iloc[0]

    assert age_result["missing_count"] == 2
    assert age_result["missing_pct"] == 50.0
    assert age_result["non_missing_count"] == 2


def test_duplicates():
    df = create_test_dataframe()

    result = check_duplicates(
        df,
        key_columns=["insured_id", "year"]
    )

    assert result["exact_duplicates"] == 1
    assert result["key_duplicates"] == 1


def test_data_types():
    df = pd.DataFrame({
        "age": ["25", "30", "inconnu"],
        "brand": ["Renault", "Peugeot", "Toyota"]
    })

    result = check_data_types(df)

    age_issue = result.loc[
        result["variable"] == "age", "issue"
    ].iloc[0]

    brand_issue = result.loc[
        result["variable"] == "brand", "issue"
    ].iloc[0]

    assert age_issue == "mixed_numeric_text"
    assert brand_issue == "none"


def test_outliers():
    df = pd.DataFrame({
        "premium": [100, 110, 120, 130, 140, 1000]
    })

    result = check_outliers(df)

    assert result.iloc[0]["outlier_count"] == 1


def test_actuarial_consistency():
    df = pd.DataFrame({
        "exposure": [1, -0.5, 0.8],
        "claims": [0, 1, -1],
        "premium": [500, -100, 300]
    })

    mapping = {
        "exposure": ["exposure"],
        "claim_count": ["claims"],
        "premium": ["premium"]
    }

    result = check_actuarial_consistency(df, mapping)

    assert (
        result.set_index("check")
        .loc["negative_exposure", "issue_count"]
    ) == 1

    assert (
        result.set_index("check")
        .loc["negative_claim_count", "issue_count"]
    ) == 1

    assert (
        result.set_index("check")
        .loc["negative_premium", "issue_count"]
    ) == 1


def test_statistical_issues():
    df = pd.DataFrame({
        "constant": [1, 1, 1, 1],
        "category": ["A", "A", "A", "B"]
    })

    result = check_statistical_issues(
        df,
        dominant_threshold=0.70
    )

    constant_issues = result.loc[
        result["variable"] == "constant", "issues"
    ].iloc[0]

    category_issues = result.loc[
        result["variable"] == "category", "issues"
    ].iloc[0]

    assert "constant_variable" in constant_issues
    assert "near_constant_variable" in category_issues


def test_generate_quality_report():
    df = create_test_dataframe()

    report = generate_quality_report(
        df,
        key_columns=["insured_id", "year"],
        column_mapping={
            "premium": ["total_premium"],
            "claim_count": ["total_claims"]
        }
    )

    expected_sections = {
        "structure",
        "missing_values",
        "duplicates",
        "data_types",
        "outliers",
        "statistical_issues",
        "actuarial_consistency"
    }

    assert set(report.keys()) == expected_sections
    assert report["structure"]["n_rows"] == 4
    assert report["duplicates"]["exact_duplicates"] == 1


def test_empty_dataframe():
    """Vérifie le comportement sur une base vide."""
    df = pd.DataFrame()

    structure = check_structure(df)
    duplicates = check_duplicates(df)
    missing = check_missing_values(df)

    assert structure["empty_dataset"] is True
    assert structure["n_rows"] == 0
    assert duplicates["exact_duplicates"] == 0
    assert missing.empty


def test_all_missing_column():
    """Vérifie une colonne entièrement manquante."""
    df = pd.DataFrame({
        "age": [None, None, None],
        "premium": [100, 200, 300]
    })

    missing = check_missing_values(df)
    types = check_data_types(df)

    assert missing.loc[
        missing["variable"] == "age",
        "missing_pct"
    ].iloc[0] == 100.0

    assert types.loc[
        types["variable"] == "age",
        "issue"
    ].iloc[0] == "all_missing"


def test_invalid_key_columns():
    """Vérifie le signalement d'une clé inexistante."""
    df = pd.DataFrame({"insured_id": [1, 2, 3]})

    with pytest.raises(ValueError):
        check_duplicates(
            df,
            key_columns=["unknown_column"]
        )


def test_outliers_constant_variable():
    """Vérifie le comportement quand l'IQR est nul."""
    df = pd.DataFrame({
        "premium": [100, 100, 100, 100]
    })

    result = check_outliers(df)

    assert result.empty

def test_actuarial_missing_column():
    df = pd.DataFrame({
        "premium": [100, 200, 300]
    })

    result = check_actuarial_consistency(
        df,
        {"premium": ["unknown_premium"]}
    )

    assert result.iloc[0]["status"] == "missing_column"
    assert pd.isna(result.iloc[0]["issue_count"])


def test_actuarial_non_numeric_column():
    df = pd.DataFrame({
        "premium": ["100", "200", "unknown"]
    })

    result = check_actuarial_consistency(
        df,
        {"premium": ["premium"]}
    )

    assert result.iloc[0]["status"] == "non_numeric_column"
    assert pd.isna(result.iloc[0]["issue_count"])

def test_report_missing_actuarial_column():
    df = pd.DataFrame({
        "premium": [100, 200, 300]
    })

    report = generate_quality_report(
        df,
        column_mapping={
            "premium": ["unknown_premium"]
        }
    )

    result = report["actuarial_consistency"][0]

    assert result["status"] == "missing_column"
    assert result["issue_count"] is None

def test_report_without_numeric_columns():
    df = pd.DataFrame({
        "vehicle_brand": ["Renault", "Peugeot", "Toyota"],
        "fuel_type": ["Diesel", "Essence", "Essence"]
    })

    report = generate_quality_report(df)

    assert report["structure"]["n_rows"] == 3
    assert report["outliers"] == []
    assert "actuarial_consistency" not in report


def test_report_empty_dataframe():
    df = pd.DataFrame()

    report = generate_quality_report(df)

    assert report["structure"]["empty_dataset"] is True
    assert report["missing_values"] == []
    assert report["outliers"] == []
    assert report["statistical_issues"] == []

def test_invalid_actuarial_mapping():
    df = pd.DataFrame({
        "total_premium": [100, 200, 300]
    })

    with pytest.raises(ValueError):
        check_actuarial_consistency(
            df,
            {"premium": "total_premium"}
        )

def test_report_json_serialization():
    df = create_test_dataframe()

    report = generate_quality_report(
        df,
        key_columns=["insured_id", "year"]
    )

    json_output = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False
    )

    assert isinstance(json_output, str)

import numpy as np


def test_infinite_values():
    df = pd.DataFrame({
        "premium": [100, 200, np.inf, -np.inf]
    })

    report = generate_quality_report(df)

    assert "structure" in report

    json.dumps(report, allow_nan=False)


def test_duplicate_column_names():
    df = pd.DataFrame(
        [[100, 200], [300, 400]],
        columns=["premium", "premium"]
    )

    result = check_structure(df)

    assert result["duplicate_column_names"] == ["premium"]


def test_all_missing_statistical():
    df = pd.DataFrame({
        "premium": [None, None, None],
        "claims": [0, 1, 0]
    })

    result = check_statistical_issues(df)

    assert "premium" not in result["variable"].tolist()
