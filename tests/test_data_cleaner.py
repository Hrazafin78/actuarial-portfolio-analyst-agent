"""Tests des propositions et actions de nettoyage contrôlé.

Vérifie les choix explicites, la conservation de la source, les journaux
et la distinction entre alertes informatives et corrections applicables.
"""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data_cleaner import (
    apply_cleaning_actions,
    generate_cleaning_report,
    suggest_cleaning_actions,
)


def fake_report(df, *, types=None, key_duplicates=0, outliers=None):
    """Construit les sections minimales du diagnostic nécessaires au scénario de nettoyage."""
    return {
        "structure": {"empty_dataset": df.empty},
        "missing_values": [
            {"variable": c, "missing_count": int(df[c].isna().sum()),
             "missing_pct": float(df[c].isna().mean() * 100)}
            for c in df if df.columns.is_unique
        ],
        "data_types": types or [],
        "duplicates": {"exact_duplicates": int(df.duplicated().sum()),
                       "key_duplicates": key_duplicates},
        "outliers": outliers or [],
        "statistical_issues": [{"variable": "region", "issue": "high_cardinality"}],
        "actuarial_consistency": [{"variable": "premium", "issue": "negative_value"}],
    }


def choice_id(suggestions, prefix, column=None):
    """Retrouve l’identifiant de proposition correspondant au type et à la colonne testés."""
    for p in suggestions["proposals"]:
        if p["id"].startswith(prefix) and (column is None or p["column"] == column):
            return p["id"]
    raise AssertionError(f"Proposition introuvable : {prefix}, {column}")


def test_no_selection_makes_no_change():
    df = pd.DataFrame({"region": ["Nord", None], "age": [30, 20]})
    original = df.copy(deep=True)
    suggestions = suggest_cleaning_actions(df, fake_report(df))
    clean, log = apply_cleaning_actions(df, {}, suggestions)
    assert_frame_equal(df, original)
    assert_frame_equal(clean, original)
    assert log == []


def test_fill_unknown_preserves_original_and_logs():
    df = pd.DataFrame({"region": [None, "Paris", None], "premium": [10, 20, 30]})
    suggestions = suggest_cleaning_actions(df, fake_report(df))
    pid = choice_id(suggestions, "missing::", "region")
    clean, log = apply_cleaning_actions(df, {pid: "fill_unknown"}, suggestions)
    assert clean["region"].tolist() == ["Inconnu", "Paris", "Inconnu"]
    assert pd.isna(df.loc[0, "region"])
    assert log[0]["affected"] == 2
    assert generate_cleaning_report(df, clean, log)["operations_applied"] == 1


def test_numeric_median_works():
    df = pd.DataFrame({"premium": [10.0, None, 30.0]})
    suggestions = suggest_cleaning_actions(df, fake_report(df))
    pid = choice_id(suggestions, "missing::", "premium")
    clean, _ = apply_cleaning_actions(df, {pid: "fill_median"}, suggestions)
    assert clean["premium"].tolist() == [10.0, 20.0, 30.0]


def test_exact_duplicates_drop():
    df = pd.DataFrame({"id": [1, 1, 2], "premium": [10, 10, 12]})
    suggestions = suggest_cleaning_actions(df, fake_report(df))
    clean, log = apply_cleaning_actions(
        df, {"duplicates::exact": "drop_exact_duplicates"}, suggestions
    )
    assert len(df) == 3
    assert len(clean) == 2
    assert log[0]["affected"] == 1


def test_key_duplicates_only_to_review():
    df = pd.DataFrame({"id": [1, 1], "premium": [10, 20]})
    suggestions = suggest_cleaning_actions(df, fake_report(df, key_duplicates=1))
    assert any(x["section"] == "duplicates" for x in suggestions["to_review"])
    assert not any("key" in p["id"] for p in suggestions["proposals"])


def test_numeric_text_comma_conversion():
    df = pd.DataFrame({" cost ": ["1,5", "2", None], "region": ["A", "B", "C"]})
    report = fake_report(df, types=[{"variable": " cost ", "issue": "numeric_stored_as_text"}])
    suggestions = suggest_cleaning_actions(df, report)
    trim = choice_id(suggestions, "columns::")
    numeric = choice_id(suggestions, "type::", " cost ")
    clean, log = apply_cleaning_actions(
        df, {numeric: "convert_numeric", trim: "trim_column_names"}, suggestions
    )
    assert "cost" in clean.columns and " cost " not in clean.columns
    assert clean["cost"].iloc[0] == 1.5
    assert pd.isna(clean["cost"].iloc[2])
    assert log[0]["action"] == "trim_column_names"
    assert_frame_equal(df, pd.DataFrame({" cost ": ["1,5", "2", None], "region": ["A", "B", "C"]}))


def test_mixed_numeric_text_only_to_review():
    df = pd.DataFrame({"code": ["12", "ABC", None]})
    suggestions = suggest_cleaning_actions(df, fake_report(
        df, types=[{"variable": "code", "issue": "mixed_numeric_text"}]))
    assert not any(p["id"].startswith("type::") for p in suggestions["proposals"])
    assert any(x["section"] == "data_types" for x in suggestions["to_review"])


def test_outlier_and_actuarial_issues_never_have_corrections():
    df = pd.DataFrame({"premium": [10, 1000], "region": ["A", "B"]})
    suggestions = suggest_cleaning_actions(df, fake_report(
        df, outliers=[{"variable": "premium", "n_outliers": 1}]))
    assert {"outliers", "actuarial_consistency", "statistical_issues"}.issubset(
        {x["section"] for x in suggestions["to_review"]}
    )
    assert all("options" not in item for item in suggestions["to_review"])


def test_all_missing_column_drop_only_when_user_selects():
    df = pd.DataFrame({"x": [None, None], "y": [1, 2]})
    suggestions = suggest_cleaning_actions(df, fake_report(df))
    pid = choice_id(suggestions, "missing::", "x")
    options = {x["id"] for x in next(p for p in suggestions["proposals"] if p["id"] == pid)["options"]}
    assert "drop_missing_rows" not in options
    clean, _ = apply_cleaning_actions(df, {pid: "drop_column"}, suggestions)
    assert list(clean.columns) == ["y"]


def test_column_name_trim_collision_only_review():
    df = pd.DataFrame({" x ": [1], "x": [2]})
    suggestions = suggest_cleaning_actions(df, fake_report(df))
    assert not any(p["id"] == "columns::trim" for p in suggestions["proposals"])
    assert any(x["section"] == "structure" for x in suggestions["to_review"])


def test_invalid_option_cannot_be_applied():
    df = pd.DataFrame({"x": [1, None]})
    suggestions = suggest_cleaning_actions(df, fake_report(df))
    pid = choice_id(suggestions, "missing::", "x")
    with pytest.raises(ValueError, match="Option invalide"):
        apply_cleaning_actions(df, {pid: "fill_unknown"}, suggestions)
    with pytest.raises(ValueError, match="Proposition inconnue"):
        apply_cleaning_actions(df, {"made_up": "drop_column"}, suggestions)


def test_drop_rows_never_empties_portfolio():
    df = pd.DataFrame({"x": [None, 1.0], "y": [1.0, None]})
    suggestions = suggest_cleaning_actions(df, fake_report(df))
    pids = {col: choice_id(suggestions, "missing::", col) for col in ("x", "y")}
    with pytest.raises(ValueError, match="viderait toute la base"):
        apply_cleaning_actions(df, {pid: "drop_missing_rows" for pid in pids.values()}, suggestions)
    assert len(df) == 2


def test_drop_missing_rows_accurate_report():
    df = pd.DataFrame({"x": [None, 1, 2], "y": [1, 1, 2]})
    suggestions = suggest_cleaning_actions(df, fake_report(df))
    pid = choice_id(suggestions, "missing::", "x")
    clean, log = apply_cleaning_actions(df, {pid: "drop_missing_rows"}, suggestions)
    report = generate_cleaning_report(df, clean, log)
    assert report["removed_rows"] == 1
    assert len(clean) == 2


def test_review_ignores_zero_outliers_even_if_bounds_exist():
    df = pd.DataFrame({"insured_id": [1, 2], "year": [2022, 2024], "age": [20, 100]})
    report = fake_report(df, outliers=[
        {"variable": "insured_id", "outlier_count": 0, "outlier_pct": 0.0},
        {"variable": "year", "outlier_count": 0, "outlier_pct": 0.0},
        {"variable": "age", "outlier_count": 1, "outlier_pct": 0.0},
    ])
    review = suggest_cleaning_actions(df, report)["to_review"]
    labels = [x["issue"] for x in review if x["section"] == "outliers"]
    assert labels == ["age : 1 valeur(s) atypique(s)"]


def test_review_ignores_statistical_rows_without_issues():
    df = pd.DataFrame({"policy_type": ["A", "B"], "id": [1, 2]})
    report = fake_report(df)
    report["statistical_issues"] = [
        {"variable": "policy_type", "issues": [], "rare_count": 0},
        {"variable": "id", "issues": ["high_cardinality"], "n_unique": 2},
    ]
    review = suggest_cleaning_actions(df, report)["to_review"]
    labels = [x["issue"] for x in review if x["section"] == "statistical_issues"]
    assert labels == ["id"]
    assert all("options" not in x for x in review)


def test_review_keeps_legacy_singular_statistical_issue():
    df = pd.DataFrame({"code": [1, 2]})
    report = fake_report(df)
    report["statistical_issues"] = [{"variable": "code", "issue": "high_cardinality"}]
    review = suggest_cleaning_actions(df, report)["to_review"]
    assert any(x["section"] == "statistical_issues" and x["issue"] == "code" for x in review)


def test_review_ignores_actuarial_entry_explicitly_without_issues():
    df = pd.DataFrame({"premium": [10, 20]})
    report = fake_report(df)
    report["actuarial_consistency"] = [{"variable": "premium", "issues": []}]
    review = suggest_cleaning_actions(df, report)["to_review"]
    assert not any(x["section"] == "actuarial_consistency" for x in review)
