import pandas as pd
import pytest

from tools import (
    load_and_validate_data,
    portfolio_summary,
    claim_frequency,
    frequency_by_segment,
    risk_segments,
    detect_anomalies,
    filter_portfolio,
    filtered_frequency
)

def create_test_portfolio():
    return pd.DataFrame({
        "policy_id": [1, 2, 3, 4],
        "exposure": [1.0, 0.5, 1.0, 0.5],
        "claim_count": [0, 1, 0, 1],
        "region": ["A", "A", "B", "B"]
    })


def test_claim_frequency():
    df = create_test_portfolio()

    result = claim_frequency(df)

    assert result["total_claims"] == 2
    assert result["total_exposure"] == 3.0
    assert result["claim_frequency"] == 2 / 3

def test_frequency_by_segment():
    df = create_test_portfolio()

    result = frequency_by_segment(
        df,
        variable="region"
    )

    region_a = result[result["region"] == "A"].iloc[0]
    region_b = result[result["region"] == "B"].iloc[0]

    assert region_a["exposure"] == 1.5
    assert region_a["claims"] == 1
    assert region_a["frequency"] == 1 / 1.5

    assert region_b["exposure"] == 1.5
    assert region_b["claims"] == 1
    assert region_b["frequency"] == 1 / 1.5


def test_invalid_exposure(tmp_path):
    df = pd.DataFrame({
        "policy_id": [1, 2],
        "exposure": [1.0, -0.5],
        "claim_count": [0, 1]
    })

    filepath = tmp_path / "invalid_portfolio.csv"
    df.to_csv(filepath, index=False)

    with pytest.raises(ValueError):
        load_and_validate_data(filepath)

def test_risk_segments():
    df = pd.DataFrame({
        "policy_id": [1, 2, 3, 4, 5, 6],
        "exposure": [1, 1, 1, 1, 1, 1],
        "claim_count": [0, 0, 1, 1, 2, 1],
        "region": ["A", "A", "B", "B", "C", "C"]
    })

    result = risk_segments(
        df,
        variable="region",
        top_n=2,
        min_exposure=1
    )

    assert len(result) == 2
    assert result.iloc[0]["region"] == "C"
    assert result.iloc[1]["region"] == "B"


def test_detect_anomalies():
    df = pd.DataFrame({
        "policy_id": [1, 1, 3],
        "exposure": [1.0, 0.0, 1.0],
        "claim_count": [0, -1, 1],
        "region": ["A", "B", None]
    })

    result = detect_anomalies(df)

    assert result["missing_values"]["region"] == 1
    assert result["invalid_exposure"] == 1
    assert result["negative_claim_count"] == 1
    assert result["duplicate_policy_id"] == 1

def test_filter_portfolio():
    df = pd.DataFrame({
        "policy_id": [1, 2, 3, 4],
        "exposure": [1.0, 1.0, 1.0, 1.0],
        "claim_count": [0, 1, 0, 1],
        "driver_age": [20, 25, 40, 60]
    })

    result = filter_portfolio(
        df,
        variable="driver_age",
        operator="<",
        value=30
    )

    assert len(result) == 2
    assert result["policy_id"].tolist() == [1, 2]


def test_filtered_frequency():
    df = pd.DataFrame({
        "policy_id": [1, 2, 3, 4],
        "exposure": [1.0, 1.0, 1.0, 1.0],
        "claim_count": [0, 1, 0, 1],
        "driver_age": [20, 25, 40, 60]
    })

    result = filtered_frequency(
        df,
        variable="driver_age",
        operator="<",
        value=30
    )

    assert result["number_of_policies"] == 2
    assert result["total_claims"] == 1
    assert result["total_exposure"] == 2.0
    assert result["claim_frequency"] == 0.5