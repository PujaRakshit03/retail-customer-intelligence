import pandas as pd

from retail_intelligence.customers import assign_customer_segments, summarize_customer_segments


def sample_rfm():
    return pd.DataFrame({
        "CustomerID": [1, 2, 3, 4, 5, 6],
        "RecencyDays": [2, 5, 10, 30, 45, 60],
        "Frequency": [8, 5, 4, 2, 1, 1],
        "MonetaryValue": [1000.0, 800.0, 500.0, 200.0, 100.0, 50.0],
        "DistinctProducts": [8, 6, 5, 3, 2, 1],
        "LastPurchase": pd.to_datetime(["2020-01-30", "2020-01-27", "2020-01-22", "2020-01-02", "2019-12-18", "2019-12-03"]),
    })


def test_segments_are_assigned_and_explainable():
    result = assign_customer_segments(sample_rfm())
    assert "Segment" in result
    assert result["Segment"].notna().all()
    assert result["SegmentReason"].str.len().gt(0).all()
    assert result["RecommendedAction"].str.len().gt(0).all()
    assert result["CustomerID"].nunique() == 6


def test_segment_summary_counts_all_customers():
    result = assign_customer_segments(sample_rfm())
    summary = summarize_customer_segments(result)
    assert summary["Customers"].sum() == 6
    assert round(summary["SharePct"].sum(), 5) == 100.0


def test_empty_segmentation_is_supported():
    empty = pd.DataFrame(columns=["CustomerID", "RecencyDays", "Frequency", "MonetaryValue", "DistinctProducts", "LastPurchase"])
    result = assign_customer_segments(empty)
    assert result.empty
    assert "Segment" in result
