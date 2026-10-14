import pandas as pd
from retail_intelligence.data import REQUIRED_COLUMNS, prepare_transactions
from retail_intelligence.analytics import overview_metrics, sales_over_time, group_performance, customer_concentration
from retail_intelligence.customers import build_rfm, add_opportunity_flags, cluster_customers

def prepared_fixture():
    rows = []
    for i, (customer, invoice, date, qty, price, product) in enumerate([
        (1, "100", "2011-01-01", 2, 5, "A"),
        (1, "101", "2011-01-10", 1, 10, "B"),
        (2, "102", "2011-01-12", 1, 20, "A"),
        (2, "103", "2011-02-12", 3, 20, "C"),
        (3, "104", "2011-02-14", 1, 100, "D"),
    ]):
        rows.append([invoice, product, product, qty, date, price, float(customer), "UK"])
    return prepare_transactions(pd.DataFrame(rows, columns=REQUIRED_COLUMNS))

def test_overview_and_sales_trend():
    p = prepared_fixture()
    metrics = overview_metrics(p)
    assert metrics["input_rows"] == 5
    assert metrics["recorded_standard_sales_value"] == 200.0
    assert metrics["identified_customers"] == 3
    trend = sales_over_time(p)
    assert len(trend) == 2
    assert trend["RecordedSalesValue"].sum() == 200.0

def test_group_performance_and_customer_concentration():
    p = prepared_fixture()
    assert group_performance(p, "Country").iloc[0]["RecordedSalesValue"] == 200.0
    assert len(customer_concentration(p)) == 3

def test_rfm_and_flags_are_explainable():
    rfm = build_rfm(prepared_fixture())
    assert set(["RecencyDays", "Frequency", "MonetaryValue", "DistinctProducts"]).issubset(rfm.columns)
    flagged = add_opportunity_flags(rfm)
    assert "Reason" in flagged.columns and "SuggestedAction" in flagged.columns
    assert len(flagged) == 3

def test_empty_rfm_flags():
    empty = pd.DataFrame(columns=["CustomerID", "RecencyDays", "Frequency", "MonetaryValue", "DistinctProducts", "LastPurchase"])
    out = add_opportunity_flags(empty)
    assert out.empty and "Opportunity" in out.columns

def test_clustering_handles_small_data():
    small = build_rfm(prepared_fixture())
    out, diagnostics = cluster_customers(small, n_clusters=4)
    assert len(out) == len(small)
    assert diagnostics["status"] == "insufficient_data"

def test_customer_cohort_retention_returns_sizes_and_matrix():
    from retail_intelligence.analytics import customer_cohort_retention
    sizes, retention = customer_cohort_retention(prepared_fixture())
    assert not sizes.empty
    assert not retention.empty
    assert sizes['Customers'].sum() >= 1
    assert retention.iloc[0, 0] == 100.0


def test_temporal_validation_handles_fixture():
    from retail_intelligence.analytics import temporal_repeat_purchase_validation
    result = temporal_repeat_purchase_validation(prepared_fixture(), horizon_days=30)
    # This fixture has insufficient history for a 30-day holdout after its final purchase.
    assert result['status'] in {'ok', 'insufficient_data'}
    if result['status'] == 'ok':
        assert 0 <= result['overall_repeat_rate_pct'] <= 100


def test_purchase_intervals_and_product_pairs():
    from retail_intelligence.analytics import purchase_interval_summary, top_product_pairs
    p = prepared_fixture()
    invoices, intervals = purchase_interval_summary(p)
    assert not invoices.empty
    assert not intervals.empty
    assert intervals["MedianDaysBetweenOrders"].min() >= 0
    pair_rows = pd.concat([p, p.iloc[[1]].assign(InvoiceNo="100", InvoiceDate=pd.Timestamp("2011-01-01"))], ignore_index=True)
    pairs = top_product_pairs(pair_rows)
    assert not pairs.empty
    assert pairs.iloc[0]["InvoicesTogether"] >= 1


def test_segment_future_outcomes_uses_chronological_holdout():
    from retail_intelligence.analytics import segment_future_outcomes
    summary, result = segment_future_outcomes(prepared_fixture(), horizon_days=30)
    assert result["status"] in {"ok", "insufficient_data"}
    if result["status"] == "ok":
        assert not summary.empty
        assert 0 <= result["overall_purchase_rate_pct"] <= 100
        assert "PurchasedInHoldout" in result["customer_outcomes"].columns


def test_cancellation_summary_empty_when_no_cancellations():
    from retail_intelligence.analytics import cancellation_summary
    months, products = cancellation_summary(prepared_fixture())
    assert months.empty and products.empty
