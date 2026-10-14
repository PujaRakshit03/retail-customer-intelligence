"""Business metrics and trend aggregations."""
from __future__ import annotations
import pandas as pd


def overview_metrics(prepared: pd.DataFrame) -> dict[str, float | int | str]:
    dates = prepared["InvoiceDate"].dropna()
    standard = prepared.loc[prepared["is_standard_sale_line"]]
    return {
        "input_rows": int(len(prepared)),
        "standard_sale_lines": int(len(standard)),
        "recorded_standard_sales_value": float(standard["StandardSalesValue"].sum()),
        "signed_line_value": float(prepared.loc[prepared["is_valid_numeric_line"], "LineValue"].sum()),
        "identified_customers": int(prepared["CustomerID"].nunique(dropna=True)),
        "customer_id_coverage_pct": float(prepared["CustomerID"].notna().mean() * 100) if len(prepared) else 0.0,
        "countries": int(prepared["Country"].nunique(dropna=True)),
        "start_date": dates.min().strftime("%Y-%m-%d") if len(dates) else "Unknown",
        "end_date": dates.max().strftime("%Y-%m-%d") if len(dates) else "Unknown",
    }


def sales_over_time(prepared: pd.DataFrame, frequency: str = "MS") -> pd.DataFrame:
    data = prepared.loc[prepared["is_standard_sale_line"]].copy()
    if data.empty:
        return pd.DataFrame(columns=["Period", "RecordedSalesValue", "Lines", "Invoices"])
    data["Period"] = data["InvoiceDate"].dt.to_period("M" if frequency == "MS" else "D").dt.to_timestamp()
    return (data.groupby("Period", as_index=False)
            .agg(RecordedSalesValue=("StandardSalesValue", "sum"),
                 Lines=("InvoiceNo", "size"), Invoices=("InvoiceNo", "nunique"))
            .sort_values("Period"))


def group_performance(prepared: pd.DataFrame, group_col: str, limit: int = 15) -> pd.DataFrame:
    if group_col not in prepared.columns:
        raise ValueError(f"Unknown grouping column: {group_col}")
    data = prepared.loc[prepared["is_standard_sale_line"] & prepared[group_col].notna()]
    result = (data.groupby(group_col, as_index=False)
              .agg(RecordedSalesValue=("StandardSalesValue", "sum"),
                   Quantity=("Quantity", "sum"), Invoices=("InvoiceNo", "nunique"),
                   Lines=("InvoiceNo", "size"))
              .sort_values("RecordedSalesValue", ascending=False))
    return result.head(limit)


def customer_concentration(prepared: pd.DataFrame) -> pd.DataFrame:
    data = prepared.loc[prepared["is_standard_sale_line"] & prepared["CustomerID"].notna()]
    result = data.groupby("CustomerID", as_index=False).agg(
        HistoricalSpend=("StandardSalesValue", "sum"),
        Invoices=("InvoiceNo", "nunique"),
        LastObservedPurchase=("InvoiceDate", "max"),
    )
    return result.sort_values("HistoricalSpend", ascending=False)


def customer_cohort_retention(prepared: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cohort-size and monthly retention matrices from standard sale lines.

    Cohort is each identified customer's first observed purchase month. Months are
    numbered relative to that cohort. This is descriptive, not causal.
    """
    data = prepared.loc[prepared["is_standard_sale_line"] & prepared["CustomerID"].notna(),
                        ["CustomerID", "InvoiceDate"]].copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    data["CohortMonth"] = data.groupby("CustomerID")["InvoiceDate"].transform("min").dt.to_period("M")
    data["InvoiceMonth"] = data["InvoiceDate"].dt.to_period("M")
    data["CohortIndex"] = (data["InvoiceMonth"].astype("int64") - data["CohortMonth"].astype("int64")).astype(int)
    activity = data.drop_duplicates(["CustomerID", "CohortMonth", "CohortIndex"])
    counts = activity.groupby(["CohortMonth", "CohortIndex"])["CustomerID"].nunique().unstack(fill_value=0)
    cohort_sizes = counts.iloc[:, 0] if 0 in counts.columns else activity.groupby("CohortMonth")["CustomerID"].nunique()
    retention = counts.div(cohort_sizes, axis=0) * 100
    sizes = cohort_sizes.rename("Customers").reset_index()
    sizes["CohortMonth"] = sizes["CohortMonth"].astype(str)
    retention.index = retention.index.astype(str)
    retention.index.name = "CohortMonth"
    retention.columns = [f"Month {int(c)}" for c in retention.columns]
    return sizes, retention.round(1)


def temporal_repeat_purchase_validation(prepared: pd.DataFrame, horizon_days: int = 90) -> dict[str, object]:
    """Simple chronological holdout: prior purchasers, then any purchase in next horizon.

    The cutoff is horizon_days before the last observed timestamp. This is a
    descriptive baseline, not a trained predictive model.
    """
    data = prepared.loc[prepared["is_standard_sale_line"] & prepared["CustomerID"].notna(),
                        ["CustomerID", "InvoiceDate", "InvoiceNo"]].copy()
    if data.empty:
        return {"status": "insufficient_data", "reason": "No identified standard sale lines."}
    end = data["InvoiceDate"].max()
    cutoff = end - pd.Timedelta(days=horizon_days)
    train = data.loc[data["InvoiceDate"] <= cutoff]
    test = data.loc[(data["InvoiceDate"] > cutoff) & (data["InvoiceDate"] <= end)]
    if train.empty or test.empty:
        return {"status": "insufficient_data", "reason": "Not enough history on both sides of the chronological cutoff."}
    train_customers = set(train["CustomerID"].unique())
    test_customers = set(test["CustomerID"].unique())
    eligible = train_customers
    returned = eligible & test_customers
    if not eligible:
        return {"status": "insufficient_data", "reason": "No customers were observed before the cutoff."}
    freq = train.groupby("CustomerID")["InvoiceNo"].nunique()
    frequent = set(freq[freq >= 2].index) & eligible
    less_frequent = eligible - frequent
    frequent_returned = frequent & returned
    less_returned = less_frequent & returned
    rate = lambda num, den: (100.0 * len(num) / len(den)) if den else None
    return {
        "status": "ok", "cutoff_date": cutoff.strftime("%Y-%m-%d"),
        "holdout_end_date": end.strftime("%Y-%m-%d"), "horizon_days": int(horizon_days),
        "eligible_prior_customers": int(len(eligible)), "returned_customers": int(len(returned)),
        "overall_repeat_rate_pct": rate(returned, eligible),
        "prior_frequency_2plus_customers": int(len(frequent)),
        "prior_frequency_2plus_return_rate_pct": rate(frequent_returned, frequent),
        "prior_frequency_1_customers": int(len(less_frequent)),
        "prior_frequency_1_return_rate_pct": rate(less_returned, less_frequent),
        "interpretation": "Descriptive chronological baseline only; it does not establish causality or validate a predictive model.",
    }


def purchase_interval_summary(prepared: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize days between distinct invoices for identified customers."""
    data = prepared.loc[
        prepared["is_standard_sale_line"] & prepared["CustomerID"].notna(),
        ["CustomerID", "InvoiceNo", "InvoiceDate", "StandardSalesValue"],
    ].copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    invoices = (data.groupby(["CustomerID", "InvoiceNo"], as_index=False)
                .agg(InvoiceDate=("InvoiceDate", "min"), OrderValue=("StandardSalesValue", "sum"))
                .sort_values(["CustomerID", "InvoiceDate", "InvoiceNo"]))
    invoices["DaysSincePreviousOrder"] = invoices.groupby("CustomerID")["InvoiceDate"].diff().dt.total_seconds() / 86400
    intervals = invoices.dropna(subset=["DaysSincePreviousOrder"])
    if intervals.empty:
        return invoices, pd.DataFrame()
    customer_summary = intervals.groupby("CustomerID", as_index=False).agg(
        RepeatIntervals=("DaysSincePreviousOrder", "count"),
        MedianDaysBetweenOrders=("DaysSincePreviousOrder", "median"),
        MeanDaysBetweenOrders=("DaysSincePreviousOrder", "mean"),
    )
    return invoices, customer_summary.sort_values("MedianDaysBetweenOrders")


def top_product_pairs(prepared: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Find product pairs occurring in the same standard-sale invoice.

    Counts invoice co-occurrence, not causal complementarity. Duplicate lines for
    the same stock code within an invoice are deduplicated before pairing.
    """
    from itertools import combinations
    data = prepared.loc[
        prepared["is_standard_sale_line"], ["InvoiceNo", "StockCode", "Description"]
    ].dropna(subset=["InvoiceNo", "StockCode"]).copy()
    if data.empty:
        return pd.DataFrame(columns=["Product A", "Product B", "InvoicesTogether"])
    descriptions = (data.dropna(subset=["Description"])
                    .drop_duplicates("StockCode").set_index("StockCode")["Description"].to_dict())
    invoice_products = data.groupby("InvoiceNo")["StockCode"].agg(lambda x: sorted(set(x.astype(str))))
    counts: dict[tuple[str, str], int] = {}
    for products in invoice_products:
        if len(products) < 2:
            continue
        for pair in combinations(products, 2):
            counts[pair] = counts.get(pair, 0) + 1
    if not counts:
        return pd.DataFrame(columns=["Product A", "Product B", "InvoicesTogether"])
    rows = [{"Product A": descriptions.get(a, a), "Product B": descriptions.get(b, b),
             "InvoicesTogether": count} for (a, b), count in counts.items()]
    return pd.DataFrame(rows).sort_values("InvoicesTogether", ascending=False).head(limit).reset_index(drop=True)


def segment_future_outcomes(prepared: pd.DataFrame, horizon_days: int = 60) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate descriptive segments against a later chronological holdout.

    Segments are assigned using only pre-cutoff transactions. Outcomes are any
    standard-sale purchase and recorded sales value during the final horizon.
    This is retrospective validation, not a causal or production prediction.
    """
    from retail_intelligence.customers import build_rfm, assign_customer_segments
    data = prepared.loc[
        prepared["is_standard_sale_line"] & prepared["CustomerID"].notna(),
        ["CustomerID", "InvoiceNo", "InvoiceDate", "StandardSalesValue"],
    ].copy()
    if data.empty:
        return pd.DataFrame(), {"status": "insufficient_data", "reason": "No identified standard sale lines."}
    end = data["InvoiceDate"].max()
    cutoff = end - pd.Timedelta(days=horizon_days)
    history = data.loc[data["InvoiceDate"] <= cutoff]
    future = data.loc[(data["InvoiceDate"] > cutoff) & (data["InvoiceDate"] <= end)]
    if history.empty or future.empty:
        return pd.DataFrame(), {"status": "insufficient_data", "reason": "Not enough data before and after the selected cutoff."}
    historical_prepared = prepared.loc[prepared["InvoiceDate"] <= cutoff].copy()
    rfm = build_rfm(historical_prepared, analysis_date=cutoff + pd.Timedelta(days=1))
    if rfm.empty:
        return pd.DataFrame(), {"status": "insufficient_data", "reason": "No eligible historical customers before cutoff."}
    segmented = assign_customer_segments(rfm)
    future_stats = future.groupby("CustomerID", as_index=False).agg(
        FutureSalesValue=("StandardSalesValue", "sum"),
        FutureInvoices=("InvoiceNo", "nunique"),
    )
    future_stats["PurchasedInHoldout"] = True
    evaluated = segmented.merge(future_stats, on="CustomerID", how="left")
    evaluated["FutureSalesValue"] = evaluated["FutureSalesValue"].fillna(0.0)
    evaluated["FutureInvoices"] = evaluated["FutureInvoices"].fillna(0).astype(int)
    evaluated["PurchasedInHoldout"] = evaluated["PurchasedInHoldout"].eq(True)
    summary = evaluated.groupby("Segment", as_index=False).agg(
        Customers=("CustomerID", "nunique"),
        HoldoutPurchaseRatePct=("PurchasedInHoldout", lambda x: x.mean() * 100),
        MedianHistoricalSpend=("MonetaryValue", "median"),
        TotalFutureSalesValue=("FutureSalesValue", "sum"),
        CustomersWithFuturePurchase=("PurchasedInHoldout", "sum"),
    ).sort_values("Customers", ascending=False)
    return summary, {"status": "ok", "cutoff_date": cutoff.strftime("%Y-%m-%d"),
                     "holdout_end_date": end.strftime("%Y-%m-%d"), "horizon_days": int(horizon_days),
                     "eligible_customers": int(len(evaluated)),
                     "overall_purchase_rate_pct": float(evaluated["PurchasedInHoldout"].mean() * 100),
                     "customer_outcomes": evaluated}


def cancellation_summary(prepared: pd.DataFrame, limit: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize cancellation-invoice patterns by product and month."""
    flagged = prepared.loc[prepared["is_cancellation_invoice"]].copy()
    if flagged.empty:
        return pd.DataFrame(), pd.DataFrame()
    flagged["Month"] = flagged["InvoiceDate"].dt.to_period("M").astype(str)
    by_month = flagged.groupby("Month", as_index=False).agg(
        FlaggedLines=("InvoiceNo", "size"),
        Invoices=("InvoiceNo", "nunique"),
        SignedLineValue=("LineValue", "sum"),
    ).sort_values("Month")
    by_product = flagged.groupby("Description", as_index=False).agg(
        FlaggedLines=("InvoiceNo", "size"),
        Invoices=("InvoiceNo", "nunique"),
        SignedLineValue=("LineValue", "sum"),
    ).sort_values("FlaggedLines", ascending=False).head(limit)
    return by_month, by_product
