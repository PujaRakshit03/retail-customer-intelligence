"""Data loading, validation, audit, and transparent transaction preparation."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load CSV/XLS/XLSX transactions and validate the expected schema."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{source}'. Place the UCI Online Retail file there "
            "or choose another file path in the dashboard."
        )
    suffix = source.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(source)
    elif suffix == ".csv":
        df = pd.read_csv(source)
    else:
        raise ValueError("Unsupported dataset format. Use .xlsx, .xls, or .csv.")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset schema does not match UCI Online Retail. Missing columns: "
            + ", ".join(missing)
        )
    return df[REQUIRED_COLUMNS].copy()


def audit_transactions(df: pd.DataFrame) -> dict[str, Any]:
    """Return a reproducible data-quality audit without mutating the input."""
    missing = {str(k): int(v) for k, v in df.isna().sum().items()}
    invoice = df["InvoiceNo"].astype("string").str.strip()
    quantity = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["UnitPrice"], errors="coerce")
    dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": missing,
        "missing_customer_id_rows": int(df["CustomerID"].isna().sum()),
        "negative_quantity_rows": int((quantity < 0).sum()),
        "zero_quantity_rows": int((quantity == 0).sum()),
        "negative_unit_price_rows": int((price < 0).sum()),
        "zero_unit_price_rows": int((price == 0).sum()),
        "invalid_date_rows": int(dates.isna().sum()),
        "cancellation_invoice_rows": int(invoice.str.upper().str.startswith("C", na=False).sum()),
        "date_min": dates.min(),
        "date_max": dates.max(),
        "unique_customers": int(df["CustomerID"].nunique(dropna=True)),
        "unique_countries": int(df["Country"].nunique(dropna=True)),
        "unique_stock_codes": int(df["StockCode"].nunique(dropna=True)),
    }


def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and add transparent analytical flags; do not drop rows.

    `is_cancellation_invoice` is a pattern flag based on invoice number, not a
    guarantee that every flagged line is a refund. `is_standard_sale_line`
    includes positive quantity and positive unit price on a non-cancellation
    invoice. Other rows remain in the prepared dataset for audit/alternative views.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    out = df[REQUIRED_COLUMNS].copy()
    out["InvoiceNo"] = out["InvoiceNo"].astype("string").str.strip()
    out["StockCode"] = out["StockCode"].astype("string").str.strip()
    out["Description"] = out["Description"].astype("string").str.strip()
    out["Country"] = out["Country"].astype("string").str.strip()
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    out["UnitPrice"] = pd.to_numeric(out["UnitPrice"], errors="coerce")
    out["CustomerID"] = pd.to_numeric(out["CustomerID"], errors="coerce")
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    out["LineValue"] = out["Quantity"] * out["UnitPrice"]
    out["is_cancellation_invoice"] = out["InvoiceNo"].str.upper().str.startswith("C", na=False)
    out["is_valid_numeric_line"] = (
        out["Quantity"].notna() & out["UnitPrice"].notna()
        & out["InvoiceDate"].notna() & out["LineValue"].notna()
    )
    out["is_standard_sale_line"] = (
        out["is_valid_numeric_line"] & ~out["is_cancellation_invoice"]
        & (out["Quantity"] > 0) & (out["UnitPrice"] > 0)
    )
    out["StandardSalesValue"] = out["LineValue"].where(out["is_standard_sale_line"], 0.0)
    out["CustomerIDDisplay"] = out["CustomerID"].map(
        lambda x: str(int(x)) if pd.notna(x) else pd.NA
    ).astype("string")
    return out


def data_quality_summary(prepared: pd.DataFrame) -> pd.DataFrame:
    """Summarize major transaction treatments for the UI."""
    rows = [
        ("All input rows", len(prepared), "No rows removed by preparation."),
        ("Standard sale lines", int(prepared["is_standard_sale_line"].sum()),
         "Positive quantity and price, valid date/numerics, non-cancellation invoice."),
        ("Cancellation-invoice lines", int(prepared["is_cancellation_invoice"].sum()),
         "Invoice number starts with C; inspect before interpreting as a refund."),
        ("Negative-quantity lines", int((prepared["Quantity"] < 0).sum()),
         "May represent returns or corrections; retained for audit."),
        ("Missing customer IDs", int(prepared["CustomerID"].isna().sum()),
         "May contribute to transaction analysis but excluded from customer-level analysis."),
        ("Invalid date/numeric lines", int((~prepared["is_valid_numeric_line"]).sum()),
         "Excluded from standard sale value but retained in the prepared data."),
        ("Duplicate rows (exact match)", int(prepared[REQUIRED_COLUMNS].duplicated().sum()),
         "Reported only; not automatically removed."),
    ]
    return pd.DataFrame(rows, columns=["Check", "Rows", "Treatment / interpretation"])
