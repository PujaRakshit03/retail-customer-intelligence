import pandas as pd
import pytest
from retail_intelligence.data import REQUIRED_COLUMNS, audit_transactions, prepare_transactions, load_transactions

@pytest.fixture
def sample():
    return pd.DataFrame([
        ["10001", "A", "Widget", 2, "2011-01-01 10:00", 5.0, 123.0, "UK"],
        ["10002", "B", "Gadget", 1, "2011-01-02 10:00", 10.0, None, "UK"],
        ["C10003", "A", "Widget", -1, "2011-01-03 10:00", 5.0, 123.0, "UK"],
        ["10004", "C", "Zero", 1, "2011-01-04 10:00", 0.0, 456.0, "France"],
        ["10005", "D", "Bad date", 1, "not-a-date", 3.0, 789.0, "France"],
    ], columns=REQUIRED_COLUMNS)

def test_prepare_preserves_rows_and_flags(sample):
    prepared = prepare_transactions(sample)
    assert len(prepared) == len(sample)
    assert prepared["is_standard_sale_line"].tolist() == [True, True, False, False, False]
    assert prepared.loc[0, "StandardSalesValue"] == 10.0
    assert prepared.loc[2, "LineValue"] == -5.0

def test_audit_reports_missing_and_cancellations(sample):
    audit = audit_transactions(sample)
    assert audit["rows"] == 5
    assert audit["missing_customer_id_rows"] == 1
    assert audit["negative_quantity_rows"] == 1
    assert audit["cancellation_invoice_rows"] == 1
    assert audit["invalid_date_rows"] == 1

def test_missing_required_column_fails(sample):
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_transactions(sample.drop(columns=["Country"]))

def test_loader_missing_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_transactions(tmp_path / "missing.xlsx")

def test_duplicate_rows_are_reported_not_removed(sample):
    repeated = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
    prepared = prepare_transactions(repeated)
    assert len(prepared) == 6
    assert audit_transactions(repeated)["duplicate_rows"] == 1
