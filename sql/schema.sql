-- SQLite-compatible schema for a prepared transaction export.
-- Import prepared records into this table before running the queries.
CREATE TABLE IF NOT EXISTS transactions (
    InvoiceNo TEXT,
    StockCode TEXT,
    Description TEXT,
    Quantity REAL,
    InvoiceDate TEXT,
    UnitPrice REAL,
    CustomerID REAL,
    Country TEXT,
    LineValue REAL,
    is_cancellation_invoice INTEGER,
    is_standard_sale_line INTEGER,
    StandardSalesValue REAL
);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(InvoiceDate);
CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(CustomerID);
