-- 1. Monthly recorded standard sales
SELECT substr(InvoiceDate, 1, 7) AS month,
       SUM(StandardSalesValue) AS recorded_sales_value,
       COUNT(*) AS sale_lines,
       COUNT(DISTINCT InvoiceNo) AS invoices
FROM transactions
WHERE is_standard_sale_line = 1
GROUP BY substr(InvoiceDate, 1, 7)
ORDER BY month;

-- 2. Top products by recorded standard sales
SELECT Description,
       SUM(StandardSalesValue) AS recorded_sales_value,
       SUM(Quantity) AS quantity,
       COUNT(DISTINCT InvoiceNo) AS invoices
FROM transactions
WHERE is_standard_sale_line = 1 AND Description IS NOT NULL
GROUP BY Description
ORDER BY recorded_sales_value DESC
LIMIT 15;

-- 3. Country performance
SELECT Country,
       SUM(StandardSalesValue) AS recorded_sales_value,
       COUNT(DISTINCT InvoiceNo) AS invoices,
       COUNT(*) AS sale_lines
FROM transactions
WHERE is_standard_sale_line = 1 AND Country IS NOT NULL
GROUP BY Country
ORDER BY recorded_sales_value DESC;

-- 4. Historical spend and purchase frequency by identified customer
SELECT CustomerID,
       SUM(StandardSalesValue) AS historical_spend,
       COUNT(DISTINCT InvoiceNo) AS invoice_frequency,
       MAX(InvoiceDate) AS last_observed_purchase
FROM transactions
WHERE is_standard_sale_line = 1 AND CustomerID IS NOT NULL
GROUP BY CustomerID
ORDER BY historical_spend DESC;

-- 5. Data quality overview; cancellation prefix is an indicator, not a verdict
SELECT COUNT(*) AS input_rows,
       SUM(CASE WHEN CustomerID IS NULL THEN 1 ELSE 0 END) AS missing_customer_id_rows,
       SUM(CASE WHEN Quantity < 0 THEN 1 ELSE 0 END) AS negative_quantity_rows,
       SUM(CASE WHEN UnitPrice <= 0 THEN 1 ELSE 0 END) AS nonpositive_price_rows,
       SUM(CASE WHEN is_cancellation_invoice = 1 THEN 1 ELSE 0 END) AS cancellation_invoice_rows,
       SUM(CASE WHEN is_standard_sale_line = 1 THEN 1 ELSE 0 END) AS standard_sale_lines
FROM transactions;

-- 6. Repeat purchasing: customers with at least two distinct standard-sale invoices
SELECT CustomerID, COUNT(DISTINCT InvoiceNo) AS invoice_frequency,
       SUM(StandardSalesValue) AS historical_spend
FROM transactions
WHERE is_standard_sale_line = 1 AND CustomerID IS NOT NULL
GROUP BY CustomerID
HAVING COUNT(DISTINCT InvoiceNo) >= 2
ORDER BY invoice_frequency DESC, historical_spend DESC;
