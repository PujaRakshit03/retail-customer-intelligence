# Retail Customer Intelligence

A local-first retail analytics and customer opportunity discovery application built on the real [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail). It combines data-quality checks, sales/product/country analytics, explainable customer segments, purchase-interval analysis, product pairs, cohort retention, and a chronological segment holdout check.

## Important analytical limits

The dataset contains no product costs or operating expenses. This project reports **recorded sales value**, not profit or margin. Customer flags identify accounts worth investigating; they are not certain churn predictions or causal estimates. Cancellation invoice prefixes and negative quantities are retained as data-quality indicators rather than assumed to have a single interpretation.

## Setup (Windows PowerShell)

1. Clone/open this repository in VS Code.
2. Download the UCI Online Retail file and place it at `data/raw/Online Retail.xlsx`.
3. Run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can avoid changing execution policy by running `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe -m streamlit run app.py`.

## Run the dashboard

```powershell
streamlit run app.py
```

The dashboard reads the default file above. Alternatively, use **Upload a dataset** in the sidebar to load an Excel or CSV file locally.

## Run tests

```powershell
python -m pytest -q
```

Tests use small deterministic fixtures to verify logic. Fixture values are not findings from the retail dataset.

## Generate reproducible reports

```powershell
python -m retail_intelligence.cli --input "data/raw/Online Retail.xlsx" --output reports/generated
```

If the package is not installed in editable mode, use:

```powershell
$env:PYTHONPATH = "src"
python -m retail_intelligence.cli --input "data/raw/Online Retail.xlsx" --output reports/generated
```

Generated reports include a JSON audit, data-quality summary, top products, country performance, overview metrics, and customer opportunity flags. Keep generated reports out of Git if they contain detailed customer-level information.

## SQL

`sql/schema.sql` defines a SQLite-compatible table for prepared transaction records. `sql/analytics.sql` contains example queries. Import the prepared dataset into that schema before running the queries; these SQL files are examples, not an automatic database loader.

## Project map

- `app.py` — Streamlit user interface.
- `src/retail_intelligence/data.py` — loading, validation, flags, audit.
- `src/retail_intelligence/analytics.py` — metrics and aggregations.
- `src/retail_intelligence/customers.py` — RFM and explainable customer segment/opportunity rules.
- `src/retail_intelligence/reporting.py` and `cli.py` — generated reports.
- `tests/` — deterministic tests.
- `docs/USER_SOP.md` — run and maintenance procedure.
- `docs/TROUBLESHOOTING.md` — common failures.

## Reproducibility and privacy

Raw and processed data folders are ignored by Git. Do not commit the source dataset, customer-level exports, credentials, or other sensitive outputs. Inspect `git status` before committing.

## Added analytical validation

The **Customer behaviour** tab includes:

- Monthly cohort retention based on each identified customer's first *observed* purchase month.
- Median time between distinct orders for repeat customers.
- Product pairs that appear on the same invoice.
- Customer segments built using only pre-cutoff history, then compared with purchases in a later holdout window.
- Cancellation-invoice patterns by month and product.
- A simple repeat-purchase baseline and monthly cohort retention.

These are descriptive historical checks, not causal analysis or a production churn model. The current dataset ends on 2011-12-09, so holdout results depend on the chosen horizon and can be sensitive to seasonality. Product pairs indicate co-occurrence, not proof of complementarity.
