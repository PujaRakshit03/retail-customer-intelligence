"""Export auditable summaries and customer analysis outputs."""
from pathlib import Path
import json
import pandas as pd
from .data import audit_transactions, prepare_transactions, data_quality_summary
from .analytics import overview_metrics, group_performance
from .customers import build_rfm, add_opportunity_flags


def generate_reports(input_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    """Generate CSV/JSON reports from a source file without altering the source."""
    from .data import load_transactions
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = load_transactions(input_path)
    prepared = prepare_transactions(raw)
    audit = audit_transactions(raw)
    (outdir / "data_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    data_quality_summary(prepared).to_csv(outdir / "data_quality.csv", index=False)
    group_performance(prepared, "Description", limit=100).to_csv(outdir / "top_products.csv", index=False)
    group_performance(prepared, "Country", limit=100).to_csv(outdir / "country_performance.csv", index=False)
    rfm = build_rfm(prepared)
    add_opportunity_flags(rfm).to_csv(outdir / "customer_opportunities.csv", index=False)
    (outdir / "overview_metrics.json").write_text(json.dumps(overview_metrics(prepared), indent=2, default=str), encoding="utf-8")
    return {"output_directory": str(outdir), "files": ", ".join(p.name for p in outdir.iterdir() if p.is_file())}
