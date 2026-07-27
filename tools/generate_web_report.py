import json
import os
from pathlib import Path


def render_metrics(title: str, metrics: dict) -> str:
    lines = [f"<h2>{title}</h2>", "<table>", "<tr><th>Metric</th><th>Value</th></tr>"]
    lines.append(f"<tr><td>Total</td><td>{metrics.get('total', '')}</td></tr>")
    lines.append(f"<tr><td>Exact Match (EM%)</td><td>{metrics.get('exact_match_pct', '')}</td></tr>")
    lines.append(f"<tr><td>Execution Accuracy (EX%)</td><td>{metrics.get('execution_pct_overall', '')}</td></tr>")
    lines.append(f"<tr><td>EX Attempted</td><td>{metrics.get('ex_attempted', '')}</td></tr>")
    lines.append(f"<tr><td>Join Path Accuracy</td><td>{metrics.get('join_path_pct', '')}</td></tr>")
    lines.append("</table>")
    return "\n".join(lines)


def render_complexity(metrics: dict) -> str:
    lines = ["<h3>Per-complexity breakdown</h3>", "<table>", "<tr><th>Complexity</th><th>Total</th><th>EM (%)</th><th>EX (%)</th><th>VQR (%)</th></tr>"]
    for name, values in metrics.get("per_complexity", {}).items():
        lines.append(
            f"<tr><td>{name}</td><td>{values.get('total', '')}</td><td>{values.get('EM (%)', '')}</td><td>{values.get('EX (%)', '')}</td><td>{values.get('VQR (%)', '')}</td></tr>"
        )
    lines.append("</table>")
    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "outputs" / "results"
    spider_path = results_dir / "eval_summary_spider.json"
    bird_path = results_dir / "eval_summary.json"
    out_path = results_dir / "metrics_report.html"

    if not spider_path.exists() or not bird_path.exists():
        raise FileNotFoundError("Required summary JSON files not found in outputs/results")

    with open(spider_path, "r", encoding="utf-8") as f:
        spider_data = json.load(f)["metrics"]
    with open(bird_path, "r", encoding="utf-8") as f:
        bird_data = json.load(f)["metrics"]

    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>ConSQL Metrics Report</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;line-height:1.6;margin:40px;}",
        "table{border-collapse:collapse;width:100%;margin-bottom:24px;}",
        "th,td{border:1px solid #ccc;padding:8px;text-align:left;}",
        "th{background:#f4f4f4;}",
        "h1,h2,h3{color:#333;}",
        ".row{display:grid;grid-template-columns:1fr 1fr;gap:20px;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>ConSQL Metrics Report</h1>",
        "<p>This report shows computed metrics for BIRD and Spider predictions.</p>",
        "<div class='row'>",
        render_metrics("BIRD (dev_results.json)", bird_data),
        render_metrics("Spider (spider_dev_results.json)", spider_data),
        "</div>",
        render_complexity(bird_data),
        render_complexity(spider_data),
        "<h3>Paper reference values</h3>",
        "<ul>",
        "<li>Spider EX = 78.6%</li>",
        "<li>BIRD EX = 64.9%</li>",
        "</ul>",
        "</body>",
        "</html>",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"Wrote web report to {out_path}")

    try:
        import webbrowser
        webbrowser.open(out_path.as_uri())
        print("Opened web report in the default browser.")
    except Exception:
        print("Could not open browser automatically; open the file manually.")


if __name__ == "__main__":
    main()
