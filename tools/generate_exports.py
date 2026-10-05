import json
import csv
import os

def write_csv(metrics_path, csv_path):
    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    m = data.get('metrics', {})

    # write top-level metrics
    rows = []
    rows.append(['metric', 'value'])
    for k, v in [('total', m.get('total')), ('exact_match', m.get('exact_match')), ('exact_match_pct', m.get('exact_match_pct')), ('execution_correct', m.get('execution_correct')), ('execution_pct_overall', m.get('execution_pct_overall')), ('execution_pct_attempted', m.get('execution_pct_attempted')), ('ex_attempted', m.get('ex_attempted')), ('join_path_correct', m.get('join_path_correct')), ('join_path_pct', m.get('join_path_pct'))]:
        rows.append([k, v])

    # per-complexity
    per = m.get('per_complexity', {})
    rows.append([])
    rows.append(['complexity', 'total', 'EM (%)', 'EX (%)', 'VQR (%)'])
    for name, stats in per.items():
        rows.append([name, stats.get('total'), stats.get('EM (%)'), stats.get('EX (%)'), stats.get('VQR (%)')])

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)


def try_plot(metrics_path, out_png):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib not available -- skipping plot')
        return False

    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    per = data.get('metrics', {}).get('per_complexity', {})
    names = list(per.keys())
    em = [per[n].get('EM (%)', 0.0) for n in names]
    ex = [per[n].get('EX (%)', 0.0) for n in names]

    x = range(len(names))
    plt.figure(figsize=(10, 5))
    plt.bar([i - 0.15 for i in x], em, width=0.3, label='EM (%)')
    plt.bar([i + 0.15 for i in x], ex, width=0.3, label='EX (%)')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Percent')
    plt.title('Per-Complexity EM / EX')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close()
    return True


def main():
    metrics = os.path.join('ConSQL', 'outputs', 'results', 'eval_summary.json')
    csv_out = os.path.join('ConSQL', 'outputs', 'results', 'eval_summary.csv')
    png_out = os.path.join('ConSQL', 'outputs', 'results', 'eval_summary.png')
    if not os.path.exists(metrics):
        print('metrics file not found:', metrics)
        return 2
    write_csv(metrics, csv_out)
    plotted = try_plot(metrics, png_out)
    if plotted:
        print('Wrote CSV and PNG:', csv_out, png_out)
    else:
        print('Wrote CSV:', csv_out)


if __name__ == '__main__':
    main()
