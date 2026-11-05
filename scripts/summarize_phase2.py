"""Summarise results/patchcore/*.json into a markdown table (mean over categories)."""
import json
from pathlib import Path

import numpy as np

rows = []
for f in sorted(Path("results/patchcore").glob("*.json")):
    d = json.loads(f.read_text())
    r = d["rows"]
    m = lambda k: np.mean([x[k] for x in r])  # noqa: E731
    per = " / ".join(f"{x['image_auroc']:.3f}" for x in r)
    rows.append(f"| {d['tag']} | {per} | {m('image_auroc'):.3f} | {m('pixel_auroc'):.3f} | {m('pro'):.3f} | "
                f"{m('recall_at_val_thr'):.2f} | {m('test_frr_at_val_thr'):.2f} | "
                f"{np.mean([x['latency_ms']['p50'] for x in r]):.0f} / "
                f"{np.mean([x['latency_ms']['p95'] for x in r]):.0f} / "
                f"{np.mean([x['latency_ms']['p99'] for x in r]):.0f} | {r[0]['bank_mb']:.1f} |")
hdr = ("| config | img AUROC bottle / nut / screw | mean img AUROC | mean pix AUROC | mean PRO | "
       "recall@val-thr | test FRR | latency ms p50/p95/p99 | bank MB |\n|---|---|---|---|---|---|---|---|---|\n")
out = hdr + "\n".join(rows)
Path("results/patchcore_summary.md").write_text(out + "\n")
print(out)
