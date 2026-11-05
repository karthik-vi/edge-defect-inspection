"""Summarise results/edge/*.json -> results/edge_summary.md (latency vs accuracy vs memory)."""
import json
from pathlib import Path

USD_PER_VCPU_HOUR = 0.04  # ASSUMPTION (placeholder list-price-like value), not measured: edit for your cloud/edge box
order = ["torch", "ort_fp32", "ort_int8", "ov_fp32", "ov_int8"]
rows = []
for b in order:
    p = Path(f"results/edge/{b}.json")
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    L = d["latency_ms_by_threads"]
    cost = L["4"]["p50"] / 1e3 * 1000 / 3600 * 4 * USD_PER_VCPU_HOUR  # 4 vCPU busy for p50 s per part
    rows.append(f"| {b} | {d['mean_image_auroc']:.4f} | {d['mean_pixel_auroc']:.4f} | {d['mean_recall_at_val_thr']:.2f} | "
                + " | ".join(f"{L[t]['p50']:.0f} / {L[t]['p99']:.0f}" for t in ("1", "2", "4"))
                + f" | {1000 / L['4']['p50']:.0f} | {d['model_mb'] or 11.0:.1f} | {d['bank_mb']:.1f} | ${cost:.4f} |")
hdr = ("| backend | img AUROC | pix AUROC | recall@val-thr | 1 thr p50/p99 ms | 2 thr | 4 thr | parts/s @4 | "
       "model MB | bank MB | $/1k parts @4 vCPU (assumed $0.04/vCPU-h) |\n" + "|---|" * 1 + "---|" * 10 + "\n")
Path("results/edge_summary.md").write_text(hdr + "\n".join(rows) + "\n")
print(hdr + "\n".join(rows))
