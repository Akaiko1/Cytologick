#!/usr/bin/env python3
"""
Official runner for segmentation model audit and checkpoint comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from config import load_config
from clogic.model_audit import AuditRunConfig, compare_audit_summaries, run_model_error_audit


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit segmentation checkpoint on full dataset. "
            "Provide --old-checkpoint to run side-by-side comparison."
        )
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--checkpoint", default="_new_best.pth", help="Current checkpoint path (.pth)")
    p.add_argument("--old-checkpoint", default="", help="Optional old checkpoint path (.pth) for comparison")
    p.add_argument("--output-root", default="temp", help="Output root directory")
    p.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    p.add_argument("--limit", type=int, default=0, help="Optional max samples (0=all)")
    p.add_argument("--top-k", type=int, default=80, help="Top samples to copy per error bucket")
    p.add_argument("--no-panels", action="store_true", help="Do not save image/gt/pred/error panels")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    save_panels = not bool(args.no_panels)

    old_ckpt = str(args.old_checkpoint or "").strip()
    if old_ckpt:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        compare_root = Path(args.output_root) / f"model_error_compare_{stamp}"
        current_root = compare_root / "current"
        old_root = compare_root / "old"

        current_summary = run_model_error_audit(
            cfg,
            AuditRunConfig(
                checkpoint=args.checkpoint,
                output_root=str(current_root),
                batch_size=args.batch_size,
                limit=args.limit,
                top_k=args.top_k,
                save_panels=save_panels,
            ),
        )
        old_summary = run_model_error_audit(
            cfg,
            AuditRunConfig(
                checkpoint=old_ckpt,
                output_root=str(old_root),
                batch_size=args.batch_size,
                limit=args.limit,
                top_k=args.top_k,
                save_panels=save_panels,
            ),
        )

        comparison = compare_audit_summaries(
            current_summary_path=current_summary,
            old_summary_path=old_summary,
            output_dir=compare_root,
        )
        print(f"[audit] comparison done: {compare_root / 'comparison_summary.json'}")
        d = comparison["delta_new_minus_old"]
        print(
            "[audit] delta(new-old): "
            f"iou={d['iou']:.4f}, f1={d['f1']:.4f}, pr_auc={d['pr_auc']:.4f}, "
            f"precision={d['precision']:.4f}, recall={d['recall']:.4f}"
        )
        return 0

    summary = run_model_error_audit(
        cfg,
        AuditRunConfig(
            checkpoint=args.checkpoint,
            output_root=args.output_root,
            batch_size=args.batch_size,
            limit=args.limit,
            top_k=args.top_k,
            save_panels=save_panels,
        ),
    )
    print(f"[audit] completed: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
