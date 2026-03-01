"""
Model audit utilities for segmentation checkpoints.

This module provides:
- full-dataset checkpoint audit with visual panels and summary metrics
- summary comparison between two checkpoints
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from clogic.ai_pytorch import _build_segmentation_model, binary_average_precision_score
from clogic.preprocessing_pytorch import preprocess_rgb_image


@dataclass
class AuditRunConfig:
    checkpoint: str
    output_root: str = "temp"
    batch_size: int = 64
    limit: int = 0
    top_k: int = 80
    save_panels: bool = True
    run_name: str | None = None


def _convert_mask_to_labels(mask: np.ndarray) -> np.ndarray:
    labels = np.zeros_like(mask, dtype=np.uint8)
    labels[mask == 127] = 1
    labels[mask == 255] = 2
    return labels


def _colorize_mask(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    out[labels == 1] = (0, 180, 0)
    out[labels == 2] = (0, 0, 220)
    return out


def _error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    out[(gt == 2) & (pred == 2)] = (0, 0, 255)
    out[(gt == 2) & (pred == 1)] = (255, 0, 255)
    out[(gt == 2) & (pred == 0)] = (128, 0, 255)
    out[(gt == 1) & (pred == 2)] = (0, 255, 255)
    out[(gt == 0) & (pred == 2)] = (0, 165, 255)
    out[(gt == 1) & (pred == 1)] = (0, 80, 0)
    return out


def _panel(img_bgr: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_c = _colorize_mask(gt)
    pred_c = _colorize_mask(pred)
    err_c = _error_map(gt, pred)
    row = np.concatenate([img_bgr, gt_c, pred_c, err_c], axis=1)

    h, w = row.shape[:2]
    header_h = 22
    canvas = np.zeros((h + header_h, w, 3), dtype=np.uint8)
    canvas[header_h:, :] = row
    labels = ["IMAGE", "GT", "PRED", "ERROR"]
    tile_w = img_bgr.shape[1]
    for i, txt in enumerate(labels):
        x = i * tile_w + 6
        cv2.putText(canvas, txt, (x, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def _calc_path_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    gt_path = gt == 2
    pred_path = pred == 2
    tp = int(np.sum(gt_path & pred_path))
    fp = int(np.sum(~gt_path & pred_path))
    fn = int(np.sum(gt_path & ~pred_path))
    tn = int(np.sum(~gt_path & ~pred_path))
    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 1.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1_den = (2 * tp + fp + fn)
    f1 = float((2 * tp) / f1_den) if f1_den > 0 else 1.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _cluster_core_issue_count(gt: np.ndarray, pred: np.ndarray, min_area: int = 25) -> int:
    """
    Count pathology components where prediction has small pathology core
    but large normal prediction around it.
    """
    mask = (gt == 2).astype(np.uint8)
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    issues = 0
    for lid in range(1, n):
        comp = labels == lid
        area = int(np.sum(comp))
        if area < min_area:
            continue
        pred_comp = pred[comp]
        red_ratio = float(np.mean(pred_comp == 2))
        normal_ratio = float(np.mean(pred_comp == 1))
        has_red = bool(np.any(pred_comp == 2))
        if has_red and red_ratio < 0.35 and normal_ratio > 0.4:
            issues += 1
    return issues


def _copy_top(src_all: Path, dst: Path, sample_rows: list[dict[str, Any]], key: str, top_k: int) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    rows = sorted(sample_rows, key=lambda r: float(r[key]), reverse=True)[: int(top_k)]
    for row in rows:
        src = src_all / str(row["file"])
        if src.exists():
            shutil.copy2(src, dst / src.name)


def _format_summary_text(summary: dict[str, Any]) -> str:
    pm = summary["pathology_metrics"]
    cluster = summary["cluster_core_issue"]
    lines = [
        "MODEL ERROR AUDIT",
        f"checkpoint: {summary['checkpoint']}",
        f"dataset_pairs: {summary['dataset_pairs']}",
        f"path_iou: {pm['iou']:.4f}",
        f"path_f1: {pm['f1']:.4f}",
        f"path_precision: {pm['precision']:.4f}",
        f"path_recall: {pm['recall']:.4f}",
        f"path_pr_auc: {pm['pr_auc']:.4f}",
        f"normal_to_path_px: {pm['normal_to_path_px']}",
        f"path_to_normal_px: {pm['path_to_normal_px']}",
        (
            "cluster_issue_samples: "
            f"{cluster['samples_with_issue']}/{summary['dataset_pairs']} "
            f"({cluster['share_of_samples'] * 100:.2f}%)"
        ),
        f"all_examples_dir: {summary['output_dirs']['all']}",
    ]
    return "\n".join(lines) + "\n"


def run_model_error_audit(cfg: Config, run_cfg: AuditRunConfig) -> Path:
    ckpt = Path(run_cfg.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    roi_dir = Path(cfg.DATASET_FOLDER) / cfg.IMAGES_FOLDER
    mask_dir = Path(cfg.DATASET_FOLDER) / cfg.MASKS_FOLDER
    if not roi_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Dataset dirs not found: {roi_dir} / {mask_dir}")

    files = sorted([f for f in os.listdir(roi_dir) if f.lower().endswith(".bmp")])
    files = [f for f in files if (mask_dir / f).exists()]
    if int(run_cfg.limit) > 0:
        files = files[: int(run_cfg.limit)]
    if not files:
        raise RuntimeError("No matched ROI/mask pairs found.")

    stamp = run_cfg.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(run_cfg.output_root) / f"model_error_audit_{stamp}"
    all_dir = out_dir / "all"
    top_fp_dir = out_dir / "top_normal_to_path"
    top_fn_dir = out_dir / "top_path_to_normal"
    top_cluster_dir = out_dir / "top_cluster_core_issue"
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(run_cfg.save_panels):
        all_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_segmentation_model(cfg, cfg.CLASSES).to(device)
    try:
        state = torch.load(str(ckpt), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state)
    model.eval()

    h, w = int(cfg.IMAGE_SHAPE[0]), int(cfg.IMAGE_SHAPE[1])
    n_classes = int(cfg.CLASSES)
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    sample_rows: list[dict[str, Any]] = []
    pr_probs: list[np.ndarray] = []
    pr_labels: list[np.ndarray] = []

    print(f"[audit] samples={len(files)} checkpoint={ckpt} device={device} input=({h},{w})")

    bs = max(1, int(run_cfg.batch_size))
    for start in range(0, len(files), bs):
        batch_files = files[start:start + bs]
        imgs_t = []
        gts = []
        vis_imgs = []
        used_names = []

        for name in batch_files:
            img_bgr = cv2.imread(str(roi_dir / name), cv2.IMREAD_COLOR)
            m_gray = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or m_gray is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if img_rgb.shape[:2] != (h, w):
                img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            gt = _convert_mask_to_labels(m_gray)
            if gt.shape[:2] != (h, w):
                gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

            img_f = preprocess_rgb_image(
                img_rgb,
                use_encoder_preprocessing=bool(cfg.PT_USE_ENCODER_PREPROCESSING),
                encoder_name=str(cfg.PT_ENCODER_NAME),
                encoder_weights=cfg.PT_ENCODER_WEIGHTS,
            )
            imgs_t.append(torch.from_numpy(img_f).float().permute(2, 0, 1))
            gts.append(gt.astype(np.uint8))
            vis_imgs.append(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            used_names.append(name)

        if not imgs_t:
            continue

        with torch.no_grad():
            x = torch.stack(imgs_t, dim=0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_hw = np.transpose(probs, (0, 2, 3, 1))
            preds = np.argmax(probs, axis=1).astype(np.uint8)

        for i, name in enumerate(used_names):
            gt = gts[i]
            pred = preds[i]
            pr_probs.append(probs_hw[i, :, :, 2].reshape(-1).astype(np.float32))
            pr_labels.append((gt == 2).astype(np.uint8).reshape(-1))

            idx = (gt.reshape(-1) * n_classes + pred.reshape(-1)).astype(np.int64)
            binc = np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
            conf += binc

            path_metrics = _calc_path_metrics(gt, pred)
            normal_to_path = int(np.sum((gt == 1) & (pred == 2)))
            path_to_normal = int(np.sum((gt == 2) & (pred == 1)))
            cluster_issue = _cluster_core_issue_count(gt, pred, min_area=25)

            out_name = f"{Path(name).stem}.png"
            if bool(run_cfg.save_panels):
                panel = _panel(vis_imgs[i], gt, pred)
                cv2.imwrite(str(all_dir / out_name), panel)

            sample_rows.append(
                {
                    "file": out_name,
                    "src_bmp": name,
                    "path_iou": path_metrics["iou"],
                    "path_f1": path_metrics["f1"],
                    "path_precision": path_metrics["precision"],
                    "path_recall": path_metrics["recall"],
                    "path_tp": path_metrics["tp"],
                    "path_fp": path_metrics["fp"],
                    "path_fn": path_metrics["fn"],
                    "normal_to_path_px": normal_to_path,
                    "path_to_normal_px": path_to_normal,
                    "cluster_core_issue_count": int(cluster_issue),
                }
            )

        done = min(start + bs, len(files))
        if done % max(bs, 256) == 0 or done == len(files):
            print(f"[audit] processed {done}/{len(files)}")

    tp_path = int(conf[2, 2])
    fp_path = int(conf[0, 2] + conf[1, 2])
    fn_path = int(conf[2, 0] + conf[2, 1])
    union = tp_path + fp_path + fn_path
    path_iou = float(tp_path / union) if union > 0 else 1.0
    path_prec = float(tp_path / (tp_path + fp_path)) if (tp_path + fp_path) > 0 else 1.0
    path_rec = float(tp_path / (tp_path + fn_path)) if (tp_path + fn_path) > 0 else 1.0
    path_f1 = float((2 * tp_path) / (2 * tp_path + fp_path + fn_path)) if (2 * tp_path + fp_path + fn_path) > 0 else 1.0
    pr_auc = float(binary_average_precision_score(np.concatenate(pr_probs), np.concatenate(pr_labels))) if pr_probs else 0.0

    mean_cluster_issue = float(np.mean([r["cluster_core_issue_count"] for r in sample_rows])) if sample_rows else 0.0
    samples_with_cluster_issue = int(np.sum([r["cluster_core_issue_count"] > 0 for r in sample_rows])) if sample_rows else 0

    summary = {
        "timestamp": stamp,
        "checkpoint": str(ckpt.resolve()),
        "dataset_pairs": len(sample_rows),
        "input_shape": [h, w],
        "confusion_matrix_rows_true_cols_pred": conf.tolist(),
        "pathology_metrics": {
            "iou": path_iou,
            "precision": path_prec,
            "recall": path_rec,
            "f1": path_f1,
            "pr_auc": pr_auc,
            "tp": tp_path,
            "fp": fp_path,
            "fn": fn_path,
            "normal_to_path_px": int(conf[1, 2]),
            "path_to_normal_px": int(conf[2, 1]),
        },
        "cluster_core_issue": {
            "samples_with_issue": samples_with_cluster_issue,
            "share_of_samples": float(samples_with_cluster_issue / max(1, len(sample_rows))),
            "mean_issue_components_per_sample": mean_cluster_issue,
        },
        "output_dirs": {
            "root": str(out_dir.resolve()),
            "all": str(all_dir.resolve()) if bool(run_cfg.save_panels) else "",
            "top_normal_to_path": str(top_fp_dir.resolve()) if bool(run_cfg.save_panels) else "",
            "top_path_to_normal": str(top_fn_dir.resolve()) if bool(run_cfg.save_panels) else "",
            "top_cluster_core_issue": str(top_cluster_dir.resolve()) if bool(run_cfg.save_panels) else "",
        },
    }

    csv_path = out_dir / "per_sample_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if sample_rows:
            writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sample_rows)

    if bool(run_cfg.save_panels):
        _copy_top(all_dir, top_fp_dir, sample_rows, "normal_to_path_px", top_k=int(run_cfg.top_k))
        _copy_top(all_dir, top_fn_dir, sample_rows, "path_to_normal_px", top_k=int(run_cfg.top_k))
        _copy_top(all_dir, top_cluster_dir, sample_rows, "cluster_core_issue_count", top_k=int(run_cfg.top_k))

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_txt = out_dir / "summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(_format_summary_text(summary))

    print(f"[audit] done. Summary: {summary_json}")
    if bool(run_cfg.save_panels):
        print(f"[audit] all examples: {all_dir}")
    return summary_json


def compare_audit_summaries(
    current_summary_path: str | Path,
    old_summary_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    current_path = Path(current_summary_path)
    old_path = Path(old_summary_path)
    with open(current_path, "r", encoding="utf-8") as f:
        current = json.load(f)
    with open(old_path, "r", encoding="utf-8") as f:
        old = json.load(f)

    cpm = current["pathology_metrics"]
    opm = old["pathology_metrics"]
    comparison = {
        "current_summary": str(current_path.resolve()),
        "old_summary": str(old_path.resolve()),
        "current_checkpoint": current["checkpoint"],
        "old_checkpoint": old["checkpoint"],
        "current": cpm,
        "old": opm,
        "delta_new_minus_old": {
            "iou": float(cpm["iou"] - opm["iou"]),
            "f1": float(cpm["f1"] - opm["f1"]),
            "pr_auc": float(cpm["pr_auc"] - opm["pr_auc"]),
            "precision": float(cpm["precision"] - opm["precision"]),
            "recall": float(cpm["recall"] - opm["recall"]),
            "normal_to_path_px": int(cpm["normal_to_path_px"] - opm["normal_to_path_px"]),
            "path_to_normal_px": int(cpm["path_to_normal_px"] - opm["path_to_normal_px"]),
            "cluster_issue_share": float(
                current["cluster_core_issue"]["share_of_samples"] - old["cluster_core_issue"]["share_of_samples"]
            ),
        },
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "comparison_summary.json"
        txt_path = out / "comparison_summary.txt"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("MODEL COMPARISON\n")
            f.write(f"current_checkpoint: {comparison['current_checkpoint']}\n")
            f.write(f"old_checkpoint: {comparison['old_checkpoint']}\n")
            f.write(
                "current "
                f"path_iou={cpm['iou']:.4f} path_f1={cpm['f1']:.4f} path_pr_auc={cpm['pr_auc']:.4f} "
                f"precision={cpm['precision']:.4f} recall={cpm['recall']:.4f} "
                f"n2p={cpm['normal_to_path_px']} p2n={cpm['path_to_normal_px']}\n"
            )
            f.write(
                "old     "
                f"path_iou={opm['iou']:.4f} path_f1={opm['f1']:.4f} path_pr_auc={opm['pr_auc']:.4f} "
                f"precision={opm['precision']:.4f} recall={opm['recall']:.4f} "
                f"n2p={opm['normal_to_path_px']} p2n={opm['path_to_normal_px']}\n"
            )
            d = comparison["delta_new_minus_old"]
            f.write(
                "delta(new-old) "
                f"path_iou={d['iou']:.4f} path_f1={d['f1']:.4f} path_pr_auc={d['pr_auc']:.4f} "
                f"precision={d['precision']:.4f} recall={d['recall']:.4f} "
                f"n2p={d['normal_to_path_px']} p2n={d['path_to_normal_px']}\n"
            )
            f.write(
                "cluster_issue_share "
                f"current={current['cluster_core_issue']['share_of_samples'] * 100:.2f}% "
                f"old={old['cluster_core_issue']['share_of_samples'] * 100:.2f}% "
                f"delta={d['cluster_issue_share'] * 100:.2f}pp\n"
            )
    return comparison
