#!/usr/bin/env python3
"""
Export PyTorch model to ONNX format for C++ inference.

Usage:
    python export_onnx.py [--input MODEL_PATH] [--output OUTPUT_PATH]

Example:
    python export_onnx.py --input ../_main/new_best.pth --output ../_main/model.onnx
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

# Make project root importable when script is run from cpp/scripts.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import load_config, get_image_shape
from clogic.ai_pytorch import _build_segmentation_model


def find_model_file(search_dir: Path) -> Path | None:
    """Find model file in directory using standard search order."""
    search_order = [
        "new_best.pth",
        "_new_best.pth",
        "new_final.pth",
        "_new_final.pth",
        "new_last.pth",
        "_new_last.pth",
        "model.pth",
        "model_best.pth",
        "model_final.pth",
    ]

    for name in search_order:
        path = search_dir / name
        if path.exists():
            return path

    # Fallback: find any .pth file
    pth_files = list(search_dir.glob("*.pth"))
    if pth_files:
        return pth_files[0]

    return None


def create_model(cfg, num_classes: int = 3) -> torch.nn.Module:
    """Create model matching current project config architecture."""
    model = _build_segmentation_model(cfg, num_classes)
    return model


def export_to_onnx(
    cfg,
    model_path: Path,
    output_path: Path,
    num_classes: int = 3,
    input_size: tuple[int, int] | None = None,
    opset_version: int = 17,
) -> bool:
    """
    Export PyTorch model to ONNX format.

    Args:
        model_path: Path to .pth state dict file
        output_path: Path for output .onnx file
        num_classes: Number of output classes
        input_size: Model input size (height, width)
        opset_version: ONNX opset version

    Returns:
        True if export successful
    """
    input_size = get_image_shape(cfg) if input_size is None else input_size

    print(f"Loading model from: {model_path}")

    # Create model architecture
    model = create_model(cfg, num_classes)

    # Load state dict
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  - Input size: {input_size}")
    print(f"  - Output classes: {num_classes}")

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    # Export to ONNX
    print(f"Exporting to: {output_path}")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
    except Exception as e:
        print(f"Export failed: {e}")
        return False

    print(f"Export successful!")
    print(f"  - Output file: {output_path}")
    print(f"  - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Verify in a separate process so native checker crashes don't abort export.
    verify_cmd = [
        sys.executable,
        "-c",
        (
            "import onnx; "
            f"m=onnx.load(r'{str(output_path)}'); "
            "onnx.checker.check_model(m); "
            "print('ok')"
        ),
    ]
    try:
        proc = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            print("  - ONNX model verification: passed")
        else:
            detail = (proc.stderr or proc.stdout or "").strip()
            if not detail:
                detail = f"checker exited with code {proc.returncode}"
            print(f"  - ONNX verification warning: {detail}")
    except Exception as e:
        print(f"  - ONNX verification warning: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch model to ONNX format"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Path to input .pth model file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Path for output .onnx file",
    )
    parser.add_argument(
        "--classes",
        "-c",
        type=int,
        default=None,
        help="Number of output classes (default: from config)",
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=None,
        help="Input size (height=width, default: from config IMAGE_SHAPE)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "config.yaml",
        help="Path to project config.yaml (default: <repo>/config.yaml)",
    )

    args = parser.parse_args()

    # Find input model
    if args.input:
        model_path = args.input
    else:
        # Search in _main directory
        script_dir = Path(__file__).parent
        main_dir = script_dir.parent.parent / "_main"

        if not main_dir.exists():
            main_dir = Path.cwd() / "_main"

        model_path = find_model_file(main_dir)

        if model_path is None:
            print(f"Error: No model file found in {main_dir}")
            print("Specify model path with --input")
            sys.exit(1)

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    # Load config to get architecture/input defaults.
    cfg = load_config(str(args.config))

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = model_path.with_suffix(".onnx")

    num_classes = int(args.classes) if args.classes is not None else int(cfg.CLASSES)
    if args.size is not None:
        input_size = (int(args.size), int(args.size))
    else:
        input_size = (int(cfg.IMAGE_SHAPE[0]), int(cfg.IMAGE_SHAPE[1]))

    print(
        f"Using config: arch={getattr(cfg, 'PT_MODEL_ARCH', 'unknown')}, "
        f"encoder={getattr(cfg, 'PT_ENCODER_NAME', 'unknown')}, "
        f"classes={num_classes}, input={input_size}"
    )

    # Export
    success = export_to_onnx(
        cfg=cfg,
        model_path=model_path,
        output_path=output_path,
        num_classes=num_classes,
        input_size=input_size,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
