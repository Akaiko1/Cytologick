#!/usr/bin/env python3
"""
Export PyTorch model to ONNX format for C++ inference.

Usage:
    python export_onnx.py [--input MODEL_PATH] [--output OUTPUT_PATH]

Example:
    python export_onnx.py --input ../_main/new_best.pth --output ../_main/model.onnx
"""

import argparse
import sys
from pathlib import Path

import torch
import segmentation_models_pytorch as smp


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


def create_model(num_classes: int = 3) -> torch.nn.Module:
    """Create U-Net model matching the training architecture."""
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,  # Will load from state dict
        classes=num_classes,
        activation=None,  # Returns logits
    )
    return model


def export_to_onnx(
    model_path: Path,
    output_path: Path,
    num_classes: int = 3,
    input_size: tuple[int, int] = (128, 128),
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
    print(f"Loading model from: {model_path}")

    # Create model architecture
    model = create_model(num_classes)

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

    # Verify the exported model
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("  - ONNX model verification: passed")
    except ImportError:
        print("  - ONNX verification skipped (onnx package not installed)")
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
        default=3,
        help="Number of output classes (default: 3)",
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=128,
        help="Input size (height=width, default: 128)",
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

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = model_path.with_suffix(".onnx")

    # Export
    success = export_to_onnx(
        model_path=model_path,
        output_path=output_path,
        num_classes=args.classes,
        input_size=(args.size, args.size),
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
