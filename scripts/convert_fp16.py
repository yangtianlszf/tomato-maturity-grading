import argparse
import os

import torch


def convert_to_fp16(input_path: str, output_path: str) -> None:
    print(f"Loading model from: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model" in ckpt and hasattr(ckpt["model"], "half"):
            ckpt["model"] = ckpt["model"].half()
            print(" - Converted 'model' to FP16")

        if "ema" in ckpt and ckpt["ema"] and hasattr(ckpt["ema"], "half"):
            ckpt["ema"] = ckpt["ema"].half()
            print(" - Converted 'ema' to FP16")
    elif isinstance(ckpt, torch.nn.Module):
        ckpt = ckpt.half()
        print(" - Converted standalone nn.Module to FP16")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(ckpt, output_path)

    src_size = os.path.getsize(input_path) / 1024 / 1024
    dst_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"Saved to: {output_path}")
    print(f"Size check: {src_size:.2f} MB -> {dst_size:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a PyTorch/Ultralytics checkpoint to FP16.")
    parser.add_argument("--input", required=True, help="Input model path")
    parser.add_argument("--output", required=True, help="Output model path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    convert_to_fp16(args.input, args.output)


if __name__ == "__main__":
    main()
