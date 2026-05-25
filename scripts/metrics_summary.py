import argparse
import os
from pathlib import Path

import pandas as pd


def get_file_size(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024 if path.exists() else 0.0


def get_best_map(csv_path: Path) -> float:
    if not csv_path.exists():
        return 0.0

    df = pd.read_csv(csv_path)
    df.columns = [column.strip() for column in df.columns]
    for column in ("metrics/mAP50(B)", "metrics/mAP50"):
        if column in df.columns:
            return float(df[column].max())
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize training artifacts.")
    parser.add_argument("--runs-dir", default="runs/nano", help="Directory containing staged training outputs")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    stages = {
        "Step 1 (Base)": ("step1_base_train/weights/best.pt", "step1_base_train/results.csv"),
        "Step 2 (Constraint)": ("step2_constraint/weights/last.pt", "step2_constraint/results.csv"),
        "Step 3 (Pruning)": ("step3_pruning/weights/prune40.pt", None),
        "Step 4 (Finetune)": ("step4_finetune/weights/best.pt", "step4_finetune/results.csv"),
        "Step 5 (Distillation)": ("step5_distillation/weights/best.pt", "step5_distillation/results.csv"),
        "Final (FP16)": ("prune40_fp16.pt", None),
    }

    print(f"{'Stage':<25} | {'Size (MB)':<10} | {'mAP@0.5':<10}")
    print("-" * 55)
    for stage, (pt_rel, csv_rel) in stages.items():
        size = get_file_size(runs_dir / pt_rel)
        map50 = get_best_map(runs_dir / csv_rel) if csv_rel else 0.0
        print(f"{stage:<25} | {size:<10.2f} | {map50:<10.4f}")


if __name__ == "__main__":
    main()
