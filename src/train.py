import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
custom_ultralytics_path = os.getenv("CUSTOM_ULTRALYTICS_PATH")
if custom_ultralytics_path and Path(custom_ultralytics_path).is_dir():
    sys.path.insert(0, custom_ultralytics_path)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ultralytics import YOLO  # noqa: E402


PROJECT_DIR = PROJECT_ROOT / "runs" / "nano"
STEP1_NAME = "step1_base_train"
STEP2_NAME = "step2_constraint"
STEP3_NAME = "step3_pruning"
STEP4_NAME = "step4_finetune"
STEP5_NAME = "step5_distillation"


def train_kwargs(yaml_path: str, project: Path, name: str, amp: bool = True, lr0: float = 5e-4) -> dict:
    return {
        "data": yaml_path,
        "epochs": 500,
        "patience": 50,
        "imgsz": 1024,
        "batch": 4,
        "device": 0,
        "workers": 8,
        "cache": "ram",
        "optimizer": "AdamW",
        "lr0": lr0,
        "lrf": 1e-6,
        "momentum": 0.937,
        "weight_decay": 5e-4,
        "warmup_epochs": 5.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 2.5,
        "dfl": 1.5,
        "hsv_h": 0.03,
        "hsv_s": 0.9,
        "hsv_v": 0.6,
        "degrees": 20.0,
        "translate": 0.25,
        "scale": 0.7,
        "shear": 8.0,
        "perspective": 0.0002,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.3,
        "copy_paste": 0.5,
        "auto_augment": "randaugment",
        "erasing": 0.4,
        "dropout": 0.0,
        "val": True,
        "plots": False,
        "deterministic": False,
        "save": True,
        "save_period": 20,
        "cos_lr": True,
        "close_mosaic": 50,
        "amp": amp,
        "fraction": 1.0,
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "verbose": True,
    }


def step1_train(yaml_path: str, weights: str) -> None:
    model = YOLO(weights)
    kwargs = train_kwargs(yaml_path, PROJECT_DIR, STEP1_NAME, amp=True, lr0=5e-4)
    kwargs.update({"epochs": 500, "patience": 50, "save_period": 10, "close_mosaic": 20})
    model.train(**kwargs)


def step2_constraint_train(yaml_path: str) -> None:
    step1_best = PROJECT_DIR / STEP1_NAME / "weights" / "best.pt"
    if not step1_best.exists():
        raise FileNotFoundError(f"Step 1 weights not found: {step1_best}")

    print("Before Step 2, enable BN/L1 sparsity regularization in the training framework.")
    model = YOLO(str(step1_best))
    kwargs = train_kwargs(yaml_path, PROJECT_DIR, STEP2_NAME, amp=False, lr0=5e-4)
    kwargs.update({"epochs": 400, "patience": 80})
    model.train(**kwargs)


def step3_pruning(yaml_path: str, pruning_rate: float) -> None:
    from utils.yolo.det_pruning import do_pruning

    step2_last = PROJECT_DIR / STEP2_NAME / "weights" / "last.pt"
    output_path = PROJECT_DIR / STEP3_NAME / "weights" / f"prune{int(pruning_rate * 100)}.pt"
    if not step2_last.exists():
        raise FileNotFoundError(f"Step 2 weights not found: {step2_last}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        do_pruning(str(step2_last), str(output_path), pruning_rate=pruning_rate, yaml_path=yaml_path)
    except TypeError:
        do_pruning(str(step2_last), str(output_path), yaml_path, pruning_rate)


def step4_finetune(yaml_path: str, pruning_rate: float) -> None:
    pruned_path = PROJECT_DIR / STEP3_NAME / "weights" / f"prune{int(pruning_rate * 100)}.pt"
    if not pruned_path.exists():
        raise FileNotFoundError(f"Pruned weights not found: {pruned_path}")

    print("Before Step 4, disable BN/L1 sparsity regularization used in Step 2.")
    model = YOLO(str(pruned_path))
    for param in model.model.parameters():
        param.requires_grad = True

    kwargs = train_kwargs(yaml_path, PROJECT_DIR, STEP4_NAME, amp=True, lr0=1e-4)
    kwargs.update({"epochs": 400, "patience": 80, "warmup_epochs": 3.0, "warmup_bias_lr": 0.05})
    model.train(**kwargs)


def step5_distillation(yaml_path: str) -> None:
    teacher_path = PROJECT_DIR / STEP1_NAME / "weights" / "best.pt"
    student_path = PROJECT_DIR / STEP4_NAME / "weights" / "best.pt"
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher weights not found: {teacher_path}")
    if not student_path.exists():
        raise FileNotFoundError(f"Student weights not found: {student_path}")

    model_t = YOLO(str(teacher_path))
    model_s = YOLO(str(student_path))
    model_s.train(
        data=yaml_path,
        Distillation=model_t.model,
        loss_type="mgd",
        layers=["6", "8", "13", "16", "19", "22"],
        amp=False,
        imgsz=1024,
        epochs=300,
        batch=4,
        device=0,
        workers=0,
        lr0=0.001,
        project=str(PROJECT_DIR),
        save_period=10,
        name=STEP5_NAME,
        exist_ok=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged training and pruning entrypoint.")
    parser.add_argument("--step", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--yaml", required=True, help="Dataset YAML path")
    parser.add_argument("--weights", default="yolo11n.pt", help="Initial pretrained weights for Step 1")
    parser.add_argument("--pruning-rate", type=float, default=0.4, help="Channel pruning rate")
    args = parser.parse_args()

    if args.step == 1:
        step1_train(args.yaml, args.weights)
    elif args.step == 2:
        step2_constraint_train(args.yaml)
    elif args.step == 3:
        step3_pruning(args.yaml, args.pruning_rate)
    elif args.step == 4:
        step4_finetune(args.yaml, args.pruning_rate)
    elif args.step == 5:
        step5_distillation(args.yaml)


if __name__ == "__main__":
    main()
