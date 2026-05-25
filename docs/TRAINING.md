# 训练与剪枝流程

本项目训练流程分为基础训练、稀疏约束训练、剪枝、微调和蒸馏五个阶段。

## 环境准备

```bash
pip install -r requirements.txt
```

需要准备：

- 竞赛数据集，整理为 YOLO 检测格式；
- YOLOv11 预训练权重，例如 `yolo11n.pt`；
- 与实验兼容的 `ultralytics` 代码。若使用修改版 `ultralytics`，可通过 `CUSTOM_ULTRALYTICS_PATH` 指定源码目录。

```bash
set CUSTOM_ULTRALYTICS_PATH=D:\path\to\custom\ultralytics-parent
```

Linux/macOS:

```bash
export CUSTOM_ULTRALYTICS_PATH=/path/to/custom/ultralytics-parent
```

## Step 1：基础训练

```bash
python src/train.py --step 1 --yaml configs/sdses_ssp.yaml --weights yolo11n.pt
```

关键配置：

- `imgsz=1024`，增强小目标检测；
- `optimizer=AdamW`；
- `mixup=0.3`、`copy_paste=0.5`；
- `cls=2.5`，增强类别不平衡场景下的分类损失。

## Step 2：稀疏约束训练

```bash
python src/train.py --step 2 --yaml configs/sdses_ssp.yaml
```

该阶段从 Step 1 最优权重加载。运行前需要在训练框架中开启 BN/L1 稀疏约束，否则剪枝效果不稳定。

## Step 3：剪枝

```bash
python src/train.py --step 3 --yaml configs/sdses_ssp.yaml --pruning-rate 0.4
```

默认输出：

```text
runs/nano/step3_pruning/weights/prune40.pt
```

本项目比赛实验中主要采用 40% 剪枝率，即剪除部分冗余通道后再评估精度与推理速度。

## Step 4：剪枝后微调

```bash
python src/train.py --step 4 --yaml configs/sdses_ssp.yaml
```

运行前需要关闭 Step 2 使用的 BN/L1 稀疏约束，避免微调阶段继续压缩权重导致精度恢复受阻。

## Step 5：蒸馏

```bash
python src/train.py --step 5 --yaml configs/sdses_ssp.yaml
```

默认使用 Step 1 模型作为教师模型，Step 4 模型作为学生模型。蒸馏是否带来收益取决于剪枝后模型精度和训练稳定性。

## FP16 导出

```bash
python scripts/convert_fp16.py \
  --input runs/nano/step3_pruning/weights/prune40.pt \
  --output runs/nano/prune40_fp16.pt
```

FP16 权重主要用于 CUDA 推理部署，CPU 推理建议使用 FP32 权重。
