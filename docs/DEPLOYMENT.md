# 推理部署说明

推理接口位于 `src/inference.py`，提供三个平台友好的函数：

- `init(model_path)`：加载模型；
- `get_status()`：返回模型状态；
- `process_image(image_path, args=None, **kwargs)`：对单张图片推理并返回结构化结果。

## 本地调用

```python
from src.inference import init, get_status, process_image

init("runs/nano/prune40_fp16.pt")
print(get_status())
print(process_image("demo.jpg"))
```

## 返回格式

```json
{
  "objects": [
    {
      "label": "RipeStage",
      "prob": 0.92,
      "type": 1,
      "points": [
        {"x": 10, "y": 20},
        {"x": 110, "y": 20},
        {"x": 110, "y": 120},
        {"x": 10, "y": 120}
      ]
    }
  ],
  "classifications": [],
  "segmentations": []
}
```

## 性能优化点

- CUDA 环境下启用 FP16 推理；
- 初始化阶段进行 warmup，降低首次推理延迟；
- 对 4K/6K 大图使用 `PIL.Image.draft` 进行快速降采样解码；
- 使用 `torch.inference_mode()` 降低推理开销；
- 根据平台 CPU 资源设置 `torch.set_num_threads()`。

## 平台部署

如评测平台要求固定接口文件名，可将 `src/inference.py` 改名或复制为平台要求的入口文件，并将模型权重放到平台指定路径。

如果平台使用自定义 `ultralytics` 代码，请设置：

```bash
CUSTOM_ULTRALYTICS_PATH=/path/to/ultralytics-parent
```

该路径应是 `ultralytics/` 包所在目录的上一级。
