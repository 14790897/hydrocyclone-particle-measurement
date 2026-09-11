# Model weights

The trained weights are stored in this repository via **Git LFS**:

```
weights/
├── yolov8-particle-best.pt   # Y-view particle detector  (YOLOv8)
├── yolov8_best.pt            # X-view particle detector  (YOLOv8, mixed-view training)
└── best_model_new_eff1.pth   # EfficientNet-B1 motion-state classifier
```

| File | Role |
|---|---|
| `yolov8-particle-best.pt` | Y-view detector |
| `yolov8_best.pt` | X-view detector |
| `best_model_new_eff1.pth` | EfficientNet-B1 classifier |

## Cloning

Install Git LFS once, then clone as usual — the weights are downloaded automatically:

```bash
git lfs install
git clone https://github.com/14790897/hydrocyclone-particle-measurement.git
```

If you cloned without LFS first, fetch the weights afterwards:

```bash
git lfs pull
```

The code expects these exact filenames under `weights/`. If you keep them elsewhere, adjust the
paths in `app.py` (the two `--yolo-model` arguments) and in `new/convert.py`.
