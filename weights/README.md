# Model weights

The trained weights are **not stored in this repository** (they are large binaries and are
hosted on Kaggle together with the datasets). Download them and place them in this folder:

```
weights/
├── yolov8-particle-best.pt   # Y-view particle detector  (YOLOv8)
├── yolov8_best.pt            # X-view particle detector  (YOLOv8, mixed-view training)
└── best_model_new_eff1.pth   # EfficientNet-B1 motion-state classifier
```

| File | Role | Kaggle source |
|---|---|---|
| `yolov8-particle-best.pt` | Y-view detector | _link_ |
| `yolov8_best.pt` | X-view detector | _link_ |
| `best_model_new_eff1.pth` | EfficientNet-B1 classifier | _link_ |

The code expects these exact filenames under `weights/`. If you keep them elsewhere, adjust the
paths in `app.py` (the two `--yolo-model` arguments) and in `new/convert.py`.
