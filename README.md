# Real-Time Object Detection — YOLOv4-tiny (OpenCV DNN)

Real-time webcam object detection and labeling using a pretrained YOLOv4-tiny model run through
OpenCV's DNN module. Built for the **Intel oneAPI Innovation Hackathon (March 2023)**.

## Overview

A live video feed is captured from a webcam, run frame-by-frame through a YOLOv4-tiny detector,
and annotated in real time with bounding boxes and class labels for every object recognized in
frame.

## Model

- **YOLOv4-tiny** (Darknet architecture), loaded via `cv2.dnn.readNet` and wrapped in
  `cv2.dnn_DetectionModel` for a simple `.detect()` inference call
- Pretrained on the **COCO dataset (80 object classes)** — see [`classes.txt`](classes.txt) —
  covering everyday categories: people, vehicles, animals, furniture, food items, electronics,
  and more
- Network input resized to **320×320** for fast inference; webcam captured at **1280×720**

## How it works (`LIVE_VIDEO.ipynb`)

1. Load the network: `cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")`
2. Read the 80 class names from `classes.txt`
3. Open the webcam: `cv2.VideoCapture(0)` at 1280×720
4. For every frame, run `model.detect(frame)` → returns `(class_id, confidence, bbox)` for each
   detected object
5. Draw a labeled bounding box per detection and render the live annotated feed with `cv2.imshow`

## Repository contents

| File | Purpose |
|---|---|
| `LIVE_VIDEO.ipynb` | Full detection loop — load model → read webcam → detect → annotate |
| `yolov4-tiny.cfg` | Darknet architecture config |
| `yolov4-tiny.weights` | Pretrained weights (COCO) |
| `classes.txt` | The 80 COCO class names the model recognizes |

## Getting started

```bash
pip install opencv-python
```

Open `LIVE_VIDEO.ipynb` in Jupyter, run all cells, and point a webcam at any of the 80 supported
object categories.

## Notes

This uses an out-of-the-box pretrained YOLOv4-tiny rather than a custom-trained model — a fast,
lightweight real-time detection baseline that runs comfortably on CPU without a dedicated GPU,
which made it a practical fit for a hackathon demo setting.

## Author

**Prantik Roy** — [GitHub](https://github.com/prantikroy1234) · [LinkedIn](https://www.linkedin.com/in/prantik-roy-9798371a9/)
