import os
import sys
import time
import argparse
import cv2
import numpy as np
import paddle

# Suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

from ppdet.core.workspace import load_config, create
from ppdet.utils.checkpoint import load_pretrain_weight

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Generate distinct colors for each class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype="uint8")

def parse_args():
    parser = argparse.ArgumentParser(description="PaddleDetection Live Webcam Detection")
    parser.add_argument("-c", "--config", default="configs/picodet/picodet_s_320_coco_lcnet.yml", help="Path to config file")
    parser.add_argument("-w", "--weights", default="/Users/devarshchauahn/.cache/paddle/weights/picodet_s_320_coco_lcnet.pdparams", help="Path to model weights")
    parser.add_argument("--camera_id", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    return parser.parse_args()

def main():
    args = parse_args()
    print("[INFO] Loading PaddleDetection model on CPU...")
    paddle.set_device("cpu")

    cfg = load_config(args.config)
    cfg.weights = args.weights
    model = create(cfg.architecture)
    load_pretrain_weight(model, cfg.weights)
    model.eval()

    print(f"[INFO] Opening camera {args.camera_id}...")
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam with ID {args.camera_id}.")
        print("[HINT] If you are using an external camera or Continuity Camera, try --camera_id 1")
        sys.exit(1)

    print("[INFO] Live detection started!")
    print("[INFO] Press 'q' or ESC in the video window to quit.")

    fps = 0.0
    prev_time = time.time()

    # Pre-calculate normalization constants
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from camera.")
            break

        orig_h, orig_w = frame.shape[:2]

        # Resize and preprocess frame to 320x320
        input_size = 320
        resized = cv2.resize(frame, (input_size, input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
        tensor_img = paddle.to_tensor(normalized.transpose((2, 0, 1))[np.newaxis, ...], dtype="float32")

        scale_y = float(input_size) / orig_h
        scale_x = float(input_size) / orig_w

        data = {
            "image": tensor_img,
            "im_shape": paddle.to_tensor([[float(input_size), float(input_size)]], dtype="float32"),
            "scale_factor": paddle.to_tensor([[scale_y, scale_x]], dtype="float32")
        }

        with paddle.no_grad():
            outs = model(data)

        # Parse detection outputs
        bboxes = outs["bbox"].numpy()
        bbox_num = outs["bbox_num"].numpy()[0]

        detections = bboxes[:bbox_num] if bbox_num > 0 else []

        for det in detections:
            cls_id, score, xmin, ymin, xmax, ymax = det
            cls_id = int(cls_id)
            if score < args.threshold:
                continue

            xmin = int(max(0, xmin))
            ymin = int(max(0, ymin))
            xmax = int(min(orig_w, xmax))
            ymax = int(min(orig_h, ymax))

            color = [int(c) for c in COLORS[cls_id % len(COLORS)]]
            label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"Class {cls_id}"
            caption = f"{label}: {score:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (xmin, max(0, ymin - th - 8)), (xmin + tw + 6, max(0, ymin)), color, -1)
            cv2.putText(frame, caption, (xmin + 3, max(0, ymin - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Calculate and draw FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to exit", (15, orig_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("PaddleDetection - Live Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera closed. Done!")

if __name__ == "__main__":
    main()
