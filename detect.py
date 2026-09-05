import os
import sys
import subprocess

photos_dir = "my_photos"
valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

if not os.path.exists(photos_dir):
    os.makedirs(photos_dir, exist_ok=True)

files = [f for f in os.listdir(photos_dir) if f.lower().endswith(valid_exts)]

if not files:
    print(f"[INFO] No photos found in "{photos_dir}/" yet.")
    print(f"[HINT] Run: open my_photos")
    print(f"[HINT] Drag & drop your photos into that folder, then run: python detect.py")
    sys.exit(0)

print(f"[INFO] Found {len(files)} photo(s) in "{photos_dir}/": {files}")
print("[INFO] Running object detection on your photos...")

cmd = [
    sys.executable, "tools/infer.py",
    "-c", "configs/picodet/picodet_s_320_coco_lcnet.yml",
    "-o", "use_gpu=false",
    "weights=/Users/devarshchauahn/.cache/paddle/weights/picodet_s_320_coco_lcnet.pdparams",
    f"--infer_dir={photos_dir}",
    "--output_dir=output",
    "--draw_threshold=0.5"
]

res = subprocess.run(cmd)

if res.returncode == 0:
    print("\n[SUCCESS] All detections completed! Opening results folder...")
    subprocess.run(["open", "output"])
