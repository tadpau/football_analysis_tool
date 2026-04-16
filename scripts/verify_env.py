"""One-shot environment verification.

Run:
    python scripts/verify_env.py

Checks:
  - Python version is 3.10-3.12 (not 3.14)
  - All required imports load
  - torch reports whether a usable GPU exists
  - ultralytics can download + instantiate yolov8m weights
"""
from __future__ import annotations

import sys


REQUIRED = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
    ("supervision", "supervision"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
    ("yaml", "pyyaml"),
]


def main() -> int:
    print(f"Python: {sys.version}")
    if sys.version_info[:2] not in {(3, 10), (3, 11), (3, 12)}:
        print("  ⚠  Expected Python 3.10-3.12; ultralytics/torch may not load.")

    missing: list[str] = []
    for module, package in REQUIRED:
        try:
            __import__(module)
            print(f"  ok   {module}")
        except ImportError as e:
            print(f"  MISS {module}  ({package})  — {e}")
            missing.append(package)

    if missing:
        print(f"\nInstall the missing packages: pip install {' '.join(missing)}")
        return 1

    import torch  # noqa: E402
    print(f"\ntorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch, "xpu"):
        print(f"  XPU (Intel) available: {torch.xpu.is_available()}")

    print("\nDownloading yolov8m.pt (≈ 50 MB) — first run only ...")
    from ultralytics import YOLO  # noqa: E402
    model = YOLO("yolov8m.pt")
    print(f"  model loaded: {type(model.model).__name__}, "
          f"{len(model.names)} classes (COCO)")
    print("\nEnvironment OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
