"""
================================================================
APP.PY v2.0 - ENTRY POINT CCTV SURVEILLANCE SYSTEM
================================================================
Cara menjalankan:
    streamlit run app.py

Browser: http://localhost:8501

================================================================
FLOWCHART SISTEM v2.0 (dengan Frame Skip + Pose)
================================================================

START
  │
  ▼
Load Settings + ROI dari roi.json
  │
  ▼
User Konfigurasi Sidebar:
  ├─ Sumber Video (Dataset/Upload/Webcam)
  ├─ Model YOLOv8 (n/s/m)
  ├─ Frame Skip (1–10)
  ├─ Confidence Threshold
  ├─ Enable Pose Detection (y/n)
  └─ Pose Model (n-pose/s-pose)
  │
  ▼
Tekan START → Buka VideoCapture
  │
  ▼
╔══════════════════════════════════╗
║   LOOP FRAME                    ║
║                                  ║
║  frame_count += 1                ║
║                                  ║
║  ┌─────────────────────────────┐ ║
║  │ FRAME SKIPPING CHECK        │ ║
║  │ if frame_count % N != 0:   │ ║
║  │     continue (lewati)       │ ║
║  └────────────┬────────────────┘ ║
║               │                  ║
║  ┌────────────▼────────────────┐ ║
║  │ YOLOv8 Detection + ByteTrack│ ║
║  │ → list[DetectionResult]     │ ║
║  └────────────┬────────────────┘ ║
║               │                  ║
║  ┌────────────▼────────────────┐ ║
║  │ Pose Detection (opsional)   │ ║
║  │ YOLOv8-Pose + ByteTrack     │ ║
║  │ → list[PoseResult]          │ ║
║  └────────────┬────────────────┘ ║
║               │                  ║
║  ┌────────────▼────────────────┐ ║
║  │ IntrusionMonitor.process()  │ ║
║  │  FOR setiap objek:          │ ║
║  │    centroid = pose > bbox   │ ║
║  │    cek ROI polygon          │ ║
║  │    update timer             │ ║
║  │    IF person + dur>thresh:  │ ║
║  │       status = ALERT        │ ║
║  │       log ke CSV            │ ║
║  └────────────┬────────────────┘ ║
║               │                  ║
║  ┌────────────▼────────────────┐ ║
║  │ VideoRenderer.render()      │ ║
║  │  - ROI overlay              │ ║
║  │  - Skeleton pose            │ ║
║  │  - Bounding boxes           │ ║
║  │  - Labels + durasi          │ ║
║  │  - Alert banner (jika ada)  │ ║
║  └────────────┬────────────────┘ ║
║               │                  ║
║  Tampilkan di Streamlit          ║
╚══════════════╪═══════════════════╝
               │ (next frame)
               ▼
             STOP
================================================================
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")
logger.info("=" * 60)
logger.info("CCTV Surveillance System v2.0 - Starting")
logger.info("Features: Frame Skip + Pose Detection + Person Alert")
logger.info("=" * 60)

from ui.streamlit_ui import main

if __name__ == "__main__":
    main()
else:
    main()
