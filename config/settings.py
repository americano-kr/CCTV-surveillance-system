"""
================================================================
KONFIGURASI GLOBAL SISTEM CCTV SURVEILLANCE (v2.0 - UPDATED)
================================================================
Update v2.0:
  + Frame Skipping Optimization
  + Human Pose Detection (YOLOv8-Pose)
  + GPU Auto-Detection & FP16 Support
  + Person-Only Alert Mode
================================================================
"""

import os
from pathlib import Path


class Settings:
    """Kelas konfigurasi global sistem CCTV Surveillance v2.0."""

    # ============================================================
    # PATH DIREKTORI PROYEK
    # ============================================================
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_PATH: Path = BASE_DIR / "models" / "yolov8n.pt"
    POSE_MODEL_PATH: Path = BASE_DIR / "models" / "yolov8n-pose.pt"
    ROI_CONFIG_PATH: Path = BASE_DIR / "config" / "roi.json"
    DATASET_DIR: Path = BASE_DIR / "datasets"
    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_FILE_PATH: Path = BASE_DIR / "logs" / "intrusion_log.csv"

    # ============================================================
    # KONFIGURASI DETEKSI OBJEK (YOLOv8)
    # ============================================================
    DETECTION_CONFIDENCE: float = 0.40
    DETECTION_IOU: float = 0.45
    INFERENCE_IMGSZ: int = 640

    # Class COCO yang dideteksi
    ALLOWED_CLASSES: dict = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    # Hanya class ini yang bisa trigger ALERT (kendaraan tidak dihitung)
    ALERT_CLASSES: set = {0}  # Hanya "person"

    # ============================================================
    # KONFIGURASI POSE DETECTION (YOLOv8-Pose)
    # ============================================================
    # Aktifkan pose detection secara default
    POSE_DETECTION_ENABLED: bool = True

    # Confidence minimum untuk pose keypoint
    POSE_KEYPOINT_CONFIDENCE: float = 0.30

    # Confidence minimum untuk bbox manusia dalam pose model
    POSE_BBOX_CONFIDENCE: float = 0.35

    # Minimum keypoint terlihat agar pose dianggap valid
    POSE_MIN_VISIBLE_KEYPOINTS: int = 3

    # Nama model pose yang tersedia
    POSE_MODEL_OPTIONS: dict = {
        "yolov8n-pose (Nano - Tercepat)": "yolov8n-pose.pt",
        "yolov8s-pose (Small - Seimbang)": "yolov8s-pose.pt",
        "yolov8m-pose (Medium - Paling Akurat)": "yolov8m-pose.pt",
    }

    # ============================================================
    # KONFIGURASI TRACKING (ByteTrack)
    # ============================================================
    TRACKER_TYPE: str = "bytetrack.yaml"

    # ============================================================
    # FRAME SKIPPING OPTIMIZATION
    # ============================================================
    # Proses 1 dari setiap N frame
    # N=1: proses semua frame (paling akurat, paling lambat)
    # N=3: proses 1 dari 3 frame (3x lebih cepat, sedikit kurang akurat)
    # N=5: proses 1 dari 5 frame (5x lebih cepat, kurang akurat)
    #
    # PENJELASAN AKADEMIK:
    # Video 30fps dengan FRAME_SKIP=3:
    #   - Hanya 10 frame/detik yang diproses
    #   - ByteTrack tetap mempertahankan ID antar frame yang di-skip
    #   - Tradeoff: kecepatan 3x vs kemungkinan miss deteksi singkat
    #
    # RUMUS EFFECTIVE FPS:
    #   effective_fps = video_fps / FRAME_SKIP
    #   Contoh: 30fps / 3 = 10 effective fps pemrosesan
    FRAME_SKIP: int = 3

    # Range slider frame skip untuk UI
    FRAME_SKIP_MIN: int = 1
    FRAME_SKIP_MAX: int = 10

    # ============================================================
    # KONFIGURASI GPU / HARDWARE ACCELERATION
    # ============================================================
    # Auto-detect: gunakan GPU jika tersedia
    AUTO_DETECT_DEVICE: bool = True

    # Gunakan FP16 half precision jika GPU tersedia
    # Keuntungan: ~50% hemat VRAM, ~30-50% lebih cepat
    # Syarat: GPU NVIDIA dengan CUDA support
    USE_HALF_PRECISION: bool = False  # True jika GPU tersedia

    # ============================================================
    # KONFIGURASI MONITORING & ALERT
    # ============================================================
    ALERT_THRESHOLD_SECONDS: float = 5.0
    LOG_COOLDOWN_SECONDS: float = 10.0

    # ============================================================
    # KONFIGURASI DISPLAY VIDEO
    # ============================================================
    # Resolusi output display
    DISPLAY_WIDTH: int = 960
    DISPLAY_HEIGHT: int = 540

    # Resolusi input ke model (lebih kecil = lebih cepat)
    # 640x480 untuk performa optimal
    INFERENCE_WIDTH: int = 640
    INFERENCE_HEIGHT: int = 480

    # ============================================================
    # WARNA (Format BGR untuk OpenCV)
    # ============================================================
    COLOR_NORMAL: tuple = (0, 255, 0)          # Hijau - di luar ROI
    COLOR_INSIDE_ROI: tuple = (0, 165, 255)    # Oranye - dalam ROI
    COLOR_ALERT: tuple = (0, 0, 255)           # Merah - alert
    COLOR_ROI_BORDER: tuple = (0, 255, 255)    # Kuning - border ROI
    COLOR_ROI_FILL: tuple = (0, 255, 255)      # Kuning - fill ROI
    COLOR_POSE_SKELETON: tuple = (0, 200, 255) # Cyan - skeleton
    ROI_ALPHA: float = 0.15

    # ============================================================
    # KONFIGURASI FONT
    # ============================================================
    FONT_FACE: int = 0              # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE: float = 0.55
    FONT_THICKNESS: int = 2

    # ============================================================
    # MODEL SIZE OPTIONS (untuk UI dropdown)
    # ============================================================
    MODEL_OPTIONS: dict = {
        "YOLOv8n (Nano - Tercepat)": "yolov8n.pt",
        "YOLOv8s (Small - Seimbang)": "yolov8s.pt",
        "YOLOv8m (Medium - Paling Akurat)": "yolov8m.pt",
    }

    # ============================================================
    # CSV LOG HEADER
    # ============================================================
    LOG_CSV_HEADER: list = [
        "timestamp",
        "object_id",
        "object_type",
        "duration_seconds",
        "status",
        "detection_mode",   # 'bbox' atau 'pose'
    ]

    def __init__(self):
        """Inisialisasi: buat direktori yang diperlukan."""
        self._ensure_directories()
        self._auto_detect_gpu()

    def _ensure_directories(self) -> None:
        """Buat direktori sistem jika belum ada."""
        for d in [self.LOG_DIR, self.DATASET_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def _auto_detect_gpu(self) -> None:
        """Auto-detect GPU dan aktifkan optimasi jika tersedia."""
        if not self.AUTO_DETECT_DEVICE:
            return
        try:
            import torch
            if torch.cuda.is_available():
                self.USE_HALF_PRECISION = True
                gpu = torch.cuda.get_device_name(0)
                import logging
                logging.getLogger(__name__).info(
                    f"GPU terdeteksi: {gpu} - FP16 diaktifkan."
                )
        except ImportError:
            pass


# Singleton instance
settings = Settings()
