"""
================================================================
MODUL DETEKSI OBJEK MENGGUNAKAN YOLOV8
================================================================
Modul ini bertanggung jawab untuk mendeteksi objek pada setiap
frame video menggunakan model YOLOv8 pretrained COCO.

Kelas utama:
    ObjectDetector  - Wrapper untuk model YOLOv8

Contoh penggunaan:
    detector = ObjectDetector()
    detections = detector.detect(frame)
    for det in detections:
        print(det['class_name'], det['bbox'], det['confidence'])
================================================================
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

# ============================================================
# SETUP LOGGING INTERNAL
# ============================================================
logger = logging.getLogger(__name__)


class DetectionResult:
    """
    Data class untuk menyimpan hasil deteksi satu objek.

    Attributes:
        bbox        : Bounding box (x1, y1, x2, y2) dalam pixel
        confidence  : Confidence score deteksi (0.0 - 1.0)
        class_id    : ID kelas dari COCO dataset
        class_name  : Nama kelas (misal: 'person', 'car')
        track_id    : ID tracking objek (None jika belum ditrack)
    """

    def __init__(
        self,
        bbox: tuple[int, int, int, int],
        confidence: float,
        class_id: int,
        class_name: str,
        track_id: Optional[int] = None,
    ):
        self.bbox: tuple[int, int, int, int] = bbox
        self.confidence: float = confidence
        self.class_id: int = class_id
        self.class_name: str = class_name
        self.track_id: Optional[int] = track_id

    @property
    def centroid(self) -> tuple[int, int]:
        """
        Menghitung centroid (titik tengah) dari bounding box.

        Rumus:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

        Returns:
            Tuple (cx, cy) koordinat centroid
        """
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return (cx, cy)

    def __repr__(self) -> str:
        return (
            f"DetectionResult("
            f"id={self.track_id}, "
            f"class={self.class_name}, "
            f"conf={self.confidence:.2f}, "
            f"bbox={self.bbox})"
        )


class ObjectDetector:
    """
    Detektor objek berbasis YOLOv8 dengan ByteTrack.

    Kelas ini membungkus (wrap) model Ultralytics YOLOv8
    dan menyediakan interface yang sederhana untuk:
    - Deteksi objek per frame
    - Tracking objek lintas frame (dengan ID konsisten)
    - Filtering kelas yang relevan

    Attributes:
        model_path      : Path ke file model .pt
        confidence      : Minimum confidence threshold
        allowed_classes : Dict {class_id: class_name} yang diizinkan
        model           : Instance model Ultralytics YOLO
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.40,
        iou: float = 0.45,
        imgsz: int = 640,
        allowed_classes: Optional[dict] = None,
    ):
        """
        Inisialisasi detektor YOLOv8.

        Args:
            model_path      : Path ke file model YOLOv8 (.pt)
            confidence      : Minimum confidence score (default: 0.40)
            iou             : IoU threshold NMS (default: 0.45)
            imgsz           : Ukuran input inferensi (default: 640)
            allowed_classes : Dict class_id yang dideteksi
        """
        # ============================================================
        # SIMPAN PARAMETER KONFIGURASI
        # ============================================================
        self.model_path: Path = Path(model_path)
        self.confidence: float = confidence
        self.iou: float = iou
        self.imgsz: int = imgsz

        # Default: person, bicycle, car, motorcycle, bus, truck
        self.allowed_classes: dict = allowed_classes or {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }

        # List class ID untuk filter YOLO
        self._class_ids: list[int] = list(self.allowed_classes.keys())

        # Instance model (lazy loading)
        self.model = None

        # ============================================================
        # LOAD MODEL SAAT INISIALISASI
        # ============================================================
        self._load_model()

    def _load_model(self) -> None:
        """
        Load model YOLOv8 dari file .pt.

        Jika file tidak ditemukan, Ultralytics akan otomatis
        mendownload model dari internet (butuh koneksi pertama kali).
        """
        try:
            # Import di sini untuk lazy loading
            from ultralytics import YOLO

            logger.info(f"Memuat model YOLO dari: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            logger.info("Model YOLOv8 berhasil dimuat.")

        except Exception as e:
            logger.error(f"Gagal memuat model YOLO: {e}")
            raise RuntimeError(f"Tidak dapat memuat model: {e}") from e

    def detect(
        self,
        frame: np.ndarray,
        use_tracking: bool = True,
        tracker: str = "bytetrack.yaml",
    ) -> list[DetectionResult]:
        """
        Deteksi objek pada satu frame video.

        Proses:
        1. Jalankan inferensi YOLOv8 pada frame
        2. Jika tracking aktif, gunakan ByteTrack untuk ID konsisten
        3. Filter hanya kelas yang diizinkan
        4. Konversi ke list DetectionResult

        Args:
            frame           : Frame BGR dari OpenCV (np.ndarray)
            use_tracking    : Aktifkan ByteTrack (default: True)
            tracker         : Config tracker Ultralytics

        Returns:
            List DetectionResult untuk semua objek terdeteksi
        """
        # ============================================================
        # VALIDASI INPUT
        # ============================================================
        if frame is None or frame.size == 0:
            logger.warning("Frame kosong diterima, skip deteksi.")
            return []

        if self.model is None:
            logger.error("Model belum dimuat.")
            return []

        detections: list[DetectionResult] = []

        try:
            # ============================================================
            # JALANKAN INFERENSI YOLO
            # ============================================================
            if use_tracking:
                # Mode tracking: ByteTrack memberikan ID konsisten
                results = self.model.track(
                    source=frame,
                    conf=self.confidence,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    classes=self._class_ids,
                    persist=True,           # Pertahankan track antar frame
                    tracker=tracker,
                    verbose=False,          # Matikan output terminal
                )
            else:
                # Mode deteksi biasa (tanpa tracking ID)
                results = self.model.predict(
                    source=frame,
                    conf=self.confidence,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    classes=self._class_ids,
                    verbose=False,
                )

            # ============================================================
            # PARSE HASIL DETEKSI
            # ============================================================
            for result in results:
                # Ambil boxes dari hasil
                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    # ----------------------------------------
                    # Ekstrak koordinat bounding box
                    # Format: [x1, y1, x2, y2]
                    # ----------------------------------------
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1 = int(xyxy[0])
                    y1 = int(xyxy[1])
                    x2 = int(xyxy[2])
                    y2 = int(xyxy[3])

                    # ----------------------------------------
                    # Ekstrak confidence score
                    # ----------------------------------------
                    conf = float(box.conf[0].cpu().numpy())

                    # ----------------------------------------
                    # Ekstrak class ID dan nama kelas
                    # ----------------------------------------
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.allowed_classes.get(cls_id, "unknown")

                    # ----------------------------------------
                    # Ekstrak tracking ID (jika ada)
                    # ----------------------------------------
                    track_id = None
                    if use_tracking and box.id is not None:
                        track_id = int(box.id[0].cpu().numpy())

                    # ----------------------------------------
                    # Buat objek DetectionResult
                    # ----------------------------------------
                    detection = DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                        track_id=track_id,
                    )
                    detections.append(detection)

        except Exception as e:
            logger.error(f"Error saat deteksi frame: {e}")

        return detections

    def reset_tracker(self) -> None:
        """
        Reset state tracker ByteTrack.
        Dipanggil saat ganti sumber video agar ID mulai dari awal.
        """
        try:
            # Reload model untuk reset internal tracker state
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            logger.info("Tracker direset.")
        except Exception as e:
            logger.error(f"Gagal reset tracker: {e}")

    def get_model_info(self) -> dict:
        """
        Mendapatkan informasi model yang sedang digunakan.

        Returns:
            Dict berisi nama model dan kelas yang dideteksi
        """
        return {
            "model_path": str(self.model_path),
            "confidence": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "allowed_classes": self.allowed_classes,
        }
