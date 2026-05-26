"""
================================================================
MODUL POSE DETECTOR - DETEKSI POSE MANUSIA DENGAN YOLOV8-POSE
================================================================
Modul ini mengimplementasikan deteksi pose manusia menggunakan
model YOLOv8-Pose yang telah dilatih pada dataset COCO.

Mengapa Pose Detection?
------------------------
Object detection (YOLO biasa) gagal mendeteksi manusia dalam
kondisi:
  ✗ Setengah badan (occlusion bawah)
  ✗ Hanya kepala/bahu terlihat
  ✗ Tubuh tertutup objek lain (partial occlusion)
  ✗ Manusia jauh dari kamera (resolusi rendah)

Pose detection memecahkan masalah ini dengan mendeteksi
keypoint (titik sendi) secara individual, sehingga TETAP
dapat mendeteksi manusia meskipun sebagian tubuh tidak terlihat.

Model yang didukung:
  - yolov8n-pose.pt  (nano,  tercepat, akurasi terendah)
  - yolov8s-pose.pt  (small, keseimbangan speed & akurasi)
  - yolov8m-pose.pt  (medium, lebih akurat, lebih lambat)

Kelas utama:
    PoseResult      - Data class hasil pose satu manusia
    PoseDetector    - Wrapper model YOLOv8-Pose

Contoh penggunaan:
    detector = PoseDetector("models/yolov8n-pose.pt")
    pose_results = detector.detect(frame)
    for pose in pose_results:
        centroid = pose.pose_centroid
        keypoints = pose.visible_keypoints
================================================================
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from pose.keypoint_utils import (
    KEYPOINT_CONFIDENCE_THRESHOLD,
    calculate_pose_centroid,
    count_visible_keypoints,
    filter_visible_keypoints,
    is_human_partially_visible,
)

# ============================================================
# SETUP LOGGING
# ============================================================
logger = logging.getLogger(__name__)


@dataclass
class PoseResult:
    """
    Data class menyimpan hasil deteksi pose satu manusia.

    Attributes:
        bbox            : Bounding box (x1,y1,x2,y2) — bisa None
                          jika manusia parsial tanpa bbox penuh
        confidence      : Confidence score bounding box (0-1)
        track_id        : ID tracking (dari ByteTrack)
        keypoints       : Array (17, 2) koordinat keypoint [x, y]
        kp_confidences  : Array (17,) confidence tiap keypoint
        visible_keypoints: Dict {idx: (x, y, conf)} kp terlihat
        pose_centroid   : Centroid dari pose skeleton (cx, cy)
        is_partial      : True jika manusia parsial (tanpa bbox)
        num_visible_kp  : Jumlah keypoint yang terlihat
    """

    bbox: Optional[tuple[int, int, int, int]]
    confidence: float
    track_id: Optional[int]
    keypoints: np.ndarray       # Shape: (17, 2)
    kp_confidences: np.ndarray  # Shape: (17,)
    visible_keypoints: dict = field(default_factory=dict)
    pose_centroid: Optional[tuple[int, int]] = None
    is_partial: bool = False
    num_visible_kp: int = 0

    def __post_init__(self):
        """Hitung atribut turunan setelah inisialisasi."""
        # Hitung keypoint yang terlihat
        self.visible_keypoints = filter_visible_keypoints(
            self.keypoints, self.kp_confidences
        )
        self.num_visible_kp = len(self.visible_keypoints)

        # Hitung centroid dari pose keypoint
        self.pose_centroid = calculate_pose_centroid(
            self.keypoints, self.kp_confidences
        )

    @property
    def effective_centroid(self) -> Optional[tuple[int, int]]:
        """
        Centroid efektif untuk ROI check.

        Prioritas:
        1. Pose centroid (dari keypoint skeleton) - lebih akurat
        2. Bbox centroid (fallback jika pose tidak ada)
        3. None jika tidak ada data
        """
        # Prioritas 1: centroid dari pose skeleton
        if self.pose_centroid is not None:
            return self.pose_centroid

        # Prioritas 2: centroid dari bounding box
        if self.bbox is not None:
            x1, y1, x2, y2 = self.bbox
            return ((x1 + x2) // 2, (y1 + y2) // 2)

        return None

    @property
    def has_valid_pose(self) -> bool:
        """True jika pose memiliki keypoint yang cukup valid."""
        return self.num_visible_kp >= 3

    def __repr__(self) -> str:
        return (
            f"PoseResult(id={self.track_id}, "
            f"conf={self.confidence:.2f}, "
            f"kp={self.num_visible_kp}/17, "
            f"partial={self.is_partial})"
        )


class PoseDetector:
    """
    Detektor pose manusia berbasis YOLOv8-Pose + ByteTrack.

    Mampu mendeteksi:
    - Manusia dengan badan penuh (body detection + keypoints)
    - Manusia parsial (hanya keypoints yang terlihat)
    - Tangan, bahu, atau kepala saja yang muncul

    Algoritma:
    1. Jalankan YOLOv8-Pose pada frame
    2. Untuk setiap manusia terdeteksi:
       a. Ambil bounding box (jika ada)
       b. Ambil 17 keypoint COCO + confidence
       c. Hitung centroid dari pose skeleton
    3. Filter berdasarkan jumlah keypoint minimal

    Attributes:
        model_path      : Path ke file model .pt
        confidence      : Minimum bbox confidence
        pose_confidence : Minimum keypoint confidence
        device          : 'cuda' atau 'cpu'
        use_half        : Gunakan FP16 (hanya GPU)
        model           : Instance YOLO
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.35,
        pose_confidence: float = 0.30,
        imgsz: int = 640,
        device: Optional[str] = None,
        use_half: bool = False,
    ):
        """
        Inisialisasi PoseDetector.

        Args:
            model_path      : Path ke yolov8n-pose.pt atau yolov8s-pose.pt
            confidence      : Min confidence deteksi manusia
            pose_confidence : Min confidence keypoint
            imgsz           : Ukuran input inferensi
            device          : 'cuda', 'cpu', atau None (auto-detect)
            use_half        : FP16 inference (GPU only)
        """
        self.model_path: Path = Path(model_path)
        self.confidence: float = confidence
        self.pose_confidence: float = pose_confidence
        self.imgsz: int = imgsz
        self.use_half: bool = use_half

        # ============================================================
        # AUTO-DETECT DEVICE (GPU/CPU)
        # ============================================================
        self.device: str = self._detect_device(device)
        logger.info(f"PoseDetector menggunakan device: {self.device}")

        # Instance model
        self.model = None
        self._load_model()

    def _detect_device(self, requested_device: Optional[str]) -> str:
        """
        Auto-detect device terbaik yang tersedia.

        Args:
            requested_device: Device yang diminta ('cuda'/'cpu'/None)

        Returns:
            'cuda' jika GPU NVIDIA tersedia, 'cpu' jika tidak
        """
        if requested_device is not None:
            return requested_device

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU terdeteksi: {gpu_name}")
                return "cuda"
        except ImportError:
            pass

        logger.info("GPU tidak tersedia, menggunakan CPU.")
        return "cpu"

    def _load_model(self) -> None:
        """
        Load model YOLOv8-Pose dari file.

        Model akan otomatis didownload jika tidak ditemukan
        (membutuhkan koneksi internet pertama kali).
        """
        try:
            from ultralytics import YOLO

            logger.info(f"Memuat Pose Model: {self.model_path}")
            self.model = YOLO(str(self.model_path))

            # ----------------------------------------
            # FP16 HALF PRECISION (GPU ONLY)
            # Mengurangi penggunaan VRAM ~50%
            # Meningkatkan kecepatan inferensi ~30-50%
            # ----------------------------------------
            if self.use_half and self.device == "cuda":
                self.model.model.half()
                logger.info("FP16 half precision diaktifkan.")

            logger.info("Pose Model berhasil dimuat.")

        except Exception as e:
            logger.error(f"Gagal memuat Pose Model: {e}")
            raise RuntimeError(f"Tidak dapat memuat pose model: {e}") from e

    def detect(
        self,
        frame: np.ndarray,
        use_tracking: bool = True,
        tracker: str = "bytetrack.yaml",
        min_visible_kp: int = 3,
    ) -> list[PoseResult]:
        """
        Deteksi pose manusia pada satu frame video.

        Keunggulan dibanding object detection biasa:
        - Mendeteksi manusia parsial (setengah badan)
        - Memberikan posisi keypoint tubuh secara individu
        - Centroid dari pose lebih akurat untuk ROI check

        Args:
            frame           : Frame BGR dari OpenCV
            use_tracking    : Aktifkan ByteTrack (ID konsisten)
            tracker         : Config tracker Ultralytics
            min_visible_kp  : Minimum keypoint terlihat agar valid

        Returns:
            List PoseResult untuk semua manusia terdeteksi
        """
        # ============================================================
        # VALIDASI INPUT
        # ============================================================
        if frame is None or frame.size == 0:
            return []

        if self.model is None:
            logger.error("Pose model belum dimuat.")
            return []

        pose_results: list[PoseResult] = []

        try:
            # ============================================================
            # INFERENSI MODEL YOLOv8-POSE
            # ============================================================
            if use_tracking:
                results = self.model.track(
                    source=frame,
                    conf=self.confidence,
                    imgsz=self.imgsz,
                    persist=True,
                    tracker=tracker,
                    verbose=False,
                    device=self.device,
                )
            else:
                results = self.model.predict(
                    source=frame,
                    conf=self.confidence,
                    imgsz=self.imgsz,
                    verbose=False,
                    device=self.device,
                )

            # ============================================================
            # PARSE SETIAP HASIL DETEKSI
            # ============================================================
            for result in results:
                if result.keypoints is None:
                    continue

                # Data keypoint dari model
                kp_data = result.keypoints
                boxes = result.boxes

                num_detections = len(kp_data.xy) if kp_data.xy is not None else 0

                for i in range(num_detections):
                    # ----------------------------------------
                    # EKSTRAK KEYPOINT KOORDINAT DAN CONFIDENCE
                    # ----------------------------------------
                    try:
                        kp_xy = kp_data.xy[i].cpu().numpy()       # Shape: (17, 2)
                        kp_conf = kp_data.conf[i].cpu().numpy()    # Shape: (17,)
                    except Exception as e:
                        logger.debug(f"Skip keypoint {i}: {e}")
                        continue

                    # ----------------------------------------
                    # CEK MINIMUM KEYPOINT TERLIHAT
                    # ----------------------------------------
                    n_visible = count_visible_keypoints(
                        kp_xy, kp_conf, self.pose_confidence
                    )

                    if n_visible < min_visible_kp:
                        # Terlalu sedikit keypoint, skip
                        continue

                    # ----------------------------------------
                    # EKSTRAK BOUNDING BOX (JIKA ADA)
                    # ----------------------------------------
                    bbox = None
                    bbox_conf = 0.0
                    track_id = None

                    if boxes is not None and i < len(boxes):
                        box = boxes[i]
                        xyxy = box.xyxy[0].cpu().numpy()
                        bbox = (
                            int(xyxy[0]), int(xyxy[1]),
                            int(xyxy[2]), int(xyxy[3]),
                        )
                        bbox_conf = float(box.conf[0].cpu().numpy())

                        # Tracking ID
                        if use_tracking and box.id is not None:
                            track_id = int(box.id[0].cpu().numpy())

                    # ----------------------------------------
                    # TENTUKAN APAKAH PARSIAL
                    # ----------------------------------------
                    is_partial = (bbox is None)

                    # ----------------------------------------
                    # BUAT PoseResult
                    # ----------------------------------------
                    pose_result = PoseResult(
                        bbox=bbox,
                        confidence=bbox_conf,
                        track_id=track_id,
                        keypoints=kp_xy,
                        kp_confidences=kp_conf,
                        is_partial=is_partial,
                    )

                    pose_results.append(pose_result)

        except Exception as e:
            logger.error(f"Error pose detection: {e}")

        return pose_results

    def reset_tracker(self) -> None:
        """Reset ByteTrack state (saat ganti sumber video)."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            if self.use_half and self.device == "cuda":
                self.model.model.half()
            logger.info("Pose tracker direset.")
        except Exception as e:
            logger.error(f"Gagal reset pose tracker: {e}")

    def get_model_info(self) -> dict:
        """Info model yang sedang digunakan."""
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "confidence": self.confidence,
            "pose_confidence": self.pose_confidence,
            "imgsz": self.imgsz,
            "use_half": self.use_half,
        }
