"""
================================================================
MODUL INTRUSION MONITOR v2.0 - DETEKSI INTRUSI DENGAN POSE
================================================================
Update v2.0:
  + Person-Only Alert (kendaraan tidak trigger alert)
  + Integrasi PoseResult sebagai input tambahan
  + Dual-mode centroid: pose centroid ATAU bbox centroid
  + detection_mode tracking ('bbox' atau 'pose')

Logika Intrusion v2.0:
  - Intrusion HANYA untuk class 'person' (class_id=0)
  - Kendaraan (car, bus, dll) dimonitor tapi TIDAK alert
  - Centroid diambil dari pose skeleton (lebih akurat)
  - Fallback ke centroid bbox jika pose tidak tersedia
================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Union

from detection.detector import DetectionResult
from monitoring.timer_manager import TimerManager
from roi.polygon_utils import is_point_in_polygon

logger = logging.getLogger(__name__)


class IntrusionStatus(Enum):
    """Status posisi objek terhadap ROI."""
    OUTSIDE = "OUTSIDE"
    INSIDE = "INSIDE"
    ALERT = "INTRUSION ALERT"


@dataclass
class IntrusionEvent:
    """
    Hasil monitoring satu objek per frame.

    Attributes:
        object_id       : ID tracking unik
        class_name      : Nama kelas ('person', 'car', dll)
        class_id        : ID kelas COCO
        bbox            : Bounding box (x1,y1,x2,y2) - bisa None
        centroid        : Titik pusat yang digunakan untuk ROI check
        confidence      : Confidence score
        status          : OUTSIDE / INSIDE / ALERT
        duration        : Durasi dalam ROI (detik)
        timestamp       : Waktu kejadian
        entry_time      : Waktu pertama masuk ROI
        detection_mode  : 'pose' atau 'bbox' (sumber centroid)
        is_partial      : Manusia parsial (dari pose tanpa bbox)
        num_keypoints   : Jumlah keypoint terlihat (0 jika tidak ada pose)
    """
    object_id: int
    class_name: str
    class_id: int
    bbox: Optional[tuple[int, int, int, int]]
    centroid: tuple[int, int]
    confidence: float
    status: IntrusionStatus
    duration: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    entry_time: Optional[datetime] = None
    detection_mode: str = "bbox"
    is_partial: bool = False
    num_keypoints: int = 0

    @property
    def is_alert(self) -> bool:
        return self.status == IntrusionStatus.ALERT

    @property
    def is_inside(self) -> bool:
        return self.status in (IntrusionStatus.INSIDE, IntrusionStatus.ALERT)

    @property
    def can_alert(self) -> bool:
        """Hanya 'person' yang bisa trigger alert."""
        return self.class_id == 0  # person

    @property
    def duration_str(self) -> str:
        if self.duration is None:
            return "0.0 sec"
        return f"{self.duration:.1f} sec"


class IntrusionMonitor:
    """
    Monitor intrusi v2.0 dengan dukungan pose detection.

    Perubahan dari v1.0:
    - Menerima DetectionResult ATAU PoseResult sebagai input
    - Alert HANYA untuk class 'person'
    - Centroid diambil dari pose skeleton jika tersedia
    - Field detection_mode mencatat sumber centroid
    """

    def __init__(
        self,
        roi_points: list[list[int]],
        threshold_seconds: float = 5.0,
        log_cooldown_seconds: float = 10.0,
        alert_class_ids: Optional[set] = None,
        on_alert_callback: Optional[callable] = None,
    ):
        """
        Args:
            roi_points          : Titik polygon ROI
            threshold_seconds   : Detik sebelum alert
            log_cooldown_seconds: Jeda minimum log per objek
            alert_class_ids     : Set class_id yang bisa alert
                                  Default: {0} = hanya person
            on_alert_callback   : Callback saat alert terjadi
        """
        self.roi_points = roi_points
        self.threshold_seconds = threshold_seconds
        self.log_cooldown_seconds = log_cooldown_seconds

        # Hanya person (class_id=0) yang bisa trigger alert
        self.alert_class_ids: set = alert_class_ids or {0}
        self.on_alert_callback = on_alert_callback

        self.timer = TimerManager()
        self._last_logged: dict[int, datetime] = {}
        self._total_alerts: int = 0

        logger.info(
            f"IntrusionMonitor v2.0 | ROI: {len(roi_points)} pts | "
            f"Threshold: {threshold_seconds}s | "
            f"Alert classes: {self.alert_class_ids}"
        )

    def process(
        self,
        detections: list[DetectionResult],
        pose_results: Optional[list] = None,
    ) -> list[IntrusionEvent]:
        """
        Proses deteksi + pose per frame, return list IntrusionEvent.

        Algoritma penggabungan detection + pose:
        1. Buat mapping track_id → DetectionResult (dari YOLO biasa)
        2. Buat mapping track_id → PoseResult (dari YOLOv8-Pose)
        3. Untuk setiap objek:
           a. Jika ada pose → gunakan pose centroid + bbox dari pose
           b. Jika hanya detection biasa → gunakan bbox centroid
        4. Tambahkan pose-only objects (parsial tanpa bbox)

        Args:
            detections      : List DetectionResult dari ObjectDetector
            pose_results    : List PoseResult dari PoseDetector (opsional)

        Returns:
            List IntrusionEvent
        """
        events: list[IntrusionEvent] = []
        active_ids: set[int] = set()

        # ============================================================
        # BUAT MAPPING TRACK_ID → DATA
        # ============================================================
        # Dari object detection biasa
        det_by_id: dict[int, DetectionResult] = {}
        for det in detections:
            if det.track_id is not None:
                det_by_id[det.track_id] = det
                active_ids.add(det.track_id)

        # Dari pose detection
        pose_by_id: dict[int, object] = {}
        if pose_results:
            for pose in pose_results:
                if pose.track_id is not None:
                    pose_by_id[pose.track_id] = pose
                    active_ids.add(pose.track_id)

        # ============================================================
        # PROSES SEMUA OBJEK AKTIF
        # ============================================================
        for tid in active_ids:
            det = det_by_id.get(tid)
            pose = pose_by_id.get(tid)

            # ----------------------------------------
            # TENTUKAN CLASS INFO
            # ----------------------------------------
            if det is not None:
                class_name = det.class_name
                class_id = det.class_id
                confidence = det.confidence
                bbox = det.bbox
            elif pose is not None:
                # Pose-only (parsial tanpa YOLO detection)
                class_name = "person"
                class_id = 0
                confidence = pose.confidence
                bbox = pose.bbox
            else:
                continue

            # ----------------------------------------
            # PILIH CENTROID: POSE > BBOX
            # ----------------------------------------
            centroid = None
            detection_mode = "bbox"
            is_partial = False
            num_kp = 0

            if pose is not None and pose.effective_centroid is not None:
                centroid = pose.effective_centroid
                detection_mode = "pose"
                is_partial = pose.is_partial
                num_kp = pose.num_visible_kp
            elif det is not None:
                centroid = det.centroid
                detection_mode = "bbox"

            if centroid is None:
                continue

            # ----------------------------------------
            # CEK ROI
            # ----------------------------------------
            roi_active = len(self.roi_points) >= 3
            in_roi = False
            if roi_active:
                in_roi = is_point_in_polygon(centroid, self.roi_points)

            # ----------------------------------------
            # UPDATE TIMER
            # ----------------------------------------
            duration = None
            if in_roi:
                self.timer.start(tid)
                duration = self.timer.get_duration(tid)
            else:
                if self.timer.is_active(tid):
                    self.timer.stop(tid)

            # ----------------------------------------
            # TENTUKAN STATUS
            # Alert HANYA untuk person
            # ----------------------------------------
            can_alert = (class_id in self.alert_class_ids)
            status = self._determine_status(
                in_roi=in_roi,
                duration=duration,
                roi_active=roi_active,
                can_alert=can_alert,
            )

            # ----------------------------------------
            # BUAT EVENT
            # ----------------------------------------
            event = IntrusionEvent(
                object_id=tid,
                class_name=class_name,
                class_id=class_id,
                bbox=bbox,
                centroid=centroid,
                confidence=confidence,
                status=status,
                duration=duration,
                timestamp=datetime.now(),
                entry_time=self.timer.get_entry_time(tid),
                detection_mode=detection_mode,
                is_partial=is_partial,
                num_keypoints=num_kp,
            )

            if event.is_alert:
                self._handle_alert(event)

            events.append(event)

        # Bersihkan timer objek yang tidak terdeteksi
        self.timer.clean_inactive(active_ids)

        return events

    def _determine_status(
        self,
        in_roi: bool,
        duration: Optional[float],
        roi_active: bool,
        can_alert: bool = True,
    ) -> IntrusionStatus:
        """
        Tentukan status intrusi.

        can_alert=False → maksimum status INSIDE (tidak bisa ALERT).
        Ini digunakan untuk kendaraan yang tidak perlu alert.
        """
        if not roi_active or not in_roi:
            return IntrusionStatus.OUTSIDE

        if duration is None:
            return IntrusionStatus.INSIDE

        if can_alert and duration >= self.threshold_seconds:
            return IntrusionStatus.ALERT
        else:
            return IntrusionStatus.INSIDE

    def _handle_alert(self, event: IntrusionEvent) -> None:
        """Handle alert: log dan trigger callback."""
        now = datetime.now()
        last = self._last_logged.get(event.object_id)
        should_log = (
            last is None
            or (now - last).total_seconds() >= self.log_cooldown_seconds
        )
        if should_log:
            self._total_alerts += 1
            self._last_logged[event.object_id] = now
            logger.warning(
                f"ALERT! ID={event.object_id} {event.class_name} "
                f"duration={event.duration:.1f}s mode={event.detection_mode}"
            )
            if self.on_alert_callback:
                try:
                    self.on_alert_callback(event)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

    def should_log_event(self, object_id: int) -> bool:
        now = datetime.now()
        last = self._last_logged.get(object_id)
        return last is None or (now - last).total_seconds() >= self.log_cooldown_seconds

    def mark_logged(self, object_id: int) -> None:
        self._last_logged[object_id] = datetime.now()

    def update_roi(self, roi_points: list[list[int]]) -> None:
        self.roi_points = roi_points
        self.timer.reset()
        self._last_logged.clear()

    def reset(self) -> None:
        self.timer.reset()
        self._last_logged.clear()
        self._total_alerts = 0

    def get_statistics(self) -> dict:
        t = self.timer.get_statistics()
        return {
            "total_alerts": self._total_alerts,
            "active_in_roi": t["active_objects_in_roi"],
            "max_current_duration": t["max_current_duration"],
            "roi_active": len(self.roi_points) >= 3,
            "threshold_seconds": self.threshold_seconds,
        }
