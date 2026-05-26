"""
================================================================
MODUL VIDEO RENDERER v2.0 - RENDERING DENGAN POSE SKELETON
================================================================
Update v2.0:
  + Render skeleton pose di atas bounding box
  + Indikator 'POSE' / 'BBOX' mode di label
  + Indikator [PARSIAL] untuk manusia tanpa bbox penuh
  + Warna bbox dibedakan untuk kendaraan vs person
  + Banner alert hanya muncul untuk person
================================================================
"""

import logging
from typing import Optional

import cv2
import numpy as np

from config.settings import Settings
from monitoring.intrusion_monitor import IntrusionEvent, IntrusionStatus

logger = logging.getLogger(__name__)


class VideoRenderer:
    """
    Renderer visual v2.0 dengan dukungan pose skeleton overlay.
    """

    def __init__(self, cfg: Optional[Settings] = None):
        self.cfg: Settings = cfg or Settings()
        self.font = self.cfg.FONT_FACE
        self.font_scale = self.cfg.FONT_SCALE
        self.thickness = self.cfg.FONT_THICKNESS

    def render(
        self,
        frame: np.ndarray,
        events: list[IntrusionEvent],
        roi_points: list[list[int]],
        pose_results: Optional[list] = None,
        show_roi: bool = True,
        show_stats: bool = True,
        show_skeleton: bool = True,
    ) -> np.ndarray:
        """
        Render semua elemen visual pada frame.

        Urutan rendering:
        1. ROI polygon overlay
        2. Skeleton pose (di bawah bbox agar bbox di atas)
        3. Bounding boxes + labels
        4. Alert banner
        5. Panel statistik

        Args:
            frame        : Frame BGR asli
            events       : List IntrusionEvent dari monitor
            roi_points   : Titik polygon ROI
            pose_results : List PoseResult (opsional, untuk skeleton)
            show_roi     : Tampilkan ROI overlay
            show_stats   : Tampilkan panel statistik
            show_skeleton: Tampilkan skeleton pose

        Returns:
            Frame yang sudah dirender
        """
        output = frame.copy()

        # 1. ROI overlay
        if show_roi and len(roi_points) >= 3:
            output = self._draw_roi(output, roi_points)

        # 2. Skeleton pose (sebelum bbox agar bbox di atas)
        if show_skeleton and pose_results:
            output = self._draw_all_skeletons(output, events, pose_results)

        # 3. Bounding boxes + labels
        has_alert = False
        alert_events = []
        for event in events:
            output = self._draw_object(output, event)
            if event.is_alert:
                has_alert = True
                alert_events.append(event)

        # 4. Alert banner
        if has_alert:
            output = self._draw_alert_banner(output, alert_events)

        # 5. Panel statistik
        if show_stats:
            output = self._draw_stats_panel(output, events, roi_points, pose_results)

        return output

    def _draw_roi(self, frame: np.ndarray, roi_points: list) -> np.ndarray:
        """Gambar ROI polygon semi-transparan."""
        pts = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color=self.cfg.COLOR_ROI_FILL)
        frame = cv2.addWeighted(overlay, self.cfg.ROI_ALPHA, frame, 1 - self.cfg.ROI_ALPHA, 0)
        cv2.polylines(frame, [pts], isClosed=True, color=self.cfg.COLOR_ROI_BORDER, thickness=2)

        top_pt = min(roi_points, key=lambda p: p[1])
        self._put_text_with_bg(
            frame, "[ RESTRICTED AREA ]",
            (max(top_pt[0], 10), max(top_pt[1] - 10, 20)),
            (0, 0, 0), self.cfg.COLOR_ROI_BORDER, 0.55, 1
        )
        return frame

    def _draw_all_skeletons(
        self,
        frame: np.ndarray,
        events: list[IntrusionEvent],
        pose_results: list,
    ) -> np.ndarray:
        """
        Gambar skeleton untuk semua pose yang terdeteksi.

        Warna skeleton disesuaikan dengan status event.
        """
        # Buat mapping track_id → event untuk warna
        event_by_id = {e.object_id: e for e in events}

        for pose in pose_results:
            if pose.keypoints is None:
                continue

            # Tentukan warna berdasarkan status event
            event = event_by_id.get(pose.track_id)
            if event:
                if event.status == IntrusionStatus.ALERT:
                    color = self.cfg.COLOR_ALERT
                elif event.status == IntrusionStatus.INSIDE:
                    color = self.cfg.COLOR_INSIDE_ROI
                else:
                    color = self.cfg.COLOR_NORMAL
            else:
                color = (200, 200, 200)  # Abu jika tidak ada event

            # Gambar skeleton menggunakan modul skeleton_renderer
            try:
                from pose.skeleton_renderer import draw_skeleton
                from pose.pose_detector import PoseResult
                frame = draw_skeleton(
                    frame=frame,
                    pose=pose,
                    draw_keypoints=True,
                    draw_connections=True,
                    draw_centroid=True,
                    draw_label=False,  # Label digambar oleh _draw_object
                    status_color=color,
                )
            except Exception as e:
                logger.debug(f"Skeleton render error: {e}")

        return frame

    def _draw_object(self, frame: np.ndarray, event: IntrusionEvent) -> np.ndarray:
        """
        Gambar bounding box dan label untuk satu objek.

        Perbedaan v2.0:
        - Indikator [POSE] atau [BBOX] di label
        - Indikator [PARSIAL] untuk manusia tanpa bbox
        - Kendaraan digambar dengan garis tipis (bukan tebal)
        """
        # Pilih warna berdasarkan status
        if event.status == IntrusionStatus.ALERT:
            color = self.cfg.COLOR_ALERT
        elif event.status == IntrusionStatus.INSIDE:
            color = self.cfg.COLOR_INSIDE_ROI
        else:
            color = self.cfg.COLOR_NORMAL

        # Kendaraan: warna lebih redup (bukan ancaman)
        box_thickness = 2
        if event.class_id != 0:  # Bukan person
            color = tuple(int(c * 0.6) for c in color)
            box_thickness = 1

        # Gambar bbox hanya jika ada
        if event.bbox is not None:
            x1, y1, x2, y2 = event.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=box_thickness)

        # Gambar centroid
        cx, cy = event.centroid
        cv2.circle(frame, (cx, cy), radius=4, color=color, thickness=-1)

        # ============================================================
        # SUSUN LABEL TEKS
        # ============================================================
        # Baris 1: Kelas + ID + Mode deteksi
        mode_badge = f"[{event.detection_mode.upper()}]"
        if event.is_partial:
            mode_badge += "[PARSIAL]"

        label1 = f"{event.class_name.upper()} ID:{event.object_id} {mode_badge}"

        # Baris 2: Confidence
        label2 = f"Conf: {event.confidence:.0%}"

        # Baris 3: Durasi (hanya jika dalam ROI)
        label3 = None
        if event.is_inside and event.duration is not None:
            label3 = f"Duration: {event.duration_str}"

        # Baris 4: Alert
        label4 = None
        if event.is_alert:
            label4 = "!! INTRUSION ALERT !!"

        # Posisi label
        if event.bbox is not None:
            lx, ly = event.bbox[0], event.bbox[1]
        else:
            lx, ly = max(cx - 60, 5), cy

        line_h = 18
        num_lines = 2 + (1 if label3 else 0) + (1 if label4 else 0)
        ly_start = max(ly - num_lines * line_h, 10)

        lines = [l for l in [label1, label2, label3, label4] if l]
        for i, line in enumerate(lines):
            bg = (0, 0, 180) if "ALERT" in line else color
            self._put_text_with_bg(
                frame, line, (lx, ly_start + i * line_h),
                (255, 255, 255), bg, 0.45, 1
            )

        return frame

    def _draw_alert_banner(
        self, frame: np.ndarray, alert_events: list[IntrusionEvent]
    ) -> np.ndarray:
        """Banner alert merah di bagian bawah frame."""
        h, w = frame.shape[:2]
        banner_h = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 180), -1)
        frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

        e = alert_events[0]
        text = (
            f"⚠ INTRUSION ALERT  |  "
            f"PERSON ID:{e.object_id}  |  "
            f"Duration: {e.duration_str}  |  "
            f"Mode: {e.detection_mode.upper()}  |  "
            f"Total: {len(alert_events)} person(s)"
        )
        cv2.putText(frame, text, (10, h - 20), self.font, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Border merah sekeliling frame
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        return frame

    def _draw_stats_panel(
        self,
        frame: np.ndarray,
        events: list[IntrusionEvent],
        roi_points: list,
        pose_results: Optional[list] = None,
    ) -> np.ndarray:
        """Panel statistik di pojok kanan atas."""
        h, w = frame.shape[:2]

        total_obj = len(events)
        persons = sum(1 for e in events if e.class_id == 0)
        in_roi = sum(1 for e in events if e.is_inside)
        alerts = sum(1 for e in events if e.is_alert)
        pose_count = len(pose_results) if pose_results else 0
        roi_status = "AKTIF" if len(roi_points) >= 3 else "TIDAK ADA"

        lines = [
            f"Detected : {total_obj}",
            f"Persons  : {persons}",
            f"Poses    : {pose_count}",
            f"In ROI   : {in_roi}",
            f"ALERT    : {alerts}",
            f"ROI      : {roi_status}",
        ]

        panel_w = 185
        panel_h = len(lines) * 20 + 14
        px, py = w - panel_w - 8, 8

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (15, 15, 15), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        for i, line in enumerate(lines):
            tc = (100, 100, 255) if "ALERT" in line and alerts > 0 else (255, 255, 255)
            cv2.putText(frame, line, (px + 7, py + 16 + i * 20),
                        self.font, 0.42, tc, 1, cv2.LINE_AA)

        return frame

    def _put_text_with_bg(
        self, frame, text, position, text_color, bg_color,
        font_scale=0.5, thickness=1, padding=3
    ):
        x, y = position
        (tw, th), bl = cv2.getTextSize(text, self.font, font_scale, thickness)
        cv2.rectangle(frame,
                      (x - padding, y - th - padding),
                      (x + tw + padding, y + bl + padding),
                      bg_color, -1)
        cv2.putText(frame, text, (x, y), self.font, font_scale,
                    text_color, thickness, cv2.LINE_AA)

    def draw_roi_preview(
        self, frame: np.ndarray, roi_points: list, in_progress: bool = False
    ) -> np.ndarray:
        """Preview ROI saat penentuan titik polygon."""
        output = frame.copy()
        if not roi_points:
            return output
        pts = np.array(roi_points, dtype=np.int32)
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(output, tuple(pts[i]), tuple(pts[i + 1]), (0, 255, 255), 2)
            if not in_progress and len(pts) >= 3:
                cv2.line(output, tuple(pts[-1]), tuple(pts[0]), (0, 255, 255), 2)
        for i, pt in enumerate(pts):
            cv2.circle(output, tuple(pt), 6, (0, 255, 0), -1)
            cv2.putText(output, f"P{i+1}", (pt[0] + 8, pt[1] - 8),
                        self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return output

    def resize_frame(self, frame: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
        if width is None and height is None:
            width, height = self.cfg.DISPLAY_WIDTH, self.cfg.DISPLAY_HEIGHT
        if width and height:
            return cv2.resize(frame, (width, height))
        h, w = frame.shape[:2]
        if width:
            return cv2.resize(frame, (width, int(h * width / w)))
        if height:
            return cv2.resize(frame, (int(w * height / h), height))
        return frame
