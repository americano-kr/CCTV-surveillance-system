"""
================================================================
MODUL SKELETON RENDERER - VISUALISASI POSE SKELETON
================================================================
Modul ini menggambar hasil deteksi pose (keypoints dan
koneksi skeleton) pada frame video menggunakan OpenCV.

Elemen yang digambar:
1. Keypoint Joints   - Lingkaran di setiap titik sendi
2. Skeleton Lines    - Garis antar sendi (tulang)
3. Label Pose        - ID tracking + jumlah keypoint
4. Centroid Pose     - Titik pusat pose (untuk ROI check)

Warna skeleton mengikuti skema anatomis:
    - Kuning : Kepala & wajah
    - Hijau   : Torso (badan tengah)
    - Biru    : Lengan kanan
    - Oranye  : Lengan kiri
    - Ungu    : Kaki kiri
    - Pink    : Kaki kanan

Fungsi utama:
    draw_skeleton()         - Gambar skeleton lengkap
    draw_keypoints_only()   - Hanya gambar titik keypoint
    draw_pose_centroid()    - Gambar centroid pose
================================================================
"""

import logging
from typing import Optional

import cv2
import numpy as np

from pose.keypoint_utils import (
    KEYPOINT_CONFIDENCE_THRESHOLD,
    SKELETON_CONNECTIONS,
    filter_visible_keypoints,
)
from pose.pose_detector import PoseResult

# ============================================================
# SETUP LOGGING
# ============================================================
logger = logging.getLogger(__name__)

# ============================================================
# KONFIGURASI VISUAL
# ============================================================
# Radius lingkaran keypoint (pixel)
KEYPOINT_RADIUS: int = 5

# Ketebalan garis tulang skeleton
SKELETON_THICKNESS: int = 2

# Warna keypoint (putih default)
KEYPOINT_COLOR: tuple = (255, 255, 255)

# Warna centroid pose
POSE_CENTROID_COLOR: tuple = (0, 255, 255)  # Kuning

# Warna label teks (putih)
LABEL_COLOR: tuple = (255, 255, 255)


def draw_skeleton(
    frame: np.ndarray,
    pose: PoseResult,
    conf_threshold: float = KEYPOINT_CONFIDENCE_THRESHOLD,
    draw_keypoints: bool = True,
    draw_connections: bool = True,
    draw_centroid: bool = True,
    draw_label: bool = True,
    status_color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Gambar skeleton pose lengkap pada frame.

    Menggambar semua elemen visual pose:
    - Garis skeleton (koneksi antar sendi)
    - Titik keypoint (lingkaran di setiap sendi)
    - Centroid pose (titik pusat)
    - Label teks (ID + keypoint count)

    Args:
        frame           : Frame BGR OpenCV
        pose            : PoseResult yang akan digambar
        conf_threshold  : Min confidence keypoint untuk ditampilkan
        draw_keypoints  : Gambar titik keypoint (default True)
        draw_connections: Gambar garis tulang (default True)
        draw_centroid   : Gambar centroid (default True)
        draw_label      : Gambar label teks (default True)
        status_color    : Warna berdasarkan status intrusi

    Returns:
        Frame yang sudah digambar skeleton
    """
    if pose.keypoints is None or len(pose.keypoints) == 0:
        return frame

    # Ambil keypoint yang terlihat
    visible = filter_visible_keypoints(
        pose.keypoints, pose.kp_confidences, conf_threshold
    )

    if not visible:
        return frame

    # ============================================================
    # STEP 1: GAMBAR GARIS SKELETON (KONEKSI TULANG)
    # ============================================================
    if draw_connections:
        frame = _draw_skeleton_lines(frame, visible, status_color)

    # ============================================================
    # STEP 2: GAMBAR TITIK KEYPOINT
    # ============================================================
    if draw_keypoints:
        frame = _draw_keypoint_circles(frame, visible, status_color)

    # ============================================================
    # STEP 3: GAMBAR CENTROID POSE
    # ============================================================
    if draw_centroid and pose.pose_centroid is not None:
        frame = draw_pose_centroid(frame, pose.pose_centroid, status_color)

    # ============================================================
    # STEP 4: GAMBAR LABEL TEKS
    # ============================================================
    if draw_label:
        frame = _draw_pose_label(frame, pose, visible, status_color)

    return frame


def _draw_skeleton_lines(
    frame: np.ndarray,
    visible_keypoints: dict[int, tuple[int, int, float]],
    color_override: Optional[tuple] = None,
) -> np.ndarray:
    """
    Gambar garis koneksi antar keypoint (skeleton lines).

    Hanya menggambar garis jika KEDUA keypoint terlihat.

    Args:
        frame               : Frame BGR
        visible_keypoints   : Dict {idx: (x, y, conf)}
        color_override      : Override warna semua garis (None = per-koneksi)

    Returns:
        Frame dengan garis skeleton
    """
    for kp_a, kp_b, default_color in SKELETON_CONNECTIONS:
        # Cek apakah kedua endpoint terlihat
        if kp_a not in visible_keypoints or kp_b not in visible_keypoints:
            continue

        x_a, y_a, _ = visible_keypoints[kp_a]
        x_b, y_b, _ = visible_keypoints[kp_b]

        # Pilih warna: override atau warna per-koneksi
        line_color = color_override if color_override else default_color

        # Buat warna lebih terang (overlay pada video)
        cv2.line(
            frame,
            (x_a, y_a),
            (x_b, y_b),
            line_color,
            thickness=SKELETON_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    return frame


def _draw_keypoint_circles(
    frame: np.ndarray,
    visible_keypoints: dict[int, tuple[int, int, float]],
    highlight_color: tuple = (0, 255, 0),
) -> np.ndarray:
    """
    Gambar lingkaran di setiap keypoint terlihat.

    Keypoint digambar dengan:
    - Lingkaran putih kecil di pusat
    - Ring berwarna di luar (sesuai status)

    Args:
        frame               : Frame BGR
        visible_keypoints   : Dict {idx: (x, y, conf)}
        highlight_color     : Warna ring luar keypoint

    Returns:
        Frame dengan keypoint circles
    """
    for idx, (x, y, conf) in visible_keypoints.items():
        # Ring luar berwarna
        cv2.circle(
            frame,
            (x, y),
            radius=KEYPOINT_RADIUS + 1,
            color=highlight_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        # Titik putih di tengah
        cv2.circle(
            frame,
            (x, y),
            radius=KEYPOINT_RADIUS - 2,
            color=(255, 255, 255),
            thickness=-1,  # Isi penuh
            lineType=cv2.LINE_AA,
        )

    return frame


def draw_pose_centroid(
    frame: np.ndarray,
    centroid: tuple[int, int],
    color: tuple = POSE_CENTROID_COLOR,
    size: int = 8,
) -> np.ndarray:
    """
    Gambar titik centroid pose (titik pusat untuk ROI check).

    Digambar sebagai diamond/rhombus untuk membedakan dari
    centroid bounding box yang biasanya berupa lingkaran.

    Args:
        frame    : Frame BGR
        centroid : Koordinat (cx, cy) centroid pose
        color    : Warna centroid
        size     : Ukuran simbol (pixel)

    Returns:
        Frame dengan centroid digambar
    """
    cx, cy = centroid

    # Gambar diamond (4 titik)
    pts = np.array([
        [cx, cy - size],      # Atas
        [cx + size, cy],      # Kanan
        [cx, cy + size],      # Bawah
        [cx - size, cy],      # Kiri
    ], dtype=np.int32)

    cv2.fillPoly(frame, [pts], color=color)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 0), thickness=1)

    return frame


def _draw_pose_label(
    frame: np.ndarray,
    pose: PoseResult,
    visible_keypoints: dict,
    status_color: tuple,
) -> np.ndarray:
    """
    Gambar label teks untuk pose (ID tracking + info).

    Args:
        frame               : Frame BGR
        pose                : PoseResult yang dilabeli
        visible_keypoints   : Dict keypoint terlihat
        status_color        : Warna sesuai status

    Returns:
        Frame dengan label teks
    """
    # Tentukan posisi label
    label_pos = _get_label_position(pose, visible_keypoints)
    if label_pos is None:
        return frame

    lx, ly = label_pos

    # ============================================================
    # BARIS 1: ID + KELAS
    # ============================================================
    id_text = f"PERSON ID:{pose.track_id}" if pose.track_id else "PERSON"
    if pose.is_partial:
        id_text += " [PARSIAL]"

    _put_text_bg(frame, id_text, (lx, ly), status_color)

    # ============================================================
    # BARIS 2: JUMLAH KEYPOINT TERLIHAT
    # ============================================================
    kp_text = f"KP: {pose.num_visible_kp}/17"
    _put_text_bg(frame, kp_text, (lx, ly + 18), (40, 40, 40))

    return frame


def _get_label_position(
    pose: PoseResult,
    visible_keypoints: dict,
) -> Optional[tuple[int, int]]:
    """
    Hitung posisi terbaik untuk label teks.

    Prioritas posisi:
    1. Di atas bounding box (jika ada)
    2. Di atas keypoint paling atas (nose/eye/shoulder)
    """
    # Dari bounding box
    if pose.bbox is not None:
        x1, y1, _, _ = pose.bbox
        return (x1, max(y1 - 25, 10))

    # Dari keypoint teratas
    if visible_keypoints:
        top_kp = min(visible_keypoints.values(), key=lambda v: v[1])
        return (max(top_kp[0] - 30, 5), max(top_kp[1] - 25, 10))

    return None


def _put_text_bg(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    bg_color: tuple,
    text_color: tuple = (255, 255, 255),
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    """
    Helper: gambar teks dengan background kotak berwarna.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Background
    cv2.rectangle(
        frame,
        (x - 2, y - th - 3),
        (x + tw + 2, y + baseline + 2),
        bg_color,
        thickness=-1,
    )
    # Teks
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_keypoints_only(
    frame: np.ndarray,
    keypoints: np.ndarray,
    confidences: np.ndarray,
    conf_threshold: float = KEYPOINT_CONFIDENCE_THRESHOLD,
    color: tuple = (0, 255, 0),
) -> np.ndarray:
    """
    Gambar hanya titik keypoint tanpa garis skeleton.

    Lebih cepat dari draw_skeleton() penuh.
    Berguna untuk frame rate tinggi.

    Args:
        frame           : Frame BGR
        keypoints       : Array (17, 2) koordinat
        confidences     : Array (17,) confidence
        conf_threshold  : Min confidence
        color           : Warna keypoint

    Returns:
        Frame dengan keypoint circles saja
    """
    visible = filter_visible_keypoints(keypoints, confidences, conf_threshold)
    return _draw_keypoint_circles(frame, visible, color)
