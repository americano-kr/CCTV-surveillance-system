"""
================================================================
MODUL KEYPOINT UTILITIES - UTILITAS KEYPOINT POSE COCO
================================================================
Modul ini menyediakan definisi dan fungsi utilitas untuk
memproses keypoint pose manusia standar COCO.

Keypoint COCO (17 titik):
    0  = nose (hidung)
    1  = left_eye       2  = right_eye
    3  = left_ear       4  = right_ear
    5  = left_shoulder  6  = right_shoulder
    7  = left_elbow     8  = right_elbow
    9  = left_wrist     10 = right_wrist
    11 = left_hip       12 = right_hip
    13 = left_knee      14 = right_knee
    15 = left_ankle     16 = right_ankle

Pasangan tulang (skeleton connections):
    Setiap pasangan = satu segmen tulang yang digambar

Fungsi utama:
    calculate_pose_centroid()   - Centroid dari keypoint tubuh
    filter_visible_keypoints()  - Filter keypoint yang terlihat
    is_human_partially_visible()- Cek apakah manusia parsial
================================================================
"""

import logging
from typing import Optional

import numpy as np

# ============================================================
# SETUP LOGGING
# ============================================================
logger = logging.getLogger(__name__)

# ============================================================
# DEFINISI KEYPOINT COCO (17 titik)
# ============================================================
KEYPOINT_NAMES: dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# ============================================================
# PASANGAN TULANG UNTUK SKELETON (koneksi antar keypoint)
# ============================================================
# Format: (index_A, index_B, warna_BGR)
SKELETON_CONNECTIONS: list[tuple[int, int, tuple]] = [
    # Kepala
    (0, 1, (255, 200, 0)),    # nose - left_eye
    (0, 2, (255, 200, 0)),    # nose - right_eye
    (1, 3, (255, 200, 0)),    # left_eye - left_ear
    (2, 4, (255, 200, 0)),    # right_eye - right_ear

    # Bahu ke pinggang
    (5, 6, (0, 255, 0)),      # left_shoulder - right_shoulder
    (5, 11, (0, 255, 0)),     # left_shoulder - left_hip
    (6, 12, (0, 255, 0)),     # right_shoulder - right_hip
    (11, 12, (0, 255, 0)),    # left_hip - right_hip

    # Lengan kiri
    (5, 7, (255, 100, 0)),    # left_shoulder - left_elbow
    (7, 9, (255, 100, 0)),    # left_elbow - left_wrist

    # Lengan kanan
    (6, 8, (0, 100, 255)),    # right_shoulder - right_elbow
    (8, 10, (0, 100, 255)),   # right_elbow - right_wrist

    # Kaki kiri
    (11, 13, (200, 0, 255)),  # left_hip - left_knee
    (13, 15, (200, 0, 255)),  # left_knee - left_ankle

    # Kaki kanan
    (12, 14, (255, 0, 200)),  # right_hip - right_knee
    (14, 16, (255, 0, 200)),  # right_knee - right_ankle
]

# ============================================================
# KEYPOINT UTAMA TUBUH (untuk centroid calculation)
# Menggunakan keypoint torso yang paling stabil
# ============================================================
BODY_KEYPOINTS: list[int] = [
    5,   # left_shoulder
    6,   # right_shoulder
    11,  # left_hip
    12,  # right_hip
]

# Keypoint upper body (untuk deteksi parsial setengah badan atas)
UPPER_BODY_KEYPOINTS: list[int] = [0, 5, 6, 7, 8, 9, 10]

# Threshold confidence keypoint agar dianggap terlihat
KEYPOINT_CONFIDENCE_THRESHOLD: float = 0.3


def calculate_pose_centroid(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    conf_threshold: float = KEYPOINT_CONFIDENCE_THRESHOLD,
) -> Optional[tuple[int, int]]:
    """
    Menghitung centroid (pusat) pose manusia dari keypoint tubuh.

    Strategi prioritas:
    1. Gunakan keypoint torso (bahu + pinggul) jika tersedia
    2. Fallback ke keypoint upper body jika torso tidak cukup
    3. Fallback ke semua keypoint yang terlihat
    4. Return None jika tidak ada keypoint valid

    Centroid dihitung sebagai rata-rata tertimbang berdasarkan
    confidence score masing-masing keypoint.

    Args:
        keypoints       : Array (17, 2) koordinat keypoint [x, y]
        confidences     : Array (17,) confidence score tiap keypoint
        conf_threshold  : Minimum confidence agar keypoint valid

    Returns:
        Tuple (cx, cy) centroid pose, atau None jika tidak ada data
    """
    if keypoints is None or len(keypoints) == 0:
        return None

    # ============================================================
    # COBA HITUNG DARI KEYPOINT TORSO (PALING AKURAT)
    # ============================================================
    centroid = _weighted_centroid(
        keypoints, confidences, BODY_KEYPOINTS, conf_threshold
    )
    if centroid is not None:
        return centroid

    # ============================================================
    # FALLBACK 1: UPPER BODY (untuk setengah badan atas)
    # ============================================================
    centroid = _weighted_centroid(
        keypoints, confidences, UPPER_BODY_KEYPOINTS, conf_threshold
    )
    if centroid is not None:
        return centroid

    # ============================================================
    # FALLBACK 2: SEMUA KEYPOINT YANG TERLIHAT
    # ============================================================
    all_indices = list(range(len(keypoints)))
    centroid = _weighted_centroid(
        keypoints, confidences, all_indices, conf_threshold
    )

    return centroid


def _weighted_centroid(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    indices: list[int],
    conf_threshold: float,
) -> Optional[tuple[int, int]]:
    """
    Helper: hitung centroid tertimbang dari subset keypoint.

    Rumus centroid tertimbang:
        cx = Σ(x_i * conf_i) / Σ(conf_i)
        cy = Σ(y_i * conf_i) / Σ(conf_i)

    Args:
        keypoints   : Array semua keypoint
        confidences : Array semua confidence
        indices     : Index keypoint yang digunakan
        conf_threshold: Minimum confidence

    Returns:
        Tuple (cx, cy) atau None jika tidak cukup keypoint valid
    """
    valid_x, valid_y, valid_conf = [], [], []

    for idx in indices:
        if idx >= len(keypoints) or idx >= len(confidences):
            continue

        conf = float(confidences[idx])
        if conf < conf_threshold:
            continue

        x, y = keypoints[idx]
        if x == 0 and y == 0:  # Skip koordinat (0,0) = tidak terdeteksi
            continue

        valid_x.append(float(x))
        valid_y.append(float(y))
        valid_conf.append(conf)

    if not valid_x:
        return None

    total_conf = sum(valid_conf)
    if total_conf == 0:
        return None

    cx = int(sum(x * c for x, c in zip(valid_x, valid_conf)) / total_conf)
    cy = int(sum(y * c for y, c in zip(valid_y, valid_conf)) / total_conf)

    return (cx, cy)


def filter_visible_keypoints(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    conf_threshold: float = KEYPOINT_CONFIDENCE_THRESHOLD,
) -> dict[int, tuple[int, int, float]]:
    """
    Filter keypoint yang terlihat (confidence > threshold).

    Args:
        keypoints       : Array (17, 2) koordinat [x, y]
        confidences     : Array (17,) confidence scores
        conf_threshold  : Minimum confidence

    Returns:
        Dict {keypoint_index: (x, y, confidence)} untuk kp terlihat
    """
    visible = {}

    for idx in range(min(len(keypoints), 17)):
        conf = float(confidences[idx]) if idx < len(confidences) else 0.0

        if conf >= conf_threshold:
            x, y = keypoints[idx]
            if not (x == 0 and y == 0):
                visible[idx] = (int(x), int(y), conf)

    return visible


def is_human_partially_visible(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    min_visible_keypoints: int = 3,
    conf_threshold: float = KEYPOINT_CONFIDENCE_THRESHOLD,
) -> bool:
    """
    Cek apakah manusia parsial terdeteksi (tanpa bounding box penuh).

    Digunakan untuk mendeteksi kasus:
    - Setengah badan (bahu + perut saja)
    - Hanya kepala dan bahu
    - Occlusion sebagian

    Args:
        keypoints               : Array (17, 2) keypoint koordinat
        confidences             : Array (17,) confidence
        min_visible_keypoints   : Minimum keypoint terlihat agar valid
        conf_threshold          : Minimum confidence

    Returns:
        True jika manusia parsial terdeteksi
    """
    visible = filter_visible_keypoints(keypoints, confidences, conf_threshold)
    return len(visible) >= min_visible_keypoints


def count_visible_keypoints(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    conf_threshold: float = KEYPOINT_CONFIDENCE_THRESHOLD,
) -> int:
    """
    Hitung jumlah keypoint yang terlihat.

    Args:
        keypoints       : Array (17, 2) keypoint koordinat
        confidences     : Array (17,) confidence
        conf_threshold  : Minimum confidence

    Returns:
        Jumlah keypoint yang terlihat
    """
    visible = filter_visible_keypoints(keypoints, confidences, conf_threshold)
    return len(visible)


def get_body_part_positions(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    conf_threshold: float = KEYPOINT_CONFIDENCE_THRESHOLD,
) -> dict[str, Optional[tuple[int, int]]]:
    """
    Mendapatkan posisi bagian tubuh utama.

    Returns:
        Dict {nama_bagian: (x, y)} atau None jika tidak terlihat
    """
    visible = filter_visible_keypoints(keypoints, confidences, conf_threshold)

    def get_pos(idx: int) -> Optional[tuple[int, int]]:
        if idx in visible:
            x, y, _ = visible[idx]
            return (x, y)
        return None

    return {
        "nose": get_pos(0),
        "left_shoulder": get_pos(5),
        "right_shoulder": get_pos(6),
        "left_wrist": get_pos(9),
        "right_wrist": get_pos(10),
        "left_hip": get_pos(11),
        "right_hip": get_pos(12),
    }
