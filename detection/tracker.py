"""
================================================================
MODUL TRACKER - WRAPPER & UTILITAS TRACKING
================================================================
Modul ini menyediakan utilitas pendukung untuk sistem tracking,
termasuk manajemen ID objek, riwayat jejak, dan status tracking.

Tracking ID sudah dikelola secara internal oleh ByteTrack
di dalam ObjectDetector. Modul ini menambahkan fitur tambahan:
- Menyimpan riwayat posisi objek (jejak pergerakan)
- Mendeteksi objek yang hilang (lost track)
- Statistik tracking per sesi

Kelas utama:
    TrackHistory    - Menyimpan riwayat centroid per objek
    TrackManager    - Manajemen lifecycle semua track
================================================================
"""

import logging
from collections import deque
from datetime import datetime
from typing import Optional

from detection.detector import DetectionResult

# ============================================================
# SETUP LOGGING INTERNAL
# ============================================================
logger = logging.getLogger(__name__)


class TrackHistory:
    """
    Menyimpan riwayat pergerakan satu objek tertrack.

    Setiap objek memiliki TrackHistory yang mencatat:
    - Posisi centroid per frame (untuk visualisasi trail)
    - Waktu pertama kali terdeteksi
    - Frame terakhir terlihat

    Attributes:
        track_id        : ID unik objek dari ByteTrack
        class_name      : Nama kelas objek
        centroids       : Deque centroid posisi (max_len terakhir)
        first_seen      : Timestamp pertama terdeteksi
        last_seen       : Timestamp terakhir terdeteksi
        frame_count     : Total frame objek terdeteksi
    """

    def __init__(
        self,
        track_id: int,
        class_name: str,
        max_trail_length: int = 30,
    ):
        """
        Args:
            track_id        : ID unik dari ByteTrack
            class_name      : Nama kelas objek
            max_trail_length: Panjang maksimum trail yang disimpan
        """
        self.track_id: int = track_id
        self.class_name: str = class_name

        # Deque: otomatis hapus data lama saat penuh
        self.centroids: deque = deque(maxlen=max_trail_length)

        # Timestamp lifecycle
        self.first_seen: datetime = datetime.now()
        self.last_seen: datetime = datetime.now()

        # Counter statistik
        self.frame_count: int = 0

    def update(self, centroid: tuple[int, int]) -> None:
        """
        Update posisi terbaru objek.

        Args:
            centroid: Koordinat (cx, cy) centroid bounding box
        """
        self.centroids.append(centroid)
        self.last_seen = datetime.now()
        self.frame_count += 1

    @property
    def age_seconds(self) -> float:
        """Hitung usia track dalam detik sejak pertama terdeteksi."""
        return (datetime.now() - self.first_seen).total_seconds()

    @property
    def latest_centroid(self) -> Optional[tuple[int, int]]:
        """Centroid terbaru, atau None jika belum ada data."""
        if self.centroids:
            return self.centroids[-1]
        return None

    def __repr__(self) -> str:
        return (
            f"TrackHistory(id={self.track_id}, "
            f"class={self.class_name}, "
            f"frames={self.frame_count})"
        )


class TrackManager:
    """
    Manajemen semua TrackHistory aktif dalam satu sesi.

    Bertanggung jawab:
    - Membuat TrackHistory baru untuk objek baru
    - Update TrackHistory yang ada
    - Menghapus track yang sudah lama tidak terlihat
    - Menyediakan statistik tracking

    Attributes:
        tracks          : Dict {track_id: TrackHistory}
        max_lost_frames : Frame tanpa deteksi sebelum track dihapus
        total_detected  : Total objek unik yang pernah dideteksi
    """

    def __init__(
        self,
        max_lost_frames: int = 30,
        max_trail_length: int = 30,
    ):
        """
        Args:
            max_lost_frames : Berapa frame tanpa deteksi sebelum dihapus
            max_trail_length: Panjang trail per track
        """
        self.tracks: dict[int, TrackHistory] = {}
        self.max_lost_frames: int = max_lost_frames
        self.max_trail_length: int = max_trail_length

        # Counter frame tanpa deteksi per track
        self._lost_counter: dict[int, int] = {}

        # Statistik sesi
        self.total_detected: int = 0
        self._active_ids: set[int] = set()

    def update(self, detections: list[DetectionResult]) -> None:
        """
        Update semua track berdasarkan deteksi frame terbaru.

        Proses:
        1. Catat ID yang aktif di frame ini
        2. Buat track baru untuk ID yang belum ada
        3. Update track yang sudah ada
        4. Tambah counter lost untuk ID yang tidak terdeteksi
        5. Hapus track yang sudah terlalu lama hilang

        Args:
            detections: List DetectionResult dari frame terbaru
        """
        # ============================================================
        # KUMPULKAN ID YANG AKTIF DI FRAME INI
        # ============================================================
        current_ids: set[int] = set()

        for det in detections:
            if det.track_id is None:
                continue

            tid = det.track_id
            current_ids.add(tid)

            # ----------------------------------------
            # Buat TrackHistory baru jika ID belum ada
            # ----------------------------------------
            if tid not in self.tracks:
                self.tracks[tid] = TrackHistory(
                    track_id=tid,
                    class_name=det.class_name,
                    max_trail_length=self.max_trail_length,
                )
                self._lost_counter[tid] = 0
                self.total_detected += 1
                logger.debug(f"Track baru: ID={tid}, class={det.class_name}")

            # ----------------------------------------
            # Update posisi track yang sudah ada
            # ----------------------------------------
            self.tracks[tid].update(det.centroid)
            self._lost_counter[tid] = 0  # Reset counter lost

        # ============================================================
        # TAMBAH COUNTER LOST UNTUK ID YANG TIDAK TERDETEKSI
        # ============================================================
        lost_ids = set(self.tracks.keys()) - current_ids
        for tid in lost_ids:
            self._lost_counter[tid] = self._lost_counter.get(tid, 0) + 1

        # ============================================================
        # HAPUS TRACK YANG SUDAH TERLALU LAMA HILANG
        # ============================================================
        expired_ids = [
            tid
            for tid, count in self._lost_counter.items()
            if count > self.max_lost_frames
        ]
        for tid in expired_ids:
            del self.tracks[tid]
            del self._lost_counter[tid]
            logger.debug(f"Track dihapus (expired): ID={tid}")

        self._active_ids = current_ids

    def get_track(self, track_id: int) -> Optional[TrackHistory]:
        """
        Mendapatkan TrackHistory berdasarkan ID.

        Args:
            track_id: ID track yang dicari

        Returns:
            TrackHistory atau None jika tidak ditemukan
        """
        return self.tracks.get(track_id)

    def get_active_tracks(self) -> list[TrackHistory]:
        """
        Mendapatkan semua track yang aktif di frame terakhir.

        Returns:
            List TrackHistory yang aktif
        """
        return [
            self.tracks[tid]
            for tid in self._active_ids
            if tid in self.tracks
        ]

    def reset(self) -> None:
        """
        Reset semua track (saat ganti sumber video).
        """
        self.tracks.clear()
        self._lost_counter.clear()
        self._active_ids.clear()
        self.total_detected = 0
        logger.info("TrackManager direset.")

    def get_statistics(self) -> dict:
        """
        Mendapatkan statistik tracking sesi saat ini.

        Returns:
            Dict berisi statistik tracking
        """
        return {
            "active_tracks": len(self._active_ids),
            "total_tracks": len(self.tracks),
            "total_detected": self.total_detected,
        }
