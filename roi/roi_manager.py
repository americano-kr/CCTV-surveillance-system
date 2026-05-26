"""
================================================================
MODUL ROI MANAGER - MANAJEMEN RESTRICTED AREA
================================================================
Modul ini menangani penyimpanan dan pembacaan konfigurasi
Restricted Area (ROI) ke/dari file JSON.

ROI didefinisikan sebagai polygon dengan N titik sudut.
Koordinat disimpan dalam format [[x1,y1],[x2,y2],...].

Kelas utama:
    ROIManager  - Load/save polygon ROI ke file JSON

Contoh penggunaan:
    roi_mgr = ROIManager("config/roi.json")
    roi_mgr.set_points([[100,100],[500,100],[500,400],[100,400]])
    roi_mgr.save()

    points = roi_mgr.get_points()
================================================================
"""

import json
import logging
from pathlib import Path
from typing import Optional

# ============================================================
# SETUP LOGGING INTERNAL
# ============================================================
logger = logging.getLogger(__name__)


class ROIManager:
    """
    Manajemen konfigurasi ROI (Restricted Area / Polygon).

    Bertanggung jawab untuk:
    - Membaca titik polygon ROI dari file JSON
    - Menyimpan titik polygon ROI ke file JSON
    - Validasi format data ROI

    Attributes:
        config_path : Path ke file JSON konfigurasi ROI
        _points     : List titik polygon [(x,y), ...]
    """

    def __init__(self, config_path: str | Path):
        """
        Inisialisasi ROIManager.

        Args:
            config_path: Path ke file roi.json
        """
        self.config_path: Path = Path(config_path)
        self._points: list[list[int]] = []

        # Load ROI dari file saat inisialisasi
        self._load()

    def _load(self) -> None:
        """
        Membaca konfigurasi ROI dari file JSON.

        Jika file tidak ada atau format salah,
        ROI diset ke list kosong (tidak ada ROI aktif).
        """
        if not self.config_path.exists():
            logger.info(
                f"File ROI tidak ditemukan: {self.config_path}. "
                "ROI kosong, silakan definisikan area baru."
            )
            self._points = []
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # ----------------------------------------
            # Validasi struktur JSON
            # ----------------------------------------
            if "points" not in data:
                logger.warning("Format JSON tidak valid: key 'points' tidak ditemukan.")
                self._points = []
                return

            points = data["points"]

            # Validasi setiap titik adalah [x, y]
            validated_points = []
            for pt in points:
                if (
                    isinstance(pt, (list, tuple))
                    and len(pt) == 2
                    and all(isinstance(v, (int, float)) for v in pt)
                ):
                    validated_points.append([int(pt[0]), int(pt[1])])
                else:
                    logger.warning(f"Titik tidak valid dilewati: {pt}")

            self._points = validated_points

            if self._points:
                logger.info(
                    f"ROI berhasil dimuat: {len(self._points)} titik "
                    f"dari {self.config_path}"
                )
            else:
                logger.info("ROI kosong - tidak ada area terlarang aktif.")

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON ROI: {e}")
            self._points = []

        except Exception as e:
            logger.error(f"Error membaca file ROI: {e}")
            self._points = []

    def save(self) -> bool:
        """
        Menyimpan konfigurasi ROI saat ini ke file JSON.

        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            # Buat direktori jika belum ada
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "points": self._points,
                "description": "Restricted Area / ROI",
                "total_points": len(self._points),
            }

            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(
                f"ROI disimpan: {len(self._points)} titik "
                f"ke {self.config_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Gagal menyimpan ROI: {e}")
            return False

    def set_points(self, points: list[list[int]]) -> None:
        """
        Set titik polygon ROI baru.

        Args:
            points: List koordinat [[x1,y1],[x2,y2],...]
                    Minimal 3 titik untuk polygon valid.
        """
        if len(points) < 3:
            logger.warning(
                f"ROI butuh minimal 3 titik, diberikan: {len(points)}"
            )

        self._points = [[int(p[0]), int(p[1])] for p in points]
        logger.info(f"ROI diset: {len(self._points)} titik")

    def get_points(self) -> list[list[int]]:
        """
        Mendapatkan list titik polygon ROI saat ini.

        Returns:
            List [[x1,y1],[x2,y2],...] atau [] jika tidak ada ROI
        """
        return self._points.copy()

    def is_active(self) -> bool:
        """
        Cek apakah ROI aktif (memiliki polygon valid).

        Returns:
            True jika ada minimal 3 titik polygon
        """
        return len(self._points) >= 3

    def clear(self) -> None:
        """Hapus ROI saat ini (set ke kosong)."""
        self._points = []
        logger.info("ROI dihapus.")

    def reload(self) -> None:
        """Reload ROI dari file (untuk sinkronisasi perubahan eksternal)."""
        self._load()

    def get_summary(self) -> dict:
        """
        Mendapatkan ringkasan informasi ROI.

        Returns:
            Dict berisi informasi ROI saat ini
        """
        return {
            "is_active": self.is_active(),
            "num_points": len(self._points),
            "points": self._points,
            "config_file": str(self.config_path),
        }
