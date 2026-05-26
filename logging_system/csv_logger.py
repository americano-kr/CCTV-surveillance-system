"""
================================================================
MODUL CSV LOGGER - LOGGING KEJADIAN INTRUSI KE FILE CSV
================================================================
Modul ini bertanggung jawab untuk mencatat semua kejadian
intrusi ke file CSV sebagai data historis dan bukti kejadian.

Format log CSV:
    timestamp, object_id, object_type, duration_seconds, status

Kelas utama:
    CSVLogger   - Menulis dan membaca log intrusi ke/dari CSV

Contoh penggunaan:
    logger = CSVLogger("logs/intrusion_log.csv")
    logger.log_event(
        object_id=1,
        object_type="person",
        duration=8.5,
        status="Intrusion"
    )
    df = logger.read_log()
================================================================
"""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# ============================================================
# SETUP LOGGING INTERNAL
# ============================================================
logger = logging.getLogger(__name__)

# Header kolom CSV
CSV_HEADER = [
    "timestamp",
    "object_id",
    "object_type",
    "duration_seconds",
    "status",
]


class CSVLogger:
    """
    Logger kejadian intrusi ke file CSV.

    Setiap event intrusi yang melebihi threshold dicatat ke CSV
    sebagai rekaman permanen untuk analisis dan audit.

    Format baris CSV:
        2025-01-01 12:01:02,1,person,8.5,Intrusion

    Attributes:
        log_path    : Path ke file CSV log
        _file_ready : Flag apakah file sudah siap ditulis
    """

    def __init__(self, log_path: str | Path):
        """
        Inisialisasi CSVLogger.

        Args:
            log_path: Path ke file CSV output
        """
        self.log_path: Path = Path(log_path)
        self._file_ready: bool = False

        # Buat direktori dan header CSV
        self._initialize_log_file()

    def _initialize_log_file(self) -> None:
        """
        Inisialisasi file log CSV.

        Jika file belum ada: buat baru dengan header.
        Jika file sudah ada: tidak diubah (append mode).
        """
        try:
            # Buat direktori jika belum ada
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

            if not self.log_path.exists():
                # ----------------------------------------
                # Buat file baru dengan header
                # ----------------------------------------
                with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(CSV_HEADER)

                logger.info(f"File log CSV baru dibuat: {self.log_path}")
            else:
                logger.info(f"File log CSV ditemukan: {self.log_path}")

            self._file_ready = True

        except Exception as e:
            logger.error(f"Gagal inisialisasi file log: {e}")
            self._file_ready = False

    def log_event(
        self,
        object_id: int,
        object_type: str,
        duration: float,
        status: str = "Intrusion",
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Catat satu event intrusi ke file CSV.

        Args:
            object_id   : ID tracking objek
            object_type : Nama kelas objek ('person', 'car', dll)
            duration    : Durasi objek dalam ROI (detik)
            status      : Status event (default: 'Intrusion')
            timestamp   : Waktu kejadian (default: sekarang)

        Returns:
            True jika berhasil dicatat, False jika gagal
        """
        if not self._file_ready:
            logger.error("File log tidak siap, event tidak dicatat.")
            return False

        try:
            # Gunakan timestamp sekarang jika tidak disediakan
            if timestamp is None:
                timestamp = datetime.now()

            # Format timestamp
            ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # ============================================================
            # TULIS BARIS KE CSV (APPEND MODE)
            # ============================================================
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ts_str,
                    object_id,
                    object_type,
                    f"{duration:.2f}",
                    status,
                ])

            logger.debug(
                f"Event dicatat: ID={object_id}, "
                f"type={object_type}, "
                f"duration={duration:.2f}s, "
                f"status={status}"
            )
            return True

        except Exception as e:
            logger.error(f"Gagal mencatat event ke CSV: {e}")
            return False

    def log_intrusion_event(self, event) -> bool:
        """
        Shortcut untuk log IntrusionEvent langsung.

        Args:
            event: IntrusionEvent object dari monitoring modul

        Returns:
            True jika berhasil
        """
        return self.log_event(
            object_id=event.object_id,
            object_type=event.class_name,
            duration=event.duration or 0.0,
            status=event.status.value,
            timestamp=event.timestamp,
        )

    def read_log(self) -> Optional["pd.DataFrame"]:
        """
        Membaca seluruh log CSV sebagai DataFrame pandas.

        Returns:
            DataFrame log, atau None jika pandas tidak tersedia/file kosong
        """
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas tidak tersedia, tidak bisa membaca log sebagai DataFrame.")
            return None

        if not self.log_path.exists():
            logger.warning("File log tidak ditemukan.")
            return None

        try:
            df = pd.read_csv(self.log_path)
            logger.info(f"Log dibaca: {len(df)} baris dari {self.log_path}")
            return df

        except Exception as e:
            logger.error(f"Gagal membaca log CSV: {e}")
            return None

    def read_log_raw(self) -> list[dict]:
        """
        Membaca seluruh log CSV sebagai list dict (tanpa pandas).

        Returns:
            List of dicts berisi data log
        """
        if not self.log_path.exists():
            return []

        try:
            rows = []
            with open(self.log_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
            return rows

        except Exception as e:
            logger.error(f"Gagal membaca log raw: {e}")
            return []

    def get_summary(self) -> dict:
        """
        Mendapatkan ringkasan statistik dari file log.

        Returns:
            Dict berisi statistik log
        """
        rows = self.read_log_raw()
        if not rows:
            return {
                "total_events": 0,
                "unique_objects": 0,
                "log_file": str(self.log_path),
            }

        unique_ids = set(row.get("object_id") for row in rows)
        object_types = {}
        for row in rows:
            ot = row.get("object_type", "unknown")
            object_types[ot] = object_types.get(ot, 0) + 1

        return {
            "total_events": len(rows),
            "unique_objects": len(unique_ids),
            "object_type_counts": object_types,
            "log_file": str(self.log_path),
        }

    def clear_log(self) -> bool:
        """
        Hapus semua isi log dan mulai dari awal (dengan header baru).

        Returns:
            True jika berhasil
        """
        try:
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)

            logger.info("File log CSV dikosongkan.")
            return True

        except Exception as e:
            logger.error(f"Gagal mengosongkan log: {e}")
            return False

    def get_file_size(self) -> str:
        """
        Mendapatkan ukuran file log dalam format human-readable.

        Returns:
            String ukuran file (contoh: '12.3 KB')
        """
        if not self.log_path.exists():
            return "0 B"

        size_bytes = self.log_path.stat().st_size
        for unit in ["B", "KB", "MB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} GB"
