"""
================================================================
MODUL TIMER MANAGER - MANAJEMEN WAKTU OBJEK DALAM ROI
================================================================
Modul ini bertanggung jawab untuk mencatat dan menghitung
berapa lama setiap objek berada di dalam Restricted Area (ROI).

Logika:
- Saat objek pertama kali masuk ROI → catat entry_time
- Setiap frame → hitung duration = current_time - entry_time
- Saat objek keluar ROI → hapus entry_time

Kelas utama:
    TimerManager    - Manajemen timer semua objek di dalam ROI

Contoh penggunaan:
    timer = TimerManager()

    # Objek ID=1 masuk ROI
    timer.start(object_id=1)

    # Hitung durasi
    duration = timer.get_duration(object_id=1)

    # Objek keluar ROI
    timer.stop(object_id=1)
================================================================
"""

import logging
from datetime import datetime
from typing import Optional

# ============================================================
# SETUP LOGGING INTERNAL
# ============================================================
logger = logging.getLogger(__name__)


class TimerManager:
    """
    Manajemen timer untuk semua objek yang berada di dalam ROI.

    Setiap objek yang memasuki ROI dicatat waktu masuknya.
    Durasi dihitung secara real-time dari waktu entry hingga sekarang.
    Saat objek keluar, timernya dihapus.

    Attributes:
        _entry_times    : Dict {object_id: datetime} waktu masuk ROI
        _exit_times     : Dict {object_id: datetime} waktu keluar ROI
        _max_durations  : Dict {object_id: float} durasi max tercatat
    """

    def __init__(self):
        """Inisialisasi TimerManager dengan storage kosong."""
        # Menyimpan waktu pertama kali objek masuk ROI
        self._entry_times: dict[int, datetime] = {}

        # Menyimpan waktu terakhir keluar ROI (untuk histori)
        self._exit_times: dict[int, datetime] = {}

        # Durasi maksimum yang pernah dicapai per objek
        self._max_durations: dict[int, float] = {}

    def start(self, object_id: int) -> None:
        """
        Mulai timer untuk objek yang baru masuk ROI.

        Jika objek sudah memiliki timer aktif, tidak ada yang berubah.
        (Mencegah reset timer saat objek sudah di dalam ROI)

        Args:
            object_id: ID unik objek dari ByteTrack
        """
        if object_id not in self._entry_times:
            self._entry_times[object_id] = datetime.now()
            logger.debug(
                f"Timer MULAI: Object ID={object_id} "
                f"masuk ROI pukul {self._entry_times[object_id].strftime('%H:%M:%S')}"
            )

    def stop(self, object_id: int) -> Optional[float]:
        """
        Hentikan timer untuk objek yang keluar ROI.

        Args:
            object_id: ID unik objek

        Returns:
            Durasi total dalam ROI (detik), atau None jika timer tidak ada
        """
        if object_id not in self._entry_times:
            return None

        # Simpan waktu keluar
        self._exit_times[object_id] = datetime.now()

        # Hitung durasi terakhir
        duration = self.get_duration(object_id)

        # Update max duration
        self._max_durations[object_id] = max(
            self._max_durations.get(object_id, 0.0),
            duration or 0.0,
        )

        # Hapus entry time (timer berhenti)
        del self._entry_times[object_id]

        logger.debug(
            f"Timer BERHENTI: Object ID={object_id}, "
            f"durasi={duration:.1f}s"
        )

        return duration

    def get_duration(self, object_id: int) -> Optional[float]:
        """
        Hitung durasi objek berada di dalam ROI hingga saat ini.

        Args:
            object_id: ID unik objek

        Returns:
            Durasi dalam detik (float), atau None jika tidak ada timer
        """
        if object_id not in self._entry_times:
            return None

        entry_time = self._entry_times[object_id]
        duration = (datetime.now() - entry_time).total_seconds()

        return duration

    def is_active(self, object_id: int) -> bool:
        """
        Cek apakah objek sedang aktif di dalam ROI.

        Args:
            object_id: ID unik objek

        Returns:
            True jika timer aktif (objek di dalam ROI)
        """
        return object_id in self._entry_times

    def get_all_active(self) -> dict[int, float]:
        """
        Mendapatkan semua objek aktif beserta durasinya.

        Returns:
            Dict {object_id: duration_seconds}
        """
        active = {}
        for oid in self._entry_times:
            duration = self.get_duration(oid)
            if duration is not None:
                active[oid] = duration
        return active

    def get_entry_time(self, object_id: int) -> Optional[datetime]:
        """
        Mendapatkan waktu masuk ROI objek.

        Args:
            object_id: ID unik objek

        Returns:
            Datetime waktu masuk, atau None jika tidak aktif
        """
        return self._entry_times.get(object_id)

    def clean_inactive(self, active_ids: set[int]) -> list[int]:
        """
        Hentikan timer untuk objek yang sudah tidak aktif terdeteksi.

        Dipanggil setiap frame untuk membersihkan timer objek
        yang sudah keluar dari layar (tidak terdeteksi lagi).

        Args:
            active_ids: Set ID objek yang aktif di frame saat ini

        Returns:
            List object_id yang timernya dihentikan
        """
        # ID yang punya timer tapi tidak terdeteksi di frame ini
        inactive_ids = set(self._entry_times.keys()) - active_ids
        stopped = []

        for oid in inactive_ids:
            self.stop(oid)
            stopped.append(oid)

        return stopped

    def reset(self) -> None:
        """
        Reset semua timer (saat ganti sumber video).
        """
        self._entry_times.clear()
        self._exit_times.clear()
        logger.info("TimerManager direset.")

    def get_statistics(self) -> dict:
        """
        Mendapatkan statistik timer saat ini.

        Returns:
            Dict berisi ringkasan status timer
        """
        active_durations = self.get_all_active()
        return {
            "active_objects_in_roi": len(active_durations),
            "max_current_duration": max(active_durations.values(), default=0.0),
            "active_ids": list(active_durations.keys()),
        }
