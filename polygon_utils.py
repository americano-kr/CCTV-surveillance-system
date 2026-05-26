"""
================================================================
MODUL POLYGON UTILITIES - OPERASI GEOMETRI ROI
================================================================
Modul ini menyediakan fungsi-fungsi geometri untuk:
- Mengecek apakah titik berada di dalam polygon (ROI)
- Menghitung area dan batas polygon
- Konversi format titik

Menggunakan library Shapely untuk operasi geometri yang akurat
dan efisien (termasuk kasus edge pada batas polygon).

Fungsi utama:
    is_point_in_polygon()   - Cek centroid dalam ROI
    get_polygon_bounds()    - Bounding box polygon
    validate_polygon()      - Validasi polygon minimal 3 titik

Contoh penggunaan:
    from roi.polygon_utils import is_point_in_polygon

    roi_points = [[100,100],[500,100],[500,400],[100,400]]
    centroid = (300, 250)

    if is_point_in_polygon(centroid, roi_points):
        print("Objek DALAM area terlarang!")
================================================================
"""

import logging
from typing import Optional

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logging.warning(
        "Shapely tidak tersedia. Menggunakan ray-casting fallback."
    )

# ============================================================
# SETUP LOGGING INTERNAL
# ============================================================
logger = logging.getLogger(__name__)


def is_point_in_polygon(
    point: tuple[int, int],
    polygon_points: list[list[int]],
) -> bool:
    """
    Cek apakah sebuah titik berada di dalam polygon ROI.

    Menggunakan Shapely Point.within(Polygon) jika tersedia,
    atau ray-casting algorithm sebagai fallback.

    Args:
        point           : Koordinat titik (x, y) yang dicek
        polygon_points  : List titik polygon [[x1,y1],[x2,y2],...]
                          Minimal 3 titik untuk polygon valid.

    Returns:
        True  jika titik di dalam atau di batas polygon
        False jika di luar polygon atau polygon tidak valid

    Catatan:
        Titik pada batas polygon dianggap INSIDE (contains).
    """
    # ============================================================
    # VALIDASI INPUT
    # ============================================================
    if len(polygon_points) < 3:
        # Polygon tidak valid (butuh minimal 3 titik)
        return False

    if point is None:
        return False

    # ============================================================
    # CEK MENGGUNAKAN SHAPELY (AKURAT)
    # ============================================================
    if SHAPELY_AVAILABLE:
        try:
            shapely_point = Point(point[0], point[1])
            shapely_polygon = Polygon(polygon_points)

            # .contains() = di dalam (tidak termasuk batas)
            # .within()   = di dalam atau di batas
            return shapely_polygon.contains(shapely_point) or \
                   shapely_polygon.boundary.contains(shapely_point)

        except Exception as e:
            logger.warning(f"Shapely error: {e}, menggunakan fallback.")
            # Fall through ke ray-casting

    # ============================================================
    # FALLBACK: RAY-CASTING ALGORITHM
    # ============================================================
    return _ray_casting(point, polygon_points)


def _ray_casting(
    point: tuple[int, int],
    polygon: list[list[int]],
) -> bool:
    """
    Implementasi Ray-Casting Algorithm untuk point-in-polygon.

    Algoritma ini menembakkan sinar horizontal dari titik ke kanan
    dan menghitung berapa kali sinar memotong sisi polygon.
    - Ganjil  = di dalam polygon
    - Genap   = di luar polygon

    Args:
        point   : Koordinat (x, y)
        polygon : List titik polygon

    Returns:
        True jika di dalam polygon
    """
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Cek apakah garis dari titik j ke i memotong garis horizontal y
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        )
        if intersect:
            inside = not inside

        j = i

    return inside


def get_polygon_bounds(
    polygon_points: list[list[int]],
) -> Optional[tuple[int, int, int, int]]:
    """
    Mendapatkan bounding box (kotak pembatas) dari polygon ROI.

    Args:
        polygon_points: List titik polygon [[x1,y1],...]

    Returns:
        Tuple (min_x, min_y, max_x, max_y) atau None jika kosong
    """
    if not polygon_points:
        return None

    try:
        xs = [p[0] for p in polygon_points]
        ys = [p[1] for p in polygon_points]

        return (min(xs), min(ys), max(xs), max(ys))

    except Exception as e:
        logger.error(f"Error menghitung bounds polygon: {e}")
        return None


def get_polygon_area(polygon_points: list[list[int]]) -> float:
    """
    Menghitung luas area polygon dalam satuan pixel persegi.

    Menggunakan Shoelace Formula (Gauss's area formula).

    Args:
        polygon_points: List titik polygon [[x1,y1],...]

    Returns:
        Luas polygon dalam pixel²
    """
    if len(polygon_points) < 3:
        return 0.0

    if SHAPELY_AVAILABLE:
        try:
            polygon = Polygon(polygon_points)
            return polygon.area
        except Exception:
            pass

    # Fallback: Shoelace Formula
    n = len(polygon_points)
    area = 0.0
    j = n - 1
    for i in range(n):
        xi, yi = polygon_points[i]
        xj, yj = polygon_points[j]
        area += (xj + xi) * (yj - yi)
        j = i

    return abs(area) / 2.0


def validate_polygon(polygon_points: list) -> tuple[bool, str]:
    """
    Validasi apakah polygon valid untuk digunakan sebagai ROI.

    Checks:
    - Minimal 3 titik
    - Setiap titik adalah [x, y]
    - Nilai koordinat adalah angka positif

    Args:
        polygon_points: Data polygon yang akan divalidasi

    Returns:
        Tuple (is_valid: bool, message: str)
    """
    if not polygon_points:
        return False, "Polygon kosong - belum ada titik ROI."

    if len(polygon_points) < 3:
        return False, f"Polygon harus memiliki minimal 3 titik (saat ini: {len(polygon_points)})."

    for i, pt in enumerate(polygon_points):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            return False, f"Titik ke-{i} tidak valid: {pt}"

        if not all(isinstance(v, (int, float)) for v in pt):
            return False, f"Koordinat titik ke-{i} bukan angka: {pt}"

        if pt[0] < 0 or pt[1] < 0:
            return False, f"Koordinat negatif di titik ke-{i}: {pt}"

    return True, f"Polygon valid: {len(polygon_points)} titik."


def scale_polygon(
    polygon_points: list[list[int]],
    original_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[list[int]]:
    """
    Skalakan koordinat polygon dari ukuran asli ke ukuran target.

    Digunakan saat ROI digambar pada resolusi berbeda dengan
    resolusi video yang diproses.

    Args:
        polygon_points  : List titik polygon asli
        original_size   : (width, height) ukuran saat ROI digambar
        target_size     : (width, height) ukuran target

    Returns:
        List titik polygon yang sudah diskalakan
    """
    if not polygon_points:
        return []

    orig_w, orig_h = original_size
    tgt_w, tgt_h = target_size

    # Faktor skala
    scale_x = tgt_w / orig_w if orig_w > 0 else 1.0
    scale_y = tgt_h / orig_h if orig_h > 0 else 1.0

    scaled = [
        [int(pt[0] * scale_x), int(pt[1] * scale_y)]
        for pt in polygon_points
    ]

    return scaled
