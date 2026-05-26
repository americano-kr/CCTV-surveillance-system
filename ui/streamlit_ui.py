"""
================================================================
STREAMLIT UI v2.0 - DASHBOARD CCTV SURVEILLANCE
================================================================
Update v2.0:
  + Slider Frame Skip
  + Slider Confidence Threshold
  + Dropdown Model Size (YOLOv8n/s/m)
  + Checkbox Enable Pose Detection
  + Dropdown Pose Model (n-pose / s-pose)
  + Integrasi PoseDetector ke pipeline utama
  + Person-only alert mode
================================================================
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings
from detection.detector import ObjectDetector
from detection.tracker import TrackManager
from logging_system.csv_logger import CSVLogger
from monitoring.intrusion_monitor import IntrusionMonitor, IntrusionStatus
from roi.roi_manager import ROIManager
from roi.polygon_utils import validate_polygon
from ui.video_renderer import VideoRenderer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)
cfg = Settings()


# ============================================================
# SESSION STATE
# ============================================================
def initialize_session_state() -> None:
    defaults = {
        "is_running": False,
        "roi_points": [],
        "detector": None,
        "pose_detector": None,
        "monitor": None,
        "renderer": None,
        "csv_logger": None,
        "roi_manager": None,
        "track_manager": None,
        "total_frames": 0,
        "total_detections": 0,
        "total_alerts": 0,
        "uploaded_video_path": None,
        "last_frame": None,
        "current_det_model": None,
        "current_pose_model": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_base_components() -> None:
    if st.session_state.roi_manager is None:
        st.session_state.roi_manager = ROIManager(cfg.ROI_CONFIG_PATH)
        st.session_state.roi_points = st.session_state.roi_manager.get_points()
    if st.session_state.csv_logger is None:
        st.session_state.csv_logger = CSVLogger(cfg.LOG_FILE_PATH)
    if st.session_state.renderer is None:
        st.session_state.renderer = VideoRenderer(cfg)
    if st.session_state.track_manager is None:
        st.session_state.track_manager = TrackManager()


def load_detector(model_name: str, confidence: float) -> bool:
    """Load / reload ObjectDetector jika model berubah."""
    if (st.session_state.detector is not None
            and st.session_state.current_det_model == model_name):
        # Update confidence saja tanpa reload model
        st.session_state.detector.confidence = confidence
        return True

    model_path = cfg.BASE_DIR / "models" / model_name
    with st.spinner(f"⏳ Memuat model {model_name}..."):
        try:
            st.session_state.detector = ObjectDetector(
                model_path=model_path,
                confidence=confidence,
                iou=cfg.DETECTION_IOU,
                imgsz=cfg.INFERENCE_IMGSZ,
                allowed_classes=cfg.ALLOWED_CLASSES,
            )
            st.session_state.current_det_model = model_name
            logger.info(f"Detector dimuat: {model_name}")
            return True
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")
            return False


def load_pose_detector(pose_model_name: str) -> bool:
    """Load / reload PoseDetector jika model berubah."""
    if (st.session_state.pose_detector is not None
            and st.session_state.current_pose_model == pose_model_name):
        return True

    model_path = cfg.BASE_DIR / "models" / pose_model_name
    with st.spinner(f"⏳ Memuat Pose Model {pose_model_name}..."):
        try:
            from pose.pose_detector import PoseDetector
            st.session_state.pose_detector = PoseDetector(
                model_path=model_path,
                confidence=cfg.POSE_BBOX_CONFIDENCE,
                pose_confidence=cfg.POSE_KEYPOINT_CONFIDENCE,
                imgsz=cfg.INFERENCE_IMGSZ,
                use_half=cfg.USE_HALF_PRECISION,
            )
            st.session_state.current_pose_model = pose_model_name
            logger.info(f"Pose Detector dimuat: {pose_model_name}")
            return True
        except Exception as e:
            st.error(f"❌ Gagal memuat pose model: {e}")
            return False


def create_monitor(threshold: float) -> IntrusionMonitor:
    """Buat IntrusionMonitor dengan alert_callback ke CSV."""
    roi_points = st.session_state.roi_points

    def alert_cb(event):
        logger_csv = st.session_state.csv_logger
        monitor = st.session_state.monitor
        if logger_csv and monitor and monitor.should_log_event(event.object_id):
            logger_csv.log_event(
                object_id=event.object_id,
                object_type=event.class_name,
                duration=event.duration or 0.0,
                status=event.status.value,
            )
            monitor.mark_logged(event.object_id)
            st.session_state.total_alerts += 1

    return IntrusionMonitor(
        roi_points=roi_points,
        threshold_seconds=threshold,
        log_cooldown_seconds=cfg.LOG_COOLDOWN_SECONDS,
        alert_class_ids={0},   # Hanya person
        on_alert_callback=alert_cb,
    )


def get_video_source(source_type, dataset_path, webcam_id):
    """Resolve sumber video ke path atau int."""
    if source_type == "Dataset Video":
        if not dataset_path:
            st.warning("⚠ Pilih video dari folder datasets/")
            return None
        return dataset_path
    elif source_type == "Upload Video":
        p = st.session_state.get("uploaded_video_path")
        if not p:
            st.warning("⚠ Upload video terlebih dahulu")
            return None
        return p
    elif source_type == "Webcam Internal":
        return 0
    elif source_type == "Webcam Eksternal":
        return int(webcam_id)
    return None


# ================================================================
# TAB: MONITORING
# ================================================================
def render_monitoring_tab(
    source_type: str,
    threshold: float,
    dataset_path: Optional[str],
    webcam_id: int,
    frame_skip: int,
    confidence: float,
    det_model_name: str,
    enable_pose: bool,
    pose_model_name: str,
    show_roi: bool,
    show_skeleton: bool,
) -> None:
    st.subheader("📹 Live Video Monitoring")

    # Tombol kontrol
    col1, col2 = st.columns([1, 1])
    with col1:
        start = st.button("▶ START", type="primary",
                          disabled=st.session_state.is_running, use_container_width=True)
    with col2:
        stop = st.button("⏹ STOP", type="secondary",
                         disabled=not st.session_state.is_running, use_container_width=True)

    if start:
        st.session_state.is_running = True
        st.rerun()
    if stop:
        st.session_state.is_running = False
        st.rerun()

    video_ph = st.empty()
    stats_ph = st.empty()

    if not st.session_state.is_running:
        if st.session_state.last_frame is not None:
            rgb = cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_BGR2RGB)
            video_ph.image(rgb, channels="RGB", use_container_width=True,
                           caption="Frame terakhir – tekan START")
        else:
            video_ph.info("⬆ Konfigurasi sidebar lalu tekan **START**")
        return

    # ============================================================
    # LOAD MODEL
    # ============================================================
    if not load_detector(det_model_name, confidence):
        st.session_state.is_running = False
        return

    if enable_pose and not load_pose_detector(pose_model_name):
        st.session_state.is_running = False
        return

    # Tentukan sumber video
    video_source = get_video_source(source_type, dataset_path, webcam_id)
    if video_source is None:
        st.session_state.is_running = False
        return

    # Reset komponen
    st.session_state.monitor = create_monitor(threshold)
    if st.session_state.track_manager:
        st.session_state.track_manager.reset()
    if st.session_state.detector:
        st.session_state.detector.reset_tracker()
    if enable_pose and st.session_state.pose_detector:
        st.session_state.pose_detector.reset_tracker()

    # ============================================================
    # BUKA VIDEO CAPTURE
    # ============================================================
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error(f"❌ Tidak dapat membuka: {video_source}")
        st.session_state.is_running = False
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    eff_fps = video_fps / frame_skip
    st.info(
        f"📽 Video | FPS: {video_fps:.1f} | Frame Skip: {frame_skip} | "
        f"Effective Processing FPS: {eff_fps:.1f} | "
        f"Pose: {'✅' if enable_pose else '❌'}"
    )

    # ============================================================
    # LOOP FRAME
    # ============================================================
    frame_count = 0

    while cap.isOpened() and st.session_state.is_running:
        ret, frame = cap.read()
        if not ret:
            st.session_state.is_running = False
            st.success("✅ Video selesai.")
            break

        frame_count += 1

        # ====================================================
        # FRAME SKIPPING OPTIMIZATION
        # ====================================================
        # Logika: hanya proses frame ke-N, lewati sisanya
        # Rumus effective_fps = video_fps / frame_skip
        # Trade-off: kecepatan ↑ vs akurasi tracking ↓
        # ====================================================
        if frame_count % frame_skip != 0:
            continue

        # Resize untuk display
        frame_display = cv2.resize(frame, (cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT))

        # Simpan preview untuk ROI tab
        if frame_count % (frame_skip * 30) == frame_skip:
            st.session_state.last_frame = frame_display.copy()

        # Resize lebih kecil untuk inferensi (lebih cepat)
        frame_infer = cv2.resize(frame, (cfg.INFERENCE_WIDTH, cfg.INFERENCE_HEIGHT))

        # ----------------------------------------
        # DETEKSI OBJEK (YOLOv8 BIASA)
        # ----------------------------------------
        detections = []
        try:
            detections = st.session_state.detector.detect(
                frame=frame_display,  # Gunakan display frame
                use_tracking=True,
                tracker=cfg.TRACKER_TYPE,
            )
        except Exception as e:
            logger.error(f"Detection error frame {frame_count}: {e}")

        # ----------------------------------------
        # POSE DETECTION (YOLOv8-Pose)
        # ----------------------------------------
        pose_results = []
        if enable_pose and st.session_state.pose_detector:
            try:
                pose_results = st.session_state.pose_detector.detect(
                    frame=frame_display,
                    use_tracking=True,
                    tracker=cfg.TRACKER_TYPE,
                    min_visible_kp=cfg.POSE_MIN_VISIBLE_KEYPOINTS,
                )
            except Exception as e:
                logger.error(f"Pose detection error frame {frame_count}: {e}")

        # Update track manager
        if st.session_state.track_manager:
            st.session_state.track_manager.update(detections)

        # ----------------------------------------
        # INTRUSION MONITORING
        # Kirim detection + pose ke monitor
        # ----------------------------------------
        events = []
        if st.session_state.monitor:
            events = st.session_state.monitor.process(
                detections=detections,
                pose_results=pose_results if enable_pose else None,
            )

        # ----------------------------------------
        # RENDER FRAME
        # ----------------------------------------
        rendered = st.session_state.renderer.render(
            frame=frame_display,
            events=events,
            roi_points=st.session_state.roi_points,
            pose_results=pose_results if enable_pose and show_skeleton else None,
            show_roi=show_roi,
            show_stats=True,
            show_skeleton=show_skeleton and enable_pose,
        )

        # Tampilkan di Streamlit
        frame_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        video_ph.image(frame_rgb, channels="RGB", use_container_width=True,
                       caption=f"Frame #{frame_count} | Det: {len(detections)} | Pose: {len(pose_results)}")

        # Update statistik
        st.session_state.total_frames = frame_count
        st.session_state.total_detections += len(detections)

        alerts = sum(1 for e in events if e.is_alert)
        inside = sum(1 for e in events if e.is_inside)
        persons = sum(1 for e in events if e.class_id == 0)

        stats_ph.markdown(
            f"**Frame:** `{frame_count}` | "
            f"**Deteksi:** `{len(detections)}` | "
            f"**Person:** `{persons}` | "
            f"**Pose:** `{len(pose_results)}` | "
            f"**In ROI:** `{inside}` | "
            f"**🚨 Alert:** `{alerts}`"
        )

    cap.release()


# ================================================================
# TAB: ROI SETUP
# ================================================================
def render_roi_tab() -> None:
    st.subheader("🗺 Konfigurasi Restricted Area (ROI)")

    st.info(
        "**Cara menentukan ROI:**\n"
        "1. Jalankan video sebentar → frame akan ter-capture otomatis\n"
        "2. Input koordinat titik polygon di bawah\n"
        "3. Klik **Simpan ROI**\n\n"
        "Frame size: **960 × 540** pixel"
    )

    current = st.session_state.roi_points

    col1, col2 = st.columns([3, 1])
    with col1:
        if len(current) >= 3:
            st.success(f"✅ ROI Aktif — {len(current)} titik polygon")
        else:
            st.warning("⚠ ROI belum dikonfigurasi")
    with col2:
        if st.button("🗑 Hapus ROI", use_container_width=True):
            st.session_state.roi_points = []
            m = st.session_state.roi_manager
            if m:
                m.clear(); m.save()
            st.rerun()

    st.divider()

    # Preview frame
    if st.session_state.last_frame is not None:
        preview = st.session_state.renderer.draw_roi_preview(
            st.session_state.last_frame.copy(), current, False
        )
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                 channels="RGB", use_container_width=True,
                 caption="Preview ROI pada frame terakhir")
    else:
        st.info("Jalankan video untuk melihat preview frame.")

    st.divider()

    # Form input koordinat
    st.markdown("### ➕ Input Titik Polygon ROI")
    with st.form("roi_form"):
        num_pts = st.number_input("Jumlah titik", 3, 10,
                                  value=max(4, len(current)), step=1)
        new_pts = []
        cols = st.columns(2)
        for i in range(num_pts):
            with cols[i % 2]:
                st.markdown(f"**Titik P{i+1}:**")
                c1, c2 = st.columns(2)
                dx = current[i][0] if i < len(current) else 100 + i * 80
                dy = current[i][1] if i < len(current) else 100
                with c1:
                    px = st.number_input(f"X{i+1}", 0, cfg.DISPLAY_WIDTH, int(dx), key=f"x{i}")
                with c2:
                    py = st.number_input(f"Y{i+1}", 0, cfg.DISPLAY_HEIGHT, int(dy), key=f"y{i}")
                new_pts.append([px, py])

        if st.form_submit_button("💾 Simpan ROI", type="primary", use_container_width=True):
            ok, msg = validate_polygon(new_pts)
            if ok:
                st.session_state.roi_points = new_pts
                m = st.session_state.roi_manager
                if m:
                    m.set_points(new_pts); m.save()
                if st.session_state.monitor:
                    st.session_state.monitor.update_roi(new_pts)
                st.success(f"✅ ROI disimpan: {len(new_pts)} titik")
                st.rerun()
            else:
                st.error(f"❌ {msg}")

    st.divider()
    st.markdown("### 📐 ROI Preset Cepat")
    pc = st.columns(3)
    w, h = cfg.DISPLAY_WIDTH, cfg.DISPLAY_HEIGHT
    presets = {
        "🔲 Tengah": [[w//4, h//4],[3*w//4, h//4],[3*w//4, 3*h//4],[w//4, 3*h//4]],
        "⬇ Bawah": [[0, h//2],[w, h//2],[w, h],[0, h]],
        "➡ Kanan": [[w//2, 0],[w, 0],[w, h],[w//2, h]],
    }
    for col, (label, pts) in zip(pc, presets.items()):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.roi_points = pts
                m = st.session_state.roi_manager
                if m:
                    m.set_points(pts); m.save()
                if st.session_state.monitor:
                    st.session_state.monitor.update_roi(pts)
                st.success(f"✅ Preset '{label}' diterapkan")
                st.rerun()


# ================================================================
# TAB: LOG
# ================================================================
def render_log_tab() -> None:
    st.subheader("📋 Log Kejadian Intrusi")
    csv_log = st.session_state.csv_logger
    if not csv_log:
        st.error("CSV Logger belum siap.")
        return

    summary = csv_log.get_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Events", summary["total_events"])
    c2.metric("Unique Objects", summary["unique_objects"])
    c3.metric("File Size", csv_log.get_file_size())

    st.divider()

    try:
        import pandas as pd
        df = csv_log.read_log()
        if df is not None and not df.empty:
            st.dataframe(df.sort_values("timestamp", ascending=False).reset_index(drop=True),
                         use_container_width=True, height=350)
            if "object_type" in df.columns:
                st.bar_chart(df["object_type"].value_counts())
            st.download_button("⬇ Download CSV",
                               df.to_csv(index=False).encode("utf-8"),
                               "intrusion_log.csv", "text/csv", type="primary")
        else:
            st.info("📭 Belum ada data log.")
    except ImportError:
        rows = csv_log.read_log_raw()
        st.write(f"{len(rows)} baris log")
        for r in rows[-20:]:
            st.text(str(r))

    st.divider()
    with st.expander("⚠ Danger Zone"):
        st.warning("Aksi ini menghapus SEMUA data log.")
        if st.button("🗑 Hapus Log", type="secondary"):
            csv_log.clear_log()
            st.success("✅ Log dihapus.")
            st.rerun()


# ================================================================
# TAB: INFO
# ================================================================
def render_info_tab() -> None:
    st.subheader("ℹ Informasi & Dokumentasi Sistem")

    st.markdown("""
## 🔬 Latar Belakang Penelitian

**Judul:** Sistem Deteksi Objek dan Monitoring Area Terlarang Menggunakan Motion Detection Berbasis CCTV

**Dataset:** DCSASS (Detection of Crime and Suspicious Activities Surveillance System)
[🔗 Kaggle](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset)

---

## ⚡ Frame Skipping — Optimasi Kecepatan

### Mengapa Diperlukan?

Video CCTV mengandung ribuan frame per menit. Melakukan inferensi
YOLO pada **setiap frame** sangat berat secara komputasi:

| Video FPS | Frame/menit | Beban Inferensi |
|---|---|---|
| 30 fps | 1.800 frame | Sangat berat |
| 30 fps + Skip=3 | 600 frame | 3× lebih ringan |
| 30 fps + Skip=5 | 360 frame | 5× lebih ringan |

### Rumus Effective FPS

```
effective_fps = video_fps / FRAME_SKIP
```

Contoh: Video 30fps dengan FRAME_SKIP=3 → hanya **10 frame/detik** yang diproses.

### Trade-off: Speed vs Accuracy

| FRAME_SKIP | Kecepatan | Akurasi Tracking | Rekomendasi |
|---|---|---|---|
| 1 | Lambat | Paling akurat | GPU powerful |
| 3 | 3× lebih cepat | Baik | CPU + GPU |
| 5 | 5× lebih cepat | Cukup | CPU only |
| 10 | 10× lebih cepat | Kadang miss | Tidak direkomendasikan |

> ByteTrack tetap mempertahankan ID objek antar frame yang di-skip
> menggunakan prediksi Kalman Filter.

---

## 🦾 Pose Detection — Deteksi Manusia Parsial

### Mengapa Diperlukan?

Object detection konvensional (YOLO biasa) **gagal** mendeteksi:
- ✗ Manusia setengah badan (tertutup meja/dinding)
- ✗ Hanya kepala dan bahu yang terlihat
- ✗ Manusia jauh dari kamera (resolusi keypoint rendah)
- ✗ Occlusion oleh objek lain

### Solusi: YOLOv8-Pose

Model YOLOv8-Pose mendeteksi **17 keypoint** tubuh manusia secara independen.
Selama minimal **3 keypoint** terlihat, manusia tetap terdeteksi.

```
Keypoint COCO yang digunakan:
  0: nose    1: left_eye    2: right_eye
  3: left_ear    4: right_ear
  5: left_shoulder    6: right_shoulder
  7: left_elbow    8: right_elbow
  9: left_wrist    10: right_wrist
  11: left_hip    12: right_hip
  13: left_knee    14: right_knee
  15: left_ankle    16: right_ankle
```

### Centroid dari Pose vs Bbox

| Metode | Kelebihan | Kekurangan |
|---|---|---|
| Bbox Centroid | Cepat, sederhana | Gagal jika bbox hilang |
| **Pose Centroid** | Akurat, tahan parsial | Sedikit lebih lambat |

**Prioritas sistem:**
1. Pose centroid (dari keypoint torso: bahu + pinggul)
2. Fallback ke bbox centroid

---

## ✅ Kelebihan Sistem v2.0

- **Lebih cepat** — frame skipping 3-5× pengurangan beban
- **Lebih tahan occlusion** — pose mendeteksi manusia parsial
- **Lebih akurat untuk surveillance** — centroid dari tubuh, bukan box
- **GPU-ready** — auto-detect CUDA, FP16 support
- **Person-only alert** — kendaraan tidak memicu false alarm

## ⚠ Kekurangan

- Pose detection **lebih berat** dari detection biasa (terutama GPU)
- Keypoint kadang **tidak stabil** untuk manusia yang bergerak cepat
- Frame skipping tinggi dapat menyebabkan **miss deteksi singkat**
- Model pose memerlukan **download tambahan** ±6MB
    """)

    st.divider()
    st.markdown("### ⚙ Status Komponen")
    status = {
        "Detector (YOLOv8)": f"✅ {st.session_state.current_det_model}" if st.session_state.detector else "⭕ Belum dimuat",
        "Pose Detector": f"✅ {st.session_state.current_pose_model}" if st.session_state.pose_detector else "⭕ Belum dimuat",
        "ROI Manager": "✅ Ready" if st.session_state.roi_manager else "❌ Error",
        "CSV Logger": "✅ Ready" if st.session_state.csv_logger else "❌ Error",
        "ROI Aktif": f"✅ {len(st.session_state.roi_points)} titik" if len(st.session_state.roi_points) >= 3 else "⚠ Belum diset",
    }
    for k, v in status.items():
        st.markdown(f"- **{k}:** {v}")

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Frame Diproses", st.session_state.total_frames)
    c2.metric("Total Deteksi", st.session_state.total_detections)
    c3.metric("Total Alert", st.session_state.total_alerts)


# ================================================================
# MAIN APP
# ================================================================
def main() -> None:
    st.set_page_config(
        page_title="CCTV Surveillance System v2.0",
        page_icon="📹",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()
    load_base_components()

    st.title("📹 Sistem Deteksi Objek & Monitoring Area Terlarang")
    st.markdown("_CCTV Surveillance System v2.0 · YOLOv8 + Pose Detection + ByteTrack_")
    st.divider()

    # ============================================================
    # SIDEBAR
    # ============================================================
    with st.sidebar:
        st.header("⚙ Konfigurasi Sistem")
        st.divider()

        # ---- SUMBER VIDEO ----
        st.subheader("📡 Sumber Video")
        source_type = st.selectbox(
            "Pilih Sumber",
            ["Dataset Video", "Upload Video", "Webcam Internal", "Webcam Eksternal"],
        )

        dataset_path = None
        webcam_id = 1

        if source_type == "Dataset Video":
            videos = _get_dataset_videos()
            if videos:
                sel = st.selectbox("Pilih Video", ["(Pilih video)"] + videos)
                dataset_path = sel if sel != "(Pilih video)" else None
            else:
                st.warning("Folder `datasets/` kosong. Download DCSASS dataset.")

        elif source_type == "Upload Video":
            uploaded = st.file_uploader("Upload Video", type=["mp4","avi","mov","mkv"])
            if uploaded:
                p = Path("/tmp") / uploaded.name
                with open(p, "wb") as f:
                    f.write(uploaded.read())
                st.session_state.uploaded_video_path = str(p)
                st.success(f"✅ {uploaded.name}")

        elif source_type == "Webcam Eksternal":
            webcam_id = st.number_input("ID Webcam", 1, 5, 1)

        st.divider()

        # ---- MODEL DETEKSI ----
        st.subheader("🤖 Model Deteksi")
        det_model_label = st.selectbox("Model YOLOv8", list(cfg.MODEL_OPTIONS.keys()))
        det_model_name = cfg.MODEL_OPTIONS[det_model_label]

        st.divider()

        # ---- KONFIGURASI DETEKSI ----
        st.subheader("🎛 Parameter Deteksi")

        # FRAME SKIP SLIDER
        frame_skip = st.slider(
            "Frame Skip",
            min_value=cfg.FRAME_SKIP_MIN,
            max_value=cfg.FRAME_SKIP_MAX,
            value=cfg.FRAME_SKIP,
            step=1,
            help=(
                "Proses 1 dari N frame.\n"
                "Skip=1: semua frame (lambat, akurat)\n"
                "Skip=3: 3× lebih cepat\n"
                "Skip=5: 5× lebih cepat (kurang akurat)"
            ),
        )

        # CONFIDENCE THRESHOLD SLIDER
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=cfg.DETECTION_CONFIDENCE,
            step=0.05,
            help="Minimum confidence score agar objek diproses.",
        )

        st.divider()

        # ---- POSE DETECTION ----
        st.subheader("🦾 Pose Detection")
        enable_pose = st.checkbox(
            "Aktifkan Pose Detection",
            value=cfg.POSE_DETECTION_ENABLED,
            help="Gunakan YOLOv8-Pose untuk deteksi manusia parsial.",
        )

        pose_model_name = "yolov8n-pose.pt"  # default
        show_skeleton = True

        if enable_pose:
            pose_model_label = st.selectbox(
                "Model Pose",
                list(cfg.POSE_MODEL_OPTIONS.keys()),
                help="n-pose: tercepat, s-pose: lebih akurat",
            )
            pose_model_name = cfg.POSE_MODEL_OPTIONS[pose_model_label]

            show_skeleton = st.checkbox("Tampilkan Skeleton", value=True,
                                        help="Gambar garis skeleton pada frame")

        st.divider()

        # ---- ALERT ----
        st.subheader("⏱ Alert Threshold")
        threshold = st.slider(
            "Durasi sebelum Alert (detik)",
            1.0, 30.0, float(cfg.ALERT_THRESHOLD_SECONDS), 0.5,
            help="Hanya berlaku untuk 'person' (kendaraan tidak alert)",
        )

        st.divider()

        # ---- TAMPILAN ----
        st.subheader("🎨 Tampilan")
        show_roi = st.checkbox("Tampilkan ROI Overlay", value=True)

        st.divider()

        # ROI status
        roi_pts = st.session_state.roi_points
        if len(roi_pts) >= 3:
            st.success(f"✅ ROI: {len(roi_pts)} titik")
        else:
            st.warning("⚠ ROI belum diset")
            st.caption("→ Tab **ROI Setup**")

        st.divider()
        st.caption("CCTV Surveillance System v2.0")
        st.caption("YOLOv8 · Pose · ByteTrack · Shapely")

    # ============================================================
    # MAIN TABS
    # ============================================================
    tab_mon, tab_roi, tab_log, tab_info = st.tabs([
        "📹 Monitoring", "🗺 ROI Setup", "📋 Log", "ℹ Info & Docs"
    ])

    with tab_mon:
        render_monitoring_tab(
            source_type=source_type,
            threshold=threshold,
            dataset_path=dataset_path,
            webcam_id=webcam_id,
            frame_skip=frame_skip,
            confidence=confidence,
            det_model_name=det_model_name,
            enable_pose=enable_pose,
            pose_model_name=pose_model_name,
            show_roi=show_roi,
            show_skeleton=show_skeleton if enable_pose else False,
        )

    with tab_roi:
        render_roi_tab()

    with tab_log:
        render_log_tab()

    with tab_info:
        render_info_tab()


def _get_dataset_videos() -> list[str]:
    vids = []
    if not cfg.DATASET_DIR.exists():
        return vids
    for ext in [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"]:
        vids.extend(str(p) for p in cfg.DATASET_DIR.rglob(f"*{ext}"))
    return sorted(vids)
