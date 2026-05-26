# 📹 CCTV Surveillance System v2.0

> Sistem Deteksi Objek dan Monitoring Area Terlarang Menggunakan Motion Detection Berbasis CCTV

**Teknologi:** YOLOv8 · YOLOv8-Pose · ByteTrack · Shapely · Streamlit · OpenCV

---

## 🗂 Struktur Folder

```
cctv_surveillance/
│
├── app.py                        ← Entry point (streamlit run app.py)
├── requirements.txt
├── README.md
│
├── config/
│   ├── __init__.py
│   ├── settings.py               ← Semua konfigurasi sistem
│   └── roi.json                  ← Penyimpanan koordinat ROI
│
├── detection/
│   ├── __init__.py
│   ├── detector.py               ← YOLOv8 object detector wrapper
│   └── tracker.py                ← TrackHistory + TrackManager
│
├── pose/                         ← [NEW v2.0] Pose Detection
│   ├── __init__.py
│   ├── pose_detector.py          ← YOLOv8-Pose wrapper + PoseResult
│   ├── keypoint_utils.py         ← COCO keypoints + centroid calculation
│   └── skeleton_renderer.py      ← Gambar skeleton pada frame
│
├── roi/
│   ├── __init__.py
│   ├── roi_manager.py            ← Load/save ROI dari JSON
│   └── polygon_utils.py          ← Point-in-polygon (Shapely + fallback)
│
├── monitoring/
│   ├── __init__.py
│   ├── intrusion_monitor.py      ← Logika intrusion + person-only alert
│   └── timer_manager.py          ← Timer durasi per objek
│
├── logging_system/
│   ├── __init__.py
│   └── csv_logger.py             ← CSV logging kejadian intrusi
│
├── ui/
│   ├── __init__.py
│   ├── streamlit_ui.py           ← Dashboard Streamlit lengkap
│   └── video_renderer.py         ← Rendering visual frame
│
├── datasets/                     ← Letakkan video DCSASS di sini
├── logs/                         ← Output CSV log otomatis
└── models/                       ← Letakkan file .pt di sini
```

---

## 🚀 Cara Instalasi dan Menjalankan

### 1. Clone / Download Proyek

```bash
# Download atau unzip proyek ke folder pilihan
cd cctv_surveillance
```

### 2. Buat Virtual Environment (Direkomendasikan)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Paket utama yang diinstall:
- `ultralytics` — YOLOv8 + ByteTrack + YOLOv8-Pose
- `opencv-python` — Pemrosesan video
- `streamlit` — Dashboard web
- `shapely` — Operasi geometri ROI
- `pandas` — Analisis data log

### 4. Download Model YOLO (Otomatis atau Manual)

**Otomatis (direkomendasikan):**
Model akan didownload otomatis saat pertama kali dijalankan.

**Manual:**
```bash
# Buat folder models
mkdir models

# Download YOLOv8n (deteksi objek)
# Taruh di: models/yolov8n.pt

# Download YOLOv8n-pose (pose detection)
# Taruh di: models/yolov8n-pose.pt
```

Atau download dari:
- https://github.com/ultralytics/assets/releases

### 5. Siapkan Dataset DCSASS

```bash
# 1. Download dari Kaggle:
#    https://www.kaggle.com/datasets/mateohervas/dcsass-dataset

# 2. Ekstrak ke folder datasets/
datasets/
├── Stealing/
│   ├── video001.mp4
│   └── video002.mp4
├── Fighting/
├── Normal/
└── ...
```

> **Catatan:** Label aktivitas (Stealing, Fighting, dll) tidak digunakan
> untuk klasifikasi. Video hanya sebagai simulasi CCTV.

### 6. Jalankan Aplikasi

```bash
streamlit run app.py
```

Buka browser: **http://localhost:8501**

---

## 🎮 Cara Penggunaan Sistem

### Step 1: Konfigurasi Sidebar

| Parameter | Penjelasan | Nilai Direkomendasikan |
|---|---|---|
| Sumber Video | Dataset/Upload/Webcam | Dataset Video |
| Model YOLOv8 | n/s/m | YOLOv8n (CPU) / YOLOv8s (GPU) |
| Frame Skip | 1-10 | 3 (CPU) / 1-2 (GPU) |
| Confidence | 0.1-0.9 | 0.4 |
| Pose Detection | On/Off | On |
| Pose Model | n-pose/s-pose | yolov8n-pose |
| Alert Threshold | 1-30 detik | 5 detik |

### Step 2: Konfigurasi ROI

1. Tekan **START** sebentar untuk capture frame
2. Tekan **STOP**
3. Buka tab **ROI Setup**
4. Masukkan koordinat titik polygon
5. Klik **Simpan ROI**

### Step 3: Mulai Monitoring

1. Pastikan ROI sudah aktif (indikator hijau di sidebar)
2. Tekan **START**
3. Sistem akan:
   - Mendeteksi objek bergerak
   - Menampilkan bounding box + skeleton
   - Memantau objek yang masuk ROI
   - Memberikan alert jika person > threshold

### Step 4: Cek Log

- Buka tab **Log** untuk melihat semua kejadian intrusi
- Download CSV untuk analisis lebih lanjut

---

## 🧠 Penjelasan Algoritma

### Frame Skipping

```python
frame_count += 1

# FRAME SKIPPING: lewati frame yang bukan kelipatan N
if frame_count % FRAME_SKIP != 0:
    continue  # Langsung ke frame berikutnya

# Hanya frame ke-N yang diproses YOLO
```

**Rumus Effective FPS:**
```
effective_fps = video_fps / FRAME_SKIP

Contoh (video 30fps):
  FRAME_SKIP=1  → 30 fps diproses  (lambat, akurat)
  FRAME_SKIP=3  → 10 fps diproses  (3× lebih cepat)
  FRAME_SKIP=5  → 6 fps diproses   (5× lebih cepat)
```

### Pose Centroid Calculation

```python
# Centroid dari pose skeleton (lebih akurat dari bbox centroid)
# Menggunakan weighted average berdasarkan confidence keypoint

cx = Σ(x_i * conf_i) / Σ(conf_i)   # untuk keypoint torso
cy = Σ(y_i * conf_i) / Σ(conf_i)

# Prioritas:
# 1. Pose centroid (dari bahu + pinggul)
# 2. Fallback: bbox centroid ((x1+x2)//2, (y1+y2)//2)
```

### Intrusion Detection v2.0

```python
for each tracked object:
    # Pilih centroid terbaik
    centroid = pose.centroid OR bbox_centroid

    # Cek apakah dalam ROI (Shapely point-in-polygon)
    in_roi = polygon.contains(Point(centroid))

    # Timer
    if in_roi:
        timer.start(object_id)        # Start jika baru masuk
        duration = now - entry_time

        # Alert HANYA untuk person
        if object.class == "person" and duration > threshold:
            status = ALERT
            log_to_csv()
    else:
        timer.stop(object_id)         # Reset jika keluar ROI
```

---

## 📊 Perbandingan Sistem

| Aspek | YOLO Biasa | YOLO + Pose (v2.0) |
|---|---|---|
| Deteksi manusia penuh | ✅ | ✅ |
| Deteksi setengah badan | ❌ | ✅ |
| Deteksi hanya bahu | ❌ | ✅ |
| Deteksi saat occlusion | Parsial | ✅ |
| Kecepatan (CPU) | Lebih cepat | Sedikit lebih lambat |
| Akurasi centroid ROI | Sedang | Tinggi |
| GPU Support | ✅ | ✅ + FP16 |

---

## ⚙ Konfigurasi Cepat (config/settings.py)

```python
# Frame skipping (ubah ini untuk performa)
FRAME_SKIP = 3          # 1=paling lambat/akurat, 10=paling cepat

# Alert threshold
ALERT_THRESHOLD_SECONDS = 5.0   # Detik sebelum alert

# Model path
MODEL_PATH = "models/yolov8n.pt"
POSE_MODEL_PATH = "models/yolov8n-pose.pt"

# Confidence
DETECTION_CONFIDENCE = 0.40
POSE_KEYPOINT_CONFIDENCE = 0.30
```

---

## 🔧 Troubleshooting

| Masalah | Solusi |
|---|---|
| Model tidak ditemukan | Cek folder `models/`, pastikan `yolov8n.pt` ada |
| Video tidak terbuka | Cek path video, coba format `.mp4` |
| Webcam tidak terdeteksi | Coba ganti ID webcam (0, 1, 2) |
| Terlalu lambat | Naikkan Frame Skip (3→5), gunakan model nano (n) |
| Pose tidak akurat | Turunkan POSE_KEYPOINT_CONFIDENCE ke 0.2 |
| Alert terlalu sensitif | Naikkan ALERT_THRESHOLD_SECONDS ke 10 |

---

## 📝 Catatan Akademik

### Dataset DCSASS

Dataset berisi video CCTV dengan kategori:
- **Normal** — aktivitas normal
- **Stealing** — aksi pencurian
- **Fighting** — perkelahian
- **Robbery** — perampokan
- **Assault** — penyerangan
- **Vandalism** — vandalisme

Dalam penelitian ini, label kategori **tidak digunakan** untuk klasifikasi.
Video digunakan sebagai **simulasi CCTV nyata** untuk menguji:
1. Akurasi deteksi objek bergerak
2. Akurasi monitoring area terlarang
3. Performa sistem secara keseluruhan

---

*Sistem ini dikembangkan untuk keperluan penelitian skripsi Computer Vision.*
*Dapat dikembangkan lebih lanjut dengan menambahkan klasifikasi aktivitas.*
