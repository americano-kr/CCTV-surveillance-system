"""Paket deteksi objek: YOLOv8 + ByteTrack."""
from detection.detector import ObjectDetector, DetectionResult
from detection.tracker import TrackManager, TrackHistory

__all__ = ["ObjectDetector", "DetectionResult", "TrackManager", "TrackHistory"]
