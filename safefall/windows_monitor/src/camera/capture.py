"""
SafeFall Camera Capture & Synthetic Frame Generator.
Supports live USB webcam, IP camera RTSP streams, recorded video files,
and synthetic automated testing streams.
"""

import cv2
import time
import threading
import numpy as np
from typing import Optional, Tuple, List
from ..pose.landmarks import Point3D


class CameraSource:
    def __init__(
        self,
        device_index: int = 0,
        video_path: Optional[str] = None,
        is_simulation: bool = False,
        target_fps: float = 24.0,
    ):
        self.device_index = device_index
        self.video_path = video_path
        self.is_simulation = is_simulation
        self.target_fps = target_fps
        self.frame_interval_sec = 1.0 / max(1.0, target_fps)

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False
        self._sim_step = 0
        self._sim_posture = "STANDING"

        # Threaded capture state for non-blocking 24 FPS hardware webcam streaming
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = time.monotonic()
        self._latest_ret: bool = False

    def open(self) -> bool:
        """Initializes camera resource."""
        if self.is_simulation:
            self._is_opened = True
            return True

        if self.video_path:
            self._cap = cv2.VideoCapture(self.video_path)
        else:
            # Open local webcam with DirectShow on Windows for fast startup
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.device_index)

        self._is_opened = self._cap is not None and self._cap.isOpened()
        if self._is_opened:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Start threaded capture worker for live webcam to eliminate USB I/O latency on UI thread
            if not self.video_path and not self.is_simulation:
                self._running = True
                self._thread = threading.Thread(target=self._capture_worker, daemon=True)
                self._thread.start()

        return self._is_opened

    def _reconnect_camera(self):
        """Attempts to cleanly re-initialize the hardware camera device."""
        try:
            if self._cap is not None:
                self._cap.release()
            time.sleep(0.3)
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.device_index)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                try:
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                print(f"[SafeFall Camera] Successfully reconnected to device {self.device_index}")
                return True
        except Exception as e:
            print(f"[SafeFall Camera Reconnect Notice] {e}")
        return False

    def _capture_worker(self):
        """
        Continuously polls the camera hardware in background with watchdog and auto-reconnect.
        Never crashes or hangs indefinitely on driver drops.
        """
        consecutive_failures = 0
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.3)
                if self._running:
                    self._reconnect_camera()
                continue

            try:
                ret, frame = self._cap.read()
                now = time.monotonic()
                if ret and frame is not None and frame.size > 0:
                    consecutive_failures = 0
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_timestamp = now
                        self._latest_ret = True
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 25:
                        print(f"[SafeFall Camera] Watchdog: Camera feed interrupted ({consecutive_failures} dropped frames). Reconnecting...")
                        self._reconnect_camera()
                        consecutive_failures = 0
                    time.sleep(0.01)
            except Exception as e:
                consecutive_failures += 1
                time.sleep(0.02)
                if consecutive_failures >= 15:
                    self._reconnect_camera()
                    consecutive_failures = 0

            # Paced capture: small yield to prevent DirectShow buffer contention
            time.sleep(0.005)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float, Optional[List[Point3D]]]:
        """
        Reads one frame.
        Returns (success, frame_bgr, capture_timestamp, optional_simulated_landmarks).
        Guarantees that stale or frozen frames (> 1.2s old) are rejected.
        """
        now = time.monotonic()

        if self.is_simulation:
            self._sim_step += 1
            frame, landmarks = self._generate_simulated_frame(self._sim_posture)
            return True, frame, now, landmarks

        if self._running and self._thread is not None:
            with self._lock:
                # Frame freshness check: if the frame is older than 1.2s, camera has stalled
                is_fresh = (now - self._latest_timestamp) < 1.2
                if self._latest_frame is not None and self._latest_ret and is_fresh:
                    return True, self._latest_frame.copy(), self._latest_timestamp, None
                return False, None, now, None

        if not self._is_opened or self._cap is None:
            return False, None, now, None

        try:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # Video file ended: cleanly re-open video capture to loop seamlessly on all codecs
                if self.video_path:
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self.video_path)
                    if self._cap.isOpened():
                        ret, frame = self._cap.read()
                if not ret or frame is None:
                    return False, None, now, None

            return True, frame, now, None
        except Exception:
            return False, None, now, None

    def set_simulation_posture(self, posture: str):
        """Allows test scenarios to manually inject posture states."""
        self._sim_posture = posture

    def release(self):
        """Releases camera resources."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.3)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False

    def is_active(self) -> bool:
        return self._is_opened

    def _generate_simulated_frame(self, posture: str) -> Tuple[np.ndarray, List[Point3D]]:
        """Synthesizes a realistic 640x480 frame with visual posture graphics and 33 body landmarks."""
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Background gradient (home living room ambiance)
        frame[:] = (45, 45, 52)
        # Floor line
        cv2.line(frame, (0, 420), (w, 420), (70, 70, 80), 2)

        landmarks = [Point3D(0.5, 0.5, 0.0, 1.0) for _ in range(33)]

        if posture == "STANDING":
            # Person upright: shoulders at y=0.30, hips at y=0.55, knees at y=0.75, ankles at y=0.90
            sh_y, hp_y, kn_y, ak_y = 0.28, 0.52, 0.72, 0.88
            sh_lx, sh_rx = 0.46, 0.54
            hp_lx, hp_rx = 0.47, 0.53
            kn_lx, kn_rx = 0.47, 0.53
            ak_lx, ak_rx = 0.47, 0.53

        elif posture == "SITTING":
            # Person sitting: shoulders at y=0.45, hips at y=0.65, knees at y=0.70, ankles at y=0.88
            sh_y, hp_y, kn_y, ak_y = 0.45, 0.65, 0.68, 0.85
            sh_lx, sh_rx = 0.45, 0.55
            hp_lx, hp_rx = 0.46, 0.54
            kn_lx, kn_rx = 0.56, 0.64 # knees bent forward
            ak_lx, ak_rx = 0.56, 0.64

        elif posture == "LYING":
            # Person horizontal on floor: y near 0.82
            sh_y, hp_y, kn_y, ak_y = 0.82, 0.84, 0.85, 0.86
            sh_lx, sh_rx = 0.20, 0.24
            hp_lx, hp_rx = 0.42, 0.46
            kn_lx, kn_rx = 0.62, 0.66
            ak_lx, ak_rx = 0.80, 0.84

        else: # UNKNOWN
            sh_y, hp_y, kn_y, ak_y = 0.5, 0.5, 0.5, 0.5
            sh_lx, sh_rx = 0.5, 0.5
            hp_lx, hp_rx = 0.5, 0.5
            kn_lx, kn_rx = 0.5, 0.5
            ak_lx, ak_rx = 0.5, 0.5

        # Populate landmarks
        landmarks[11] = Point3D(sh_lx, sh_y, 0.0, 0.95)
        landmarks[12] = Point3D(sh_rx, sh_y, 0.0, 0.95)
        landmarks[23] = Point3D(hp_lx, hp_y, 0.0, 0.95)
        landmarks[24] = Point3D(hp_rx, hp_y, 0.0, 0.95)
        landmarks[25] = Point3D(kn_lx, kn_y, 0.0, 0.95)
        landmarks[26] = Point3D(kn_rx, kn_y, 0.0, 0.95)
        landmarks[27] = Point3D(ak_lx, ak_y, 0.0, 0.95)
        landmarks[28] = Point3D(ak_rx, ak_y, 0.0, 0.95)

        # Draw wireframe body on canvas
        for pt in [landmarks[11], landmarks[12], landmarks[23], landmarks[24], landmarks[25], landmarks[26], landmarks[27], landmarks[28]]:
            px = int(pt.x * w)
            py = int(pt.y * h)
            cv2.circle(frame, (px, py), 6, (0, 255, 128), -1)

        # Draw connections
        def _line(p1, p2):
            cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (0, 200, 255), 3)

        _line(landmarks[11], landmarks[12])
        _line(landmarks[11], landmarks[23])
        _line(landmarks[12], landmarks[24])
        _line(landmarks[23], landmarks[24])
        _line(landmarks[23], landmarks[25])
        _line(landmarks[24], landmarks[26])
        _line(landmarks[25], landmarks[27])
        _line(landmarks[26], landmarks[28])

        # Status text overlay
        cv2.putText(frame, f"SafeFall Simulation Mode | Posture: {posture}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {time.strftime('%H:%M:%S')}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return frame, landmarks
