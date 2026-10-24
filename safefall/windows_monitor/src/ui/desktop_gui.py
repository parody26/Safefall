"""
SafeFall Windows Desktop Monitoring Application GUI.
Features live camera preview, real-time posture badges, monotonic countdown indicators,
hardware status indicators (Camera, Mic, Cloud), and test controls.
"""

import cv2
import time
import json
import os
import threading
import ctypes
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

# Windows Power Management Keep-Awake API
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def enable_windows_keep_awake():
    """Prevents Windows from sleeping, dimming the screen, or suspending USB cameras while monitoring."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
    except Exception:
        pass

def restore_windows_sleep():
    """Restores standard Windows power management sleep policy."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass

from ..camera.capture import CameraSource
from ..pose.landmarks import extract_geometric_features, Point3D
from ..pose.posture_classifier import PostureClassifier, Posture
from ..pose.real_pose_detector import RealPoseDetector
from ..detection.temporal_detector import TemporalFallDetector
from ..detection.activity_classifier import ActivityState
from ..detection.state_machine import EmergencyStateMachine, EmergencyState
from ..alarm.alarm_service import AlarmService
from ..speech.speech_service import SpeechService, VerbalIntent
from ..backend.cloud_client import CloudClient


class SafeFallDesktopApp:
    def __init__(
        self,
        root: tk.Tk,
        camera_source: CameraSource,
        state_machine: EmergencyStateMachine,
        posture_classifier: PostureClassifier,
        temporal_detector: TemporalFallDetector,
        cloud_client: CloudClient,
        pose_detector: Optional[RealPoseDetector] = None,
        debug_mode: bool = False,
    ):
        self.root = root
        self.camera_source = camera_source
        self.state_machine = state_machine
        self.classifier = posture_classifier
        self.temporal_detector = temporal_detector
        self.cloud_client = cloud_client
        self.pose_detector = pose_detector or RealPoseDetector()
        self.debug_mode = debug_mode

        self.is_monitoring_active = True
        self._current_photo = None
        self._last_frame_time = time.time()
        self._last_logged_time = 0.0
        self._last_logged_state = None
        self._last_backend_poll = 0.0

        # Threaded pose estimation & high-performance display pipeline
        self._target_fps = getattr(self.camera_source, "target_fps", 24.0)
        self._pose_lock = threading.Lock()
        self._latest_landmarks_33 = None
        self._latest_kpts = None
        self._latest_posture = Posture.UNKNOWN
        self._latest_conf = 0.0
        self._latest_features = None
        self._latest_eval_result = None
        self._is_pose_inferring = False
        self._canvas_img_id = None

        # Prevent Windows display/USB sleep during continuous surveillance
        enable_windows_keep_awake()

        # Dedicated persistent pose inference worker to eliminate thread creation churn
        self._pose_pending_frame = None
        self._pose_pending_ts = 0.0
        self._pose_worker_active = True
        self._pose_event = threading.Event()
        self._pose_worker_thread = threading.Thread(target=self._persistent_pose_worker_loop, daemon=True)
        self._pose_worker_thread.start()

        # Privacy-safe log file setup (logs metadata and decisions, NEVER saves ordinary camera images)
        logs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self._decisions_log_path = os.path.join(logs_dir, "detection_decisions.jsonl")

        self._init_window()
        self._build_ui()
        self._setup_callbacks()
        self._schedule_loop()

    def _init_window(self):
        self.root.title("SafeFall — Windows AI Fall Monitoring Engine")
        self.root.geometry("1100x720")
        self.root.minsize(950, 650)
        self.root.configure(bg="#121417")

        # Custom styling
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#121417", foreground="#E0E6ED")

    def _build_ui(self):
        # 1. Top Header Bar
        header = tk.Frame(self.root, bg="#1A1E24", height=65)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title_box = tk.Frame(header, bg="#1A1E24")
        title_box.pack(side=tk.LEFT, padx=20, pady=12)

        lbl_logo = tk.Label(title_box, text="🛡️ SafeFall", font=("Segoe UI", 16, "bold"), fg="#38BDF8", bg="#1A1E24")
        lbl_logo.pack(side=tk.LEFT)

        lbl_sub = tk.Label(title_box, text=" | AI Home Monitoring Station", font=("Segoe UI", 11), fg="#94A3B8", bg="#1A1E24")
        lbl_sub.pack(side=tk.LEFT)

        # Device info badge in header
        dev_box = tk.Frame(header, bg="#1A1E24")
        dev_box.pack(side=tk.RIGHT, padx=20)

        self.lbl_dev_name = tk.Label(
            dev_box,
            text=f"📍 {self.state_machine.device_name} ({self.state_machine.location_label})",
            font=("Segoe UI", 10),
            fg="#CBD5E1",
            bg="#242B35",
            padx=10,
            pady=4,
        )
        self.lbl_dev_name.pack(side=tk.RIGHT)

        # 2. Main Content Split (Left: Video & Alerts, Right: Control & Telemetry)
        content = tk.Frame(self.root, bg="#121417")
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)

        # Left Column: Video Preview and Emergency Banner
        left_col = tk.Frame(content, bg="#121417")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Emergency Countdown Banner (Visual Strobe / Alert)
        self.alert_banner = tk.Frame(left_col, bg="#242B35", height=50)
        self.alert_banner.pack(fill=tk.X, pady=(0, 10))
        self.alert_banner.pack_propagate(False)

        self.lbl_banner = tk.Label(
            self.alert_banner,
            text="🟢 SYSTEM NORMAL — Continuous surveillance active",
            font=("Segoe UI", 11, "bold"),
            fg="#22C55E",
            bg="#242B35",
        )
        self.lbl_banner.pack(expand=True)

        # Video Canvas
        self.canvas_video = tk.Canvas(left_col, bg="#0A0B0D", highlightthickness=1, highlightbackground="#2D3748")
        self.canvas_video.pack(fill=tk.BOTH, expand=True)

        # Right Column: Controls, Telemetry & Status Badges
        right_col = tk.Frame(content, bg="#1A1E24", width=340)
        right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_col.pack_propagate(False)

        self._build_telemetry_panel(right_col)

    def _build_telemetry_panel(self, parent: tk.Frame):
        p = tk.Frame(parent, bg="#1A1E24", padx=16, pady=16)
        p.pack(fill=tk.BOTH, expand=True)

        # Section: AI Posture Detection
        tk.Label(p, text="AI BIOMECHANICAL POSTURE", font=("Segoe UI", 9, "bold"), fg="#64748B", bg="#1A1E24").pack(anchor="w")

        posture_card = tk.Frame(p, bg="#242B35", padx=12, pady=10)
        posture_card.pack(fill=tk.X, pady=(6, 14))

        self.lbl_posture = tk.Label(posture_card, text="STANDING", font=("Segoe UI", 18, "bold"), fg="#38BDF8", bg="#242B35")
        self.lbl_posture.pack(anchor="w")

        self.lbl_confidence = tk.Label(posture_card, text="Confidence: 94% | Torso Angle: 8.2°", font=("Segoe UI", 9), fg="#94A3B8", bg="#242B35")
        self.lbl_confidence.pack(anchor="w", pady=(2, 0))

        # Section: AI Dynamic Activity & Kinematics
        tk.Label(p, text="DYNAMIC ACTIVITY CLASSIFICATION", font=("Segoe UI", 9, "bold"), fg="#64748B", bg="#1A1E24").pack(anchor="w")

        activity_card = tk.Frame(p, bg="#242B35", padx=12, pady=10)
        activity_card.pack(fill=tk.X, pady=(6, 14))

        self.lbl_activity_state = tk.Label(activity_card, text="NORMAL_MOVEMENT", font=("Segoe UI", 13, "bold"), fg="#22C55E", bg="#242B35")
        self.lbl_activity_state.pack(anchor="w")

        self.lbl_kinematics = tk.Label(activity_card, text="Vy: +0.00 | w: 0.0°/s | Smooth: 1.00", font=("Segoe UI", 8), fg="#94A3B8", bg="#242B35")
        self.lbl_kinematics.pack(anchor="w", pady=(2, 0))

        self.lbl_trigger_reason = tk.Label(activity_card, text="Reason: Baseline continuous monitoring", font=("Segoe UI", 8), fg="#64748B", bg="#242B35", wraplength=280, justify="left")
        self.lbl_trigger_reason.pack(anchor="w", pady=(2, 0))

        # Section: Diagnostics & Hardware Status
        tk.Label(p, text="SYSTEM STATUS & SENSORS", font=("Segoe UI", 9, "bold"), fg="#64748B", bg="#1A1E24").pack(anchor="w")

        status_box = tk.Frame(p, bg="#242B35", padx=12, pady=8)
        status_box.pack(fill=tk.X, pady=(6, 14))

        self.lbl_stat_cam = tk.Label(status_box, text="📷 Camera: ONLINE (24 FPS)", font=("Segoe UI", 9), fg="#22C55E", bg="#242B35")
        self.lbl_stat_cam.pack(anchor="w", pady=2)

        # Detect active microphone name
        active_mic = "ONLINE (Default Mic)"
        try:
            mics = SpeechService.list_input_devices()
            default_m = next((m for m in mics if m.get("is_default")), None) or (mics[0] if mics else None)
            if default_m:
                active_mic = f"ONLINE ({default_m['name'][:18]})"
        except Exception:
            pass

        self.lbl_stat_mic = tk.Label(status_box, text=f"🎙️ Mic: {active_mic}", font=("Segoe UI", 9), fg="#22C55E", bg="#242B35")
        self.lbl_stat_mic.pack(anchor="w", pady=2)

        self.lbl_voice_status = tk.Label(status_box, text="🎙️ Voice Engine: READY", font=("Segoe UI", 9), fg="#94A3B8", bg="#242B35")
        self.lbl_voice_status.pack(anchor="w", pady=2)

        self.lbl_voice_heard = tk.Label(status_box, text="🗣️ Heard: (awaiting speech)", font=("Segoe UI", 8), fg="#64748B", bg="#242B35", wraplength=270, justify="left")
        self.lbl_voice_heard.pack(anchor="w", pady=2)

        self.lbl_stat_cloud = tk.Label(status_box, text="☁️ Cloud Sync: CONNECTED", font=("Segoe UI", 9), fg="#22C55E", bg="#242B35")
        self.lbl_stat_cloud.pack(anchor="w", pady=2)

        self.lbl_stat_det = tk.Label(status_box, text="⚡ Detection Engine: ACTIVE (24 Hz)", font=("Segoe UI", 9), fg="#38BDF8", bg="#242B35")
        self.lbl_stat_det.pack(anchor="w", pady=2)

        # Section: Monitoring Controls
        tk.Label(p, text="MONITORING CONTROLS", font=("Segoe UI", 9, "bold"), fg="#64748B", bg="#1A1E24").pack(anchor="w")

        btn_box = tk.Frame(p, bg="#1A1E24")
        btn_box.pack(fill=tk.X, pady=(6, 14))

        self.btn_toggle_mon = tk.Button(
            btn_box,
            text="⏸️ Stop Fall Detection",
            font=("Segoe UI", 10, "bold"),
            bg="#334155",
            fg="white",
            relief=tk.FLAT,
            command=self._toggle_monitoring,
            pady=6,
        )
        self.btn_toggle_mon.pack(fill=tk.X, pady=(0, 6))

        self.btn_toggle_landmarks = tk.Button(
            btn_box,
            text="👁️ Landmarks Overlay: ON",
            font=("Segoe UI", 9, "bold"),
            bg="#047857",
            fg="white",
            relief=tk.FLAT,
            command=self._toggle_landmarks,
            pady=5,
        )
        self.btn_toggle_landmarks.pack(fill=tk.X, pady=(0, 6))

        self.btn_toggle_debug = tk.Button(
            btn_box,
            text=f"🔍 Telemetry HUD: {'ON' if self.debug_mode else 'OFF'}",
            font=("Segoe UI", 9, "bold"),
            bg="#0284C7" if self.debug_mode else "#475569",
            fg="white",
            relief=tk.FLAT,
            command=self._toggle_debug,
            pady=5,
        )
        self.btn_toggle_debug.pack(fill=tk.X, pady=(0, 6))

    def _setup_callbacks(self):
        # Bind state machine changes to UI updates
        def on_change(state: EmergencyState, inc):
            self.root.after(0, self._render_state, state, inc)

        def on_voice_stat(msg: str):
            try:
                self.root.after(0, lambda: self.lbl_voice_status.configure(text=f"🎙️ {msg[:36]}", fg="#38BDF8"))
            except Exception:
                pass

        def on_voice_audio(rms: float, is_active: bool):
            try:
                if self.state_machine.current_state == EmergencyState.RESPONSE_PROCESSING:
                    bars = min(10, max(0, int(rms / 120)))
                    bar_str = "█" * bars + "░" * (10 - bars)
                    fg_c = "#22C55E" if is_active else "#94A3B8"
                    rem = self.state_machine.get_remaining_voice_sec()
                    self.root.after(0, lambda: self.lbl_voice_heard.configure(
                        text=f"🗣️ Mic: [{bar_str}] {'(Voice detected!)' if is_active else f'({rem:.0f}s left)'}",
                        fg=fg_c
                    ))
            except Exception:
                pass

        self.state_machine.on_state_change = on_change
        self.state_machine.on_publish_incident = self.cloud_client.publish_incident
        self.state_machine.on_voice_status = on_voice_stat
        self.state_machine.on_voice_audio_level = on_voice_audio

    def _render_state(self, state: EmergencyState, inc):
        if state == EmergencyState.NORMAL:
            self.alert_banner.configure(bg="#242B35")
            self.lbl_banner.configure(
                text="🟢 SYSTEM NORMAL — Continuous surveillance active",
                bg="#242B35",
                fg="#22C55E",
            )
            self.lbl_voice_status.configure(text="🎙️ Voice Engine: READY (Standing by)", fg="#94A3B8")
        elif state == EmergencyState.SUSPECTED_FALL:
            self.alert_banner.configure(bg="#B45309")
            self.lbl_banner.configure(
                text="⚠️ SUSPECTED FALL DETECTED — Confirming posture (3.0s grace)...",
                bg="#B45309",
                fg="#FEF08A",
            )
            self.lbl_voice_status.configure(text="🎙️ Voice Engine: ARMED", fg="#F59E0B")
        elif state in (EmergencyState.LOCAL_ALARM, EmergencyState.RESPONSE_PROCESSING):
            self.alert_banner.configure(bg="#DC2626")
            self.lbl_banner.configure(
                text="🚨 EMERGENCY ALARM ACTIVE — Verbal Check: Say 'I am fine' or 'Help'",
                bg="#DC2626",
                fg="#FFFFFF",
            )
            self.lbl_voice_status.configure(text="🎙️ Voice Engine: LISTENING FOR 'I AM FINE' (10s)", fg="#38BDF8")
            self.lbl_voice_heard.configure(text="🗣️ Listening for voice: Say 'I am fine' to silence alarm!", fg="#FEF08A")
        elif state == EmergencyState.CANCELLED_OR_FALSE_ALARM:
            self.alert_banner.configure(bg="#15803D")
            self.lbl_banner.configure(
                text="✅ FALSE ALARM / CONFIRMED SAFE — Alarm silenced",
                bg="#15803D",
                fg="#FFFFFF",
            )
            self.lbl_voice_status.configure(text="🎙️ Voice Engine: CONFIRMED SAFE ✅", fg="#22C55E")
            txt = inc.voice_transcript if inc and inc.voice_transcript else "I am fine"
            self.lbl_voice_heard.configure(text=f"🗣️ Safety Confirmed: '{txt}' (Alarm Silenced)", fg="#22C55E")
        elif state in (EmergencyState.EMERGENCY_TRIGGERED, EmergencyState.EMERGENCY_PUBLISHED):
            self.alert_banner.configure(bg="#7F1D1D")
            self.lbl_banner.configure(
                text=f"🆘 EMERGENCY PUBLISHED TO FAMILY — Incident ID: {inc.incident_id if inc else '---'}",
                bg="#7F1D1D",
                fg="#FCA5A5",
            )
            status_txt = inc.voice_response_status if inc else "EMERGENCY"
            self.lbl_voice_status.configure(text=f"🎙️ Voice Engine: {status_txt}", fg="#EF4444")
        elif state == EmergencyState.ACKNOWLEDGED:
            self.alert_banner.configure(bg="#1D4ED8")
            self.lbl_banner.configure(
                text=f"👁️ ALERT ACKNOWLEDGED BY FAMILY ({inc.acknowledged_by if inc else 'Admin'})",
                bg="#1D4ED8",
                fg="#BFDBFE",
            )
        elif state == EmergencyState.RESOLVED:
            self.alert_banner.configure(bg="#15803D")
            self.lbl_banner.configure(
                text=f"✅ INCIDENT RESOLVED — {inc.resolution_note if inc else 'Checked'}",
                bg="#15803D",
                fg="#BBF7D0",
            )
            self.lbl_voice_status.configure(text="🎙️ Voice Engine: READY (Standing by)", fg="#94A3B8")

    def _toggle_monitoring(self):
        self.is_monitoring_active = not self.is_monitoring_active
        if self.is_monitoring_active:
            self.btn_toggle_mon.configure(text="⏸️ Stop Fall Detection", bg="#334155")
            self.lbl_stat_det.configure(text="⚡ Detection Engine: ACTIVE", fg="#38BDF8")
        else:
            self.btn_toggle_mon.configure(text="▶️ Start Fall Detection", bg="#059669")
            self.lbl_stat_det.configure(text="⏸️ Detection Engine: PAUSED (Camera ON)", fg="#F59E0B")

    def _toggle_landmarks(self):
        self.pose_detector.show_landmarks = not self.pose_detector.show_landmarks
        if self.pose_detector.show_landmarks:
            self.btn_toggle_landmarks.configure(text="👁️ Landmarks Overlay: ON", bg="#047857")
        else:
            self.btn_toggle_landmarks.configure(text="👁️ Landmarks Overlay: OFF", bg="#475569")

    def _toggle_debug(self):
        self.debug_mode = not self.debug_mode
        self.btn_toggle_debug.configure(
            text=f"🔍 Telemetry HUD: {'ON' if self.debug_mode else 'OFF'}",
            bg="#0284C7" if self.debug_mode else "#475569",
        )

    def _log_decision(self, eval_res, now_sec: float):
        # Privacy safeguard: Log decisions and kinematic features WITHOUT saving raw ordinary camera images
        if eval_res is None:
            return
        state_changed = (eval_res.activity_state != self._last_logged_state)
        time_elapsed = (now_sec - self._last_logged_time) >= 2.0
        if state_changed or time_elapsed:
            self._last_logged_state = eval_res.activity_state
            self._last_logged_time = now_sec
            km = eval_res.kinematics
            record = {
                "timestamp_sec": round(now_sec, 3),
                "activity_state": eval_res.activity_state.value,
                "confidence": round(eval_res.confidence, 3),
                "trigger_reason": eval_res.trigger_reason,
                "kinematics": {
                    "vy": km.downward_hip_velocity,
                    "ang_vel": km.angular_velocity_deg_s,
                    "accel": km.downward_hip_acceleration,
                    "smoothness": km.trajectory_smoothness,
                    "motion_energy": km.motion_energy,
                } if km else None,
            }
            try:
                with open(self._decisions_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                pass

    def _test_live_microphone(self):
        """Runs an interactive diagnostic on the live physical microphone."""
        def _on_stat(msg: str):
            try:
                self.root.after(0, lambda: self.lbl_voice_status.configure(text=f"🎙️ {msg[:36]}", fg="#38BDF8"))
            except Exception:
                pass

        def _on_level(rms: float, is_active: bool):
            try:
                bars = min(10, max(0, int(rms / 120)))
                bar_str = "█" * bars + "░" * (10 - bars)
                fg_c = "#22C55E" if is_active else "#94A3B8"
                self.root.after(0, lambda: self.lbl_voice_heard.configure(
                    text=f"🗣️ Mic Level: [{bar_str}] {'(Speaking...)' if is_active else '(Listening for speech...)'}",
                    fg=fg_c
                ))
            except Exception:
                pass

        def _on_result(intent, transcript):
            def _update():
                try:
                    if intent == VerbalIntent.CONFIRM_SAFE:
                        self.lbl_voice_status.configure(text="🎙️ Voice Engine: CONFIRMED SAFE ✅", fg="#22C55E")
                        self.lbl_voice_heard.configure(text=f"🗣️ Heard: '{transcript}' -> [CONFIRM_SAFE] ✅ Mic Working!", fg="#22C55E")
                    elif intent == VerbalIntent.REQUEST_HELP:
                        self.lbl_voice_status.configure(text="🎙️ Voice Engine: HELP REQUESTED 🆘", fg="#EF4444")
                        self.lbl_voice_heard.configure(text=f"🗣️ Heard: '{transcript}' -> [REQUEST_HELP] 🆘 Help Detected!", fg="#EF4444")
                    else:
                        self.lbl_voice_status.configure(text=f"🎙️ Voice Engine: {intent.value}", fg="#F59E0B")
                        disp_t = transcript if transcript else "[No speech detected within 8s]"
                        self.lbl_voice_heard.configure(text=f"🗣️ Heard: '{disp_t}' -> [{intent.value}]", fg="#F59E0B")
                except Exception:
                    pass
            self.root.after(0, _update)

        self.lbl_voice_status.configure(text="🎙️ Prompting... Say 'Help' or 'I am fine'!", fg="#F59E0B")
        self.lbl_voice_heard.configure(text="🗣️ Listening on microphone for 8.0 seconds...", fg="#38BDF8")

        self.state_machine.speech_service.prompt_verbal_check(
            prompt_text="Please say Help me, or I am fine, now.",
            on_finished=lambda: self.state_machine.speech_service.listen_for_response(
                on_result=_on_result,
                timeout_sec=8.0,
                on_status=_on_stat,
                on_audio_level=_on_level,
            )
        )

    def _sync_remote_incident_status(self):
        inc = self.state_machine.active_incident
        if not inc:
            return
        inc_id = inc.incident_id

        def _worker():
            data = self.cloud_client.get_incident_status(inc_id)
            if not data:
                return
            status = data.get("incidentStatus")
            if status == "ACKNOWLEDGED" and self.state_machine.current_state == EmergencyState.EMERGENCY_PUBLISHED:
                actor = data.get("acknowledgedBy", "Family Member")
                self.root.after(0, lambda: self.state_machine.acknowledge_incident(actor, actor))
            elif status == "RESOLVED" and self.state_machine.current_state in (EmergencyState.EMERGENCY_PUBLISHED, EmergencyState.ACKNOWLEDGED):
                actor = data.get("resolvedBy", "Family Caregiver")
                note = data.get("resolutionNote", "Resolved via Family Safety App")
                self.root.after(0, lambda: self.state_machine.resolve_incident(actor, actor, note))

        threading.Thread(target=_worker, daemon=True).start()

    def _simulate_test_fall(self):
        self.camera_source.set_simulation_posture("STANDING")
        time.sleep(0.3)
        self.camera_source.set_simulation_posture("LYING")

    def _test_alarm_sound(self):
        def _beep():
            self.state_machine.alarm_service.start()
            time.sleep(1.5)
            self.state_machine.alarm_service.stop()
        threading.Thread(target=_beep, daemon=True).start()

    def _schedule_loop(self):
        delay_ms = 40
        try:
            t0 = time.monotonic()
            self._process_frame_tick()
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            target_interval_ms = int(1000.0 / max(1.0, self._target_fps))
            delay_ms = max(5, target_interval_ms - elapsed_ms)
        except Exception as e:
            print(f"[SafeFall GUI Frame Loop Error] {e}")
            delay_ms = max(10, int(1000.0 / max(1.0, self._target_fps)))
        finally:
            if getattr(self, "root", None) and self.root.winfo_exists():
                self.root.after(delay_ms, self._schedule_loop)

    def _dispatch_async_pose_worker(self, frame_bgr: np.ndarray, timestamp: float):
        """Dispatches latest frame to persistent background pose worker without thread churn."""
        if not self._is_pose_inferring:
            self._is_pose_inferring = True
            self._pose_pending_frame = frame_bgr.copy()
            self._pose_pending_ts = timestamp
            self._pose_event.set()

    def _persistent_pose_worker_loop(self):
        """Dedicated persistent background worker: processes latest available frame continuously."""
        while self._pose_worker_active:
            # Wait for frame notification or periodic timeout
            signaled = self._pose_event.wait(timeout=0.08)
            if not self._pose_worker_active:
                break
            if self._pose_pending_frame is None:
                continue

            self._pose_event.clear()
            frame_copy = self._pose_pending_frame
            timestamp = self._pose_pending_ts
            self._pose_pending_frame = None

            try:
                landmarks_33, kpts = self.pose_detector.detect_landmarks_raw(frame_copy)
                posture = Posture.UNKNOWN
                conf = 0.0
                features = None
                eval_result = None

                if landmarks_33:
                    features = extract_geometric_features(landmarks_33)
                    if features:
                        posture, conf = self.classifier.classify(features)

                if self.is_monitoring_active and posture != Posture.UNKNOWN:
                    candidate_fall = self.temporal_detector.add_observation(
                        posture=posture,
                        confidence=conf,
                        features=features,
                        timestamp=timestamp,
                        landmarks=landmarks_33,
                    )
                    eval_result = self.temporal_detector.last_activity_result
                    if candidate_fall:
                        _, enc = cv2.imencode(".jpg", frame_copy)
                        self.root.after(0, self.state_machine.handle_candidate_fall, candidate_fall, enc.tobytes())

                    self.root.after(0, self.state_machine.handle_posture_update, posture.value)
                    if eval_result:
                        self._log_decision(eval_result, timestamp or time.time())

                with self._pose_lock:
                    self._latest_landmarks_33 = landmarks_33
                    self._latest_kpts = kpts
                    self._latest_posture = posture
                    self._latest_conf = conf
                    self._latest_features = features
                    self._latest_eval_result = eval_result
            except Exception as ex:
                print(f"[SafeFall Async Pose Worker Error] {ex}")
            finally:
                self._is_pose_inferring = False

    def _process_frame_tick(self):
        ret, frame, timestamp, sim_landmarks = self.camera_source.read_frame()
        if not ret or frame is None:
            self.lbl_stat_cam.configure(text="📷 Camera: CONNECTING...", fg="#F59E0B")
            return

        self.lbl_stat_cam.configure(text=f"📷 Camera: ONLINE ({int(self._target_fps)} FPS)", fg="#22C55E")

        if sim_landmarks:
            # Synthetic simulation stream
            features = extract_geometric_features(sim_landmarks)
            posture = Posture.UNKNOWN
            conf = 0.0
            if features:
                posture, conf = self.classifier.classify(features)

            eval_result = None
            if self.is_monitoring_active and posture != Posture.UNKNOWN:
                candidate_fall = self.temporal_detector.add_observation(
                    posture=posture,
                    confidence=conf,
                    features=features,
                    timestamp=timestamp,
                    landmarks=sim_landmarks,
                )
                eval_result = self.temporal_detector.last_activity_result
                if candidate_fall:
                    _, enc = cv2.imencode(".jpg", frame)
                    self.state_machine.handle_candidate_fall(candidate_fall, enc.tobytes())
                self.state_machine.handle_posture_update(posture.value)
                if eval_result:
                    self._log_decision(eval_result, timestamp or time.time())

            self._update_posture_and_telemetry(posture, conf, features, eval_result)
        else:
            # Live optical webcam:
            # 1. Asynchronously dispatch pose inference if previous inference finished
            self._dispatch_async_pose_worker(frame, timestamp)

            # 2. Get latest available pose estimation under lock
            with self._pose_lock:
                kpts = self._latest_kpts
                posture = self._latest_posture
                conf = self._latest_conf
                features = self._latest_features
                eval_result = self._latest_eval_result

            # 3. Draw real-time skeleton overlay on display frame
            if self.pose_detector.show_landmarks and kpts is not None and hasattr(self.pose_detector, "draw_skeleton"):
                self.pose_detector.draw_skeleton(frame, kpts)

            self._update_posture_and_telemetry(posture, conf, features, eval_result)

        # Draw HUD text on video frame
        hud_color = (0, 255, 0) if posture == Posture.STANDING else ((0, 165, 255) if posture == Posture.SITTING else ((0, 0, 255) if posture == Posture.LYING else (180, 180, 180)))
        torso_angle_str = f"{features.torso_angle_deg:.1f}°" if features else "---"
        cv2.putText(frame, f"SafeFall AI | Posture: {posture.value} ({int(conf*100)}%)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"SafeFall AI | Posture: {posture.value} ({int(conf*100)}%)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Torso Tilt: {torso_angle_str} | Station: {self.state_machine.current_state.value}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Extended Debug Telemetry HUD Overlay
        if self.debug_mode and eval_result:
            self._draw_debug_hud(frame, eval_result)

        # Check for remote Family App acknowledgement or resolution
        now_mono = time.monotonic()
        if (
            self.state_machine.active_incident
            and self.state_machine.current_state in (EmergencyState.EMERGENCY_PUBLISHED, EmergencyState.ACKNOWLEDGED)
            and (now_mono - self._last_backend_poll) >= 2.0
        ):
            self._last_backend_poll = now_mono
            self._sync_remote_incident_status()

        # Render video frame to Tkinter Canvas at full 24 FPS
        self._render_video_frame(frame)

    def _update_posture_and_telemetry(self, posture: Posture, conf: float, features: Optional[GeometricFeatures], eval_result):
        torso_angle_str = f"{features.torso_angle_deg:.1f}°" if features else "---"
        self.lbl_posture.configure(text=posture.value)
        self.lbl_confidence.configure(text=f"Confidence: {int(conf * 100)}% | Torso Angle: {torso_angle_str}")

        if posture == Posture.STANDING:
            self.lbl_posture.configure(fg="#38BDF8")
        elif posture == Posture.SITTING:
            self.lbl_posture.configure(fg="#F59E0B")
        elif posture == Posture.LYING:
            self.lbl_posture.configure(fg="#EF4444")
        else:
            self.lbl_posture.configure(fg="#94A3B8")

        if eval_result:
            state_val = eval_result.activity_state.value
            self.lbl_activity_state.configure(text=state_val)
            state_colors = {
                "NORMAL_MOVEMENT": "#22C55E",
                "SITTING_DOWN": "#F59E0B",
                "GETTING_UP": "#38BDF8",
                "INTENTIONAL_LYING_DOWN": "#A855F7",
                "LYING_OR_SLEEPING": "#6366F1",
                "SUSPECTED_FALL": "#EF4444",
                "UNCERTAIN": "#94A3B8",
            }
            self.lbl_activity_state.configure(fg=state_colors.get(state_val, "#E2E8F0"))

            km = eval_result.kinematics
            if km:
                self.lbl_kinematics.configure(
                    text=f"Vy: {km.downward_hip_velocity:+.2f} | w: {km.angular_velocity_deg_s:.1f}°/s | Smooth: {km.trajectory_smoothness:.2f}"
                )
            self.lbl_trigger_reason.configure(text=f"Reason: {eval_result.trigger_reason[:48]}")

    def _draw_debug_hud(self, frame: np.ndarray, eval_result):
        km = eval_result.kinematics
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, frame.shape[0] - 80), (frame.shape[1] - 10, frame.shape[0] - 10), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        hud_act = eval_result.activity_state.value
        hud_act_color = (0, 255, 0) if "NORMAL" in hud_act else ((0, 0, 255) if "FALL" in hud_act else (255, 200, 0))
        cv2.putText(frame, f"DEBUG HUD: [STATE: {hud_act}] (Conf: {int(eval_result.confidence*100)}%)", (20, frame.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_act_color, 2, cv2.LINE_AA)

        km_str = f"Vy: {km.downward_hip_velocity:+.2f} norm/s | w: {km.angular_velocity_deg_s:.1f} deg/s | Acc: {km.downward_hip_acceleration:+.2f} | Smooth: {km.trajectory_smoothness:.2f}" if km else "Kinematics: accumulating"
        cv2.putText(frame, km_str, (20, frame.shape[0] - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Trigger: {eval_result.trigger_reason[:65]}", (20, frame.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 180, 200), 1, cv2.LINE_AA)

    def _render_video_frame(self, frame_bgr: np.ndarray):
        canvas_w = self.canvas_video.winfo_width()
        canvas_h = self.canvas_video.winfo_height()
        if canvas_w < 50 or canvas_h < 50:
            return

        fh, fw = frame_bgr.shape[:2]
        scale = min(canvas_w / fw, canvas_h / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))

        # cv2.resize with LINEAR interpolation is fast and consumes negligible CPU
        resized_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)

        # In-place update if dimensions match to prevent Tcl native image buffer memory leak
        if (
            self._current_photo is not None
            and getattr(self, "_photo_w", None) == nw
            and getattr(self, "_photo_h", None) == nh
        ):
            try:
                self._current_photo.paste(im)
                return
            except Exception:
                pass

        # If dimensions changed or paste failed, recreate cleanly and delete old Tcl image
        self._recreate_canvas_photo(im, nw, nh, canvas_w, canvas_h)

    def _recreate_canvas_photo(self, im: Image.Image, nw: int, nh: int, canvas_w: int, canvas_h: int):
        """Creates new PhotoImage and deletes the previous Tcl image to guarantee zero memory leakage."""
        old_photo = self._current_photo
        try:
            self._current_photo = ImageTk.PhotoImage(image=im)
            self._photo_w = nw
            self._photo_h = nh

            if self._canvas_img_id is None:
                self._canvas_img_id = self.canvas_video.create_image(
                    canvas_w // 2, canvas_h // 2, image=self._current_photo, anchor=tk.CENTER
                )
            else:
                self.canvas_video.itemconfig(self._canvas_img_id, image=self._current_photo)
                self.canvas_video.coords(self._canvas_img_id, canvas_w // 2, canvas_h // 2)
        finally:
            if old_photo is not None:
                try:
                    self.root.tk.call("image", "delete", str(old_photo))
                except Exception:
                    pass

    def close(self):
        """Cleans up background workers and restores Windows power settings."""
        self._pose_worker_active = False
        self._pose_event.set()
        restore_windows_sleep()
