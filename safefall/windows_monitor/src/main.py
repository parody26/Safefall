"""
SafeFall Windows Monitoring Application Main Entrypoint.
Orchestrates Camera, AI Pose Landmarking, Temporal Fall Detection,
Acoustic Beep-Beep Alarm, Speech Interaction, and Windows Desktop UI.
"""

import os
import sys
import argparse
import tkinter as tk

from src.camera.capture import CameraSource
from src.pose.posture_classifier import PostureClassifier
from src.detection.temporal_detector import TemporalFallDetector
from src.detection.state_machine import EmergencyStateMachine
from src.alarm.alarm_service import AlarmService
from src.speech.speech_service import SpeechService
from src.backend.cloud_client import CloudClient
from src.pose.real_pose_detector import RealPoseDetector
from src.ui.desktop_gui import SafeFallDesktopApp


def main():
    parser = argparse.ArgumentParser(description="SafeFall Windows AI Fall Monitoring Engine")
    parser.add_argument("--simulation", action="store_true", help="Run with simulated synthetic postures and video")
    parser.add_argument("--device-index", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--video-source", type=str, default=None, help="Path to video file for offline replay")
    parser.add_argument("--replay", type=str, default=None, help="Alias for replay mode video path")
    parser.add_argument("--debug", action="store_true", help="Enable kinematic debug telemetry HUD and decision logging")
    parser.add_argument("--camera-placement", type=str, default="table_level", choices=["table_level", "eye_level", "ceiling_mount"], help="Camera mounting preset")
    parser.add_argument("--api-url", type=str, default=os.environ.get("SAFEFALL_API_URL", "http://localhost:8080/api/v1"), help="SafeFall backend API URL")
    parser.add_argument("--device-name", type=str, default="Windows Living Room Monitor", help="Friendly device name")
    parser.add_argument("--location-label", type=str, default="Living Room", help="Room/Location label")
    parser.add_argument("--no-landmarks", action="store_true", help="Hide visual landmark overlay on video stream")
    args = parser.parse_args()

    replay_path = args.replay or args.video_source

    print("=" * 60)
    print("  [SafeFall] AI Fall Detection & Emergency Station")
    print("=" * 60)
    print(f" Mode: {'Simulation (Synthetic AI)' if args.simulation else ('Offline Replay (' + replay_path + ')' if replay_path else 'Live Optical Camera')}")
    print(f" Camera Calibration: {args.camera_placement}")
    print(f" Debug HUD / Replay Mode: {'ENABLED' if args.debug else 'DISABLED'}")
    print(f" Device Name: {args.device_name}")
    print(f" Location: {args.location_label}")
    print(f" Visual Landmarks: {'DISABLED' if args.no_landmarks else 'ENABLED (Real-Time Skeleton Overlay)'}")
    print(f" Backend Endpoint: {args.api_url}")
    print("=" * 60)

    # 1. Initialize Core Services
    camera = CameraSource(
        device_index=args.device_index,
        video_path=replay_path,
        is_simulation=args.simulation,
        target_fps=24.0,
    )
    if not camera.open():
        print(f"[SafeFall] Warning: Could not open camera {args.device_index}. Falling back to simulation mode.")
        camera.is_simulation = True
        camera.open()

    alarm = AlarmService()
    speech = SpeechService(response_timeout_sec=10.0)
    cloud_client = CloudClient(api_base_url=args.api_url)

    from src.detection.activity_classifier import CameraCalibration
    calibration = CameraCalibration.for_placement(args.camera_placement)

    classifier = PostureClassifier()
    temporal_detector = TemporalFallDetector(calibration=calibration)

    state_machine = EmergencyStateMachine(
        family_id="fam_demo_101",
        device_id="dev_win_01",
        device_name=args.device_name,
        location_label=args.location_label,
        suspected_fall_grace_sec=3.0,
        voice_response_timeout_sec=10.0,
        alarm_service=alarm,
        speech_service=speech,
    )

    pose_detector = RealPoseDetector()
    pose_detector.show_landmarks = not args.no_landmarks

    # 2. Launch GUI
    root = tk.Tk()
    app = SafeFallDesktopApp(
        root=root,
        camera_source=camera,
        state_machine=state_machine,
        posture_classifier=classifier,
        temporal_detector=temporal_detector,
        cloud_client=cloud_client,
        pose_detector=pose_detector,
        debug_mode=args.debug,
    )

    def on_closing():
        print("[SafeFall] Shutting down monitoring station...")
        app.close()
        alarm.stop()
        camera.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
