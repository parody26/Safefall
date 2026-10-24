"""
SafeFall Family Safety Application (Desktop / Windows / Cross-Platform).
Allows family members and caregivers to receive real-time push alerts,
inspect evidentiary camera keyframes, review voice responses ("Help me!"),
acknowledge incidents, and resolve emergencies.
"""

import os
import sys
import io
import time
import base64
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Optional, Dict, Any, List

import requests
from PIL import Image, ImageTk

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


BACKEND_URL = os.environ.get("SAFEFALL_API_URL", "http://localhost:8080/api/v1")
CAREGIVER_NAME = "Sarah (Family Member)"
CAREGIVER_UID = "usr_sarah"


class SafeFallFamilyApp:
    def __init__(self, root: tk.Tk, backend_url: str = BACKEND_URL):
        self.root = root
        self.backend_url = backend_url.rstrip("/")
        self.active_incidents: List[Dict[str, Any]] = []
        self.selected_incident: Optional[Dict[str, Any]] = None
        self._alerted_incident_ids: set = set()
        self._displayed_incident_id: Optional[str] = None
        self._last_poll_ts = time.time()
        self._poll_running = True
        self._current_photo = None

        self._init_window()
        self._create_widgets()
        self._start_notification_listener()

    def _init_window(self):
        self.root.title("SafeFall — Family Safety & Emergency Response")
        self.root.geometry("1100x720")
        self.root.minsize(950, 640)
        self.root.configure(bg="#0B0F17")

        # Configure dark ttk styles
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#0B0F17", borderwidth=0)
        style.configure("TNotebook.Tab", background="#1E293B", foreground="#94A3B8", padding=[16, 8], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#0EA5E9")], foreground=[("selected", "#FFFFFF")])

    def _create_widgets(self):
        # 1. Top Global Navigation & Status Bar
        header = tk.Frame(self.root, bg="#111827", height=60, padx=20, pady=10)
        header.pack(fill=tk.X)

        title_box = tk.Frame(header, bg="#111827")
        title_box.pack(side=tk.LEFT)

        tk.Label(title_box, text="🛡️ SafeFall", font=("Segoe UI", 16, "bold"), fg="#38BDF8", bg="#111827").pack(side=tk.LEFT)
        tk.Label(title_box, text=" | FAMILY SAFETY STATION", font=("Segoe UI", 11, "bold"), fg="#94A3B8", bg="#111827").pack(side=tk.LEFT, padx=(4, 0))

        status_box = tk.Frame(header, bg="#111827")
        status_box.pack(side=tk.RIGHT)

        self.lbl_cloud_status = tk.Label(status_box, text="🟢 Cloud Sync: CONNECTED", font=("Segoe UI", 9, "bold"), fg="#10B981", bg="#111827")
        self.lbl_cloud_status.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(status_box, text=f"👤 {CAREGIVER_NAME}", font=("Segoe UI", 9), fg="#E2E8F0", bg="#111827").pack(side=tk.LEFT)

        # 2. Emergency Alert Banner (Hidden by default, shown during active fall alert)
        self.alert_banner = tk.Frame(self.root, bg="#1E293B", padx=20, pady=10)
        self.alert_banner.pack(fill=tk.X)

        self.lbl_banner_icon = tk.Label(self.alert_banner, text="🟢", font=("Segoe UI", 14), bg="#1E293B", fg="#10B981")
        self.lbl_banner_icon.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_banner_text = tk.Label(
            self.alert_banner,
            text="ALL MONITORED RESIDENCES NORMAL — No Active Fall Emergencies",
            font=("Segoe UI", 11, "bold"),
            fg="#10B981",
            bg="#1E293B",
        )
        self.lbl_banner_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.btn_banner_action = tk.Button(
            self.alert_banner,
            text="🔄 Refresh Feed",
            font=("Segoe UI", 9, "bold"),
            bg="#334155",
            fg="white",
            relief=tk.FLAT,
            command=self._fetch_incidents,
            padx=12,
            pady=4,
        )
        self.btn_banner_action.pack(side=tk.RIGHT)

        self.btn_clear_resolved = tk.Button(
            self.alert_banner,
            text="🧹 Clear Resolved",
            font=("Segoe UI", 9, "bold"),
            bg="#1E293B",
            fg="#94A3B8",
            relief=tk.FLAT,
            command=self._clear_resolved_incidents,
            padx=10,
            pady=4,
        )
        self.btn_clear_resolved.pack(side=tk.RIGHT, padx=(0, 8))

        # 3. Main Content Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        self.tab_incidents = tk.Frame(self.notebook, bg="#0B0F17")
        self.tab_devices = tk.Frame(self.notebook, bg="#0B0F17")
        self.tab_family = tk.Frame(self.notebook, bg="#0B0F17")

        self.notebook.add(self.tab_incidents, text="🚨 Incidents & Alerts")
        self.notebook.add(self.tab_devices, text="📷 Monitoring Devices")
        self.notebook.add(self.tab_family, text="👥 Family & Contacts")

        self._build_incidents_tab(self.tab_incidents)
        self._build_devices_tab(self.tab_devices)
        self._build_family_tab(self.tab_family)

    def _build_incidents_tab(self, parent: tk.Frame):
        # Split layout: Left = Incident Feed / List, Right = Incident Evidence & Details
        container = tk.Frame(parent, bg="#0B0F17")
        container.pack(fill=tk.BOTH, expand=True)

        left_pane = tk.Frame(container, bg="#0B0F17", width=360)
        left_pane.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        # Title & Test Fall Button
        lf_header = tk.Frame(left_pane, bg="#0B0F17")
        lf_header.pack(fill=tk.X, pady=(0, 8))

        tk.Label(lf_header, text="INCIDENT STREAM", font=("Segoe UI", 10, "bold"), fg="#64748B", bg="#0B0F17").pack(side=tk.LEFT)

        btn_sim_help = tk.Button(
            lf_header,
            text="⚡ Test 'Help' Alert",
            font=("Segoe UI", 8, "bold"),
            bg="#DC2626",
            fg="white",
            relief=tk.FLAT,
            command=self._simulate_help_incident,
            padx=8,
            pady=2,
        )
        btn_sim_help.pack(side=tk.RIGHT)

        # Scrollable Incident List
        list_card = tk.Frame(left_pane, bg="#1E293B", padx=2, pady=2)
        list_card.pack(fill=tk.BOTH, expand=True)

        self.incident_listbox = tk.Listbox(
            list_card,
            bg="#1E293B",
            fg="#F8FAFC",
            font=("Segoe UI", 10),
            selectbackground="#0EA5E9",
            selectforeground="#FFFFFF",
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            activestyle="none",
        )
        self.incident_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.incident_listbox.bind("<<ListboxSelect>>", self._on_incident_selected)

        scrollbar = tk.Scrollbar(list_card, command=self.incident_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.incident_listbox.config(yscrollcommand=scrollbar.set)

        # Right Pane: Incident Evidence & Actions
        self.right_pane = tk.Frame(container, bg="#1E293B", padx=20, pady=18)
        self.right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.lbl_detail_title = tk.Label(
            self.right_pane,
            text="Select an incident from the stream to view evidence and emergency details",
            font=("Segoe UI", 13, "bold"),
            fg="#94A3B8",
            bg="#1E293B",
            wraplength=520,
            justify="left",
        )
        self.lbl_detail_title.pack(anchor="w", pady=(0, 10))

        # Evidentiary Photo Canvas
        self.photo_frame = tk.Frame(self.right_pane, bg="#0F172A", height=280)
        self.photo_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        self.lbl_photo = tk.Label(self.photo_frame, text="[No Incident Selected]", bg="#0F172A", fg="#64748B", font=("Segoe UI", 11))
        self.lbl_photo.pack(fill=tk.BOTH, expand=True)

        # Voice Response Highlight Box (Crucial user request!)
        self.voice_box = tk.Frame(self.right_pane, bg="#334155", padx=12, pady=10)
        self.voice_box.pack(fill=tk.X, pady=(0, 12))

        self.lbl_voice_title = tk.Label(
            self.voice_box,
            text="🗣️ VERBAL RESPONSE CLASSIFICATION",
            font=("Segoe UI", 9, "bold"),
            fg="#94A3B8",
            bg="#334155",
        )
        self.lbl_voice_title.pack(anchor="w")

        self.lbl_voice_transcript = tk.Label(
            self.voice_box,
            text="Waiting for incident selection...",
            font=("Segoe UI", 12, "bold"),
            fg="#FFFFFF",
            bg="#334155",
            wraplength=500,
            justify="left",
        )
        self.lbl_voice_transcript.pack(anchor="w", pady=(4, 0))

        # Metadata Tags
        self.meta_frame = tk.Frame(self.right_pane, bg="#1E293B")
        self.meta_frame.pack(fill=tk.X, pady=(0, 16))

        self.lbl_meta_time = tk.Label(self.meta_frame, text="Detected: ---", font=("Segoe UI", 9), fg="#94A3B8", bg="#1E293B")
        self.lbl_meta_time.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_meta_conf = tk.Label(self.meta_frame, text="Confidence: ---", font=("Segoe UI", 9), fg="#94A3B8", bg="#1E293B")
        self.lbl_meta_conf.pack(side=tk.LEFT, padx=(0, 20))

        self.lbl_meta_status = tk.Label(self.meta_frame, text="Status: ---", font=("Segoe UI", 9, "bold"), fg="#94A3B8", bg="#1E293B")
        self.lbl_meta_status.pack(side=tk.LEFT)

        # Emergency Action Buttons Row
        actions_bar = tk.Frame(self.right_pane, bg="#1E293B")
        actions_bar.pack(fill=tk.X)

        self.btn_ack = tk.Button(
            actions_bar,
            text="👁️ Acknowledge Incident",
            font=("Segoe UI", 10, "bold"),
            bg="#0284C7",
            fg="white",
            relief=tk.FLAT,
            command=self._acknowledge_selected,
            padx=16,
            pady=8,
            state=tk.DISABLED,
        )
        self.btn_ack.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_call = tk.Button(
            actions_bar,
            text="📞 Call Emergency (911 / 112)",
            font=("Segoe UI", 10, "bold"),
            bg="#B91C1C",
            fg="white",
            relief=tk.FLAT,
            command=self._call_emergency_dialog,
            padx=16,
            pady=8,
            state=tk.DISABLED,
        )
        self.btn_call.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_resolve = tk.Button(
            actions_bar,
            text="✅ Resolve Emergency",
            font=("Segoe UI", 10, "bold"),
            bg="#059669",
            fg="white",
            relief=tk.FLAT,
            command=self._resolve_selected_dialog,
            padx=16,
            pady=8,
            state=tk.DISABLED,
        )
        self.btn_resolve.pack(side=tk.LEFT)

    def _build_devices_tab(self, parent: tk.Frame):
        p = tk.Frame(parent, bg="#0B0F17", padx=20, pady=20)
        p.pack(fill=tk.BOTH, expand=True)

        tk.Label(p, text="CONNECTED MONITORING STATIONS", font=("Segoe UI", 12, "bold"), fg="#64748B", bg="#0B0F17").pack(anchor="w", pady=(0, 12))

        # Station Card
        dev_card = tk.Frame(p, bg="#1E293B", padx=18, pady=16)
        dev_card.pack(fill=tk.X, pady=(0, 14))

        tk.Label(dev_card, text="📷 Windows Living Room Station (dev_win_01)", font=("Segoe UI", 14, "bold"), fg="#38BDF8", bg="#1E293B").pack(anchor="w")
        tk.Label(dev_card, text="Location: Living Room | Platform: Windows x64 | Camera: 24.0 FPS (Online)", font=("Segoe UI", 10), fg="#94A3B8", bg="#1E293B").pack(anchor="w", pady=(4, 8))

        stats_row = tk.Frame(dev_card, bg="#1E293B")
        stats_row.pack(fill=tk.X)

        tk.Label(stats_row, text="● Camera: ACTIVE", font=("Segoe UI", 9, "bold"), fg="#10B981", bg="#1E293B").pack(side=tk.LEFT, padx=(0, 16))
        tk.Label(stats_row, text="● Microphone: ONLINE (Logi C270)", font=("Segoe UI", 9, "bold"), fg="#10B981", bg="#1E293B").pack(side=tk.LEFT, padx=(0, 16))
        tk.Label(stats_row, text="● Fall Engine: 24 Hz KINEMATIC DYNAMICS", font=("Segoe UI", 9, "bold"), fg="#38BDF8", bg="#1E293B").pack(side=tk.LEFT)

    def _build_family_tab(self, parent: tk.Frame):
        p = tk.Frame(parent, bg="#0B0F17", padx=20, pady=20)
        p.pack(fill=tk.BOTH, expand=True)

        tk.Label(p, text="AUTHORIZED FAMILY CAREGIVERS & EMERGENCY CONTACTS", font=("Segoe UI", 12, "bold"), fg="#64748B", bg="#0B0F17").pack(anchor="w", pady=(0, 12))

        for m in [
            {"name": "Sarah (Daughter)", "role": "Primary Caregiver (Family Admin)", "phone": "+1-555-0192", "status": "Active (This Device)"},
            {"name": "David (Son)", "role": "Caregiver", "phone": "+1-555-0193", "status": "Active (Mobile)"},
            {"name": "Nurse Evelyn", "role": "Medical Attendant", "phone": "+1-555-0199", "status": "On-Call Emergency"},
        ]:
            card = tk.Frame(p, bg="#1E293B", padx=16, pady=12)
            card.pack(fill=tk.X, pady=4)
            tk.Label(card, text=f"👤 {m['name']} — {m['role']}", font=("Segoe UI", 11, "bold"), fg="#F8FAFC", bg="#1E293B").pack(anchor="w")
            tk.Label(card, text=f"📞 Phone: {m['phone']} | Status: {m['status']}", font=("Segoe UI", 9), fg="#94A3B8", bg="#1E293B").pack(anchor="w", pady=(2, 0))

    def _start_notification_listener(self):
        """Starts background poll thread to receive real-time push alerts."""
        def _poll_worker():
            while self._poll_running:
                try:
                    resp = requests.get(f"{self.backend_url}/notifications/poll?since={self._last_poll_ts}&timeout=5", timeout=7)
                    if resp.status_code == 200:
                        data = resp.json()
                        events = data.get("events", [])
                        if events:
                            self._last_poll_ts = data.get("timestamp", time.time())
                            self.root.after(0, self._handle_incoming_events, events)
                except Exception:
                    pass
                time.sleep(1.0)

        threading.Thread(target=_poll_worker, daemon=True).start()
        # Initial load
        self.root.after(300, self._fetch_incidents)

    def _handle_incoming_events(self, events: List[Dict[str, Any]]):
        """Called whenever the backend reports newly created or updated incidents."""
        new_emergencies = []

        for inc in events:
            inc_id = inc.get("incidentId")
            status = inc.get("incidentStatus", "")
            if status in ("EMERGENCY_PUBLISHED", "EMERGENCY_TRIGGERED"):
                if inc_id and inc_id not in self._alerted_incident_ids:
                    self._alerted_incident_ids.add(inc_id)
                    new_emergencies.append(inc)

        if new_emergencies:
            # Sound emergency chime strictly ONCE when a new fall incident arrives
            self._sound_emergency_chime()
            latest_emerg = new_emergencies[-1]

            # Bring window to front
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(1500, lambda: self.root.attributes("-topmost", False))

            transcript = latest_emerg.get("voiceTranscript") or "[No verbal reassurance]"
            location = latest_emerg.get("locationLabel", "Home")
            voice_stat = latest_emerg.get("voiceResponseStatus", "NO_RESPONSE")

            # Update banner to urgent red
            self.alert_banner.configure(bg="#DC2626")
            self.lbl_banner_icon.configure(text="🚨", bg="#DC2626", fg="#FFFFFF")
            self.lbl_banner_text.configure(
                text=f"🚨 CRITICAL FALL DETECTED: {location}! User Voice: '{transcript}' ({voice_stat})",
                bg="#DC2626",
                fg="#FFFFFF",
            )
            # Auto-select newly arrived emergency
            self._fetch_incidents(auto_select_first=True)
        else:
            self._fetch_incidents(auto_select_first=False)

    def _sound_emergency_chime(self):
        def _beep():
            if HAS_WINSOUND:
                try:
                    for _ in range(2):
                        winsound.Beep(2000, 180)
                        time.sleep(0.08)
                        winsound.Beep(2400, 220)
                        time.sleep(0.15)
                except Exception:
                    pass
        threading.Thread(target=_beep, daemon=True).start()

    def _fetch_incidents(self, auto_select_first: bool = False):
        def _worker():
            try:
                r = requests.get(f"{self.backend_url}/incidents", timeout=3.0)
                if r.status_code == 200:
                    incidents = r.json().get("incidents", [])
                    self.root.after(0, self._render_incidents_list, incidents, auto_select_first)
            except Exception as e:
                print(f"[FamilyApp] Fetch error: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def _render_incidents_list(self, incidents: List[Dict[str, Any]], auto_select_first: bool = False):
        self.active_incidents = incidents
        self.incident_listbox.delete(0, tk.END)

        has_unresolved = False

        for idx, inc in enumerate(incidents):
            status = inc.get("incidentStatus", "UNKNOWN")
            loc = inc.get("locationLabel", "Home")
            detected = inc.get("detectedAt", "")[11:19]
            voice = inc.get("voiceResponseStatus", "")
            transcript = inc.get("voiceTranscript", "")

            if status in ("EMERGENCY_PUBLISHED", "EMERGENCY_TRIGGERED"):
                has_unresolved = True
                badge = "🚨 FALL (HELP!)" if voice == "REQUEST_HELP" else "🚨 EMERGENCY"
            elif status == "ACKNOWLEDGED":
                badge = "👁️ ACKNOWLEDGED"
            elif status == "RESOLVED":
                badge = "✅ RESOLVED"
            else:
                badge = f"● {status}"

            snippet = f" '{transcript}'" if transcript else ""
            line = f"{badge} | {loc} ({detected}){snippet}"
            self.incident_listbox.insert(tk.END, line)

            # Color styling
            if "FALL" in badge or "EMERGENCY" in badge:
                self.incident_listbox.itemconfig(idx, fg="#FCA5A5")
            elif "ACK" in badge:
                self.incident_listbox.itemconfig(idx, fg="#93C5FD")
            elif "RESOLVED" in badge:
                self.incident_listbox.itemconfig(idx, fg="#86EFAC")

        if not has_unresolved:
            self.alert_banner.configure(bg="#1E293B")
            self.lbl_banner_icon.configure(text="🟢", bg="#1E293B", fg="#10B981")
            self.lbl_banner_text.configure(
                text="ALL MONITORED RESIDENCES NORMAL — No Active Fall Emergencies",
                bg="#1E293B",
                fg="#10B981",
            )

        if auto_select_first and incidents:
            self.incident_listbox.selection_set(0)
            self._display_incident(incidents[0])
        elif self.selected_incident:
            # Refresh currently viewed incident
            updated = next((i for i in incidents if i["incidentId"] == self.selected_incident["incidentId"]), None)
            if updated:
                self._display_incident(updated)
            elif incidents:
                self.incident_listbox.selection_set(0)
                self._display_incident(incidents[0])
            else:
                self._clear_details_view()
        elif incidents:
            self.incident_listbox.selection_set(0)
            self._display_incident(incidents[0])
        else:
            self._clear_details_view()

    def _clear_details_view(self):
        self.selected_incident = None
        self._displayed_incident_id = None
        self._current_photo = None
        self.lbl_detail_title.configure(text="No active incidents in stream", fg="#94A3B8")
        self.lbl_photo.configure(image="", text="[No Incident Selected]", fg="#64748B")
        self.voice_box.configure(bg="#334155")
        self.lbl_voice_title.configure(text="🗣️ VERBAL RESPONSE CLASSIFICATION", fg="#94A3B8", bg="#334155")
        self.lbl_voice_transcript.configure(text="Waiting for incidents...", fg="#FFFFFF", bg="#334155")
        self.lbl_meta_time.configure(text="Detected: ---")
        self.lbl_meta_conf.configure(text="Confidence: ---")
        self.lbl_meta_status.configure(text="Status: NORMAL", fg="#10B981")
        self.btn_ack.configure(state=tk.DISABLED)
        self.btn_call.configure(state=tk.DISABLED)
        self.btn_resolve.configure(state=tk.DISABLED)

    def _on_incident_selected(self, event):
        sel = self.incident_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < len(self.active_incidents):
            self._display_incident(self.active_incidents[idx])

    def _display_incident(self, inc: Dict[str, Any]):
        self.selected_incident = inc
        status = inc.get("incidentStatus", "UNKNOWN")
        loc = inc.get("locationLabel", "Room")
        detected = inc.get("detectedAt", "")
        conf = int(inc.get("confidence", 0.9) * 100)
        voice_status = inc.get("voiceResponseStatus", "NO_RESPONSE")
        transcript = inc.get("voiceTranscript", "")

        # 1. Title
        self.lbl_detail_title.configure(
            text=f"🚨 INCIDENT #{inc['incidentId']} — {loc.upper()}",
            fg="#EF4444" if status in ("EMERGENCY_PUBLISHED", "EMERGENCY_TRIGGERED") else "#38BDF8",
        )

        # 2. Voice Response Highlight
        if voice_status == "REQUEST_HELP":
            self.voice_box.configure(bg="#7F1D1D")
            self.lbl_voice_title.configure(text="🆘 USER VOCALLY CRIED FOR HELP! (IMMEDIATE ESCALATION)", fg="#FCA5A5", bg="#7F1D1D")
            self.lbl_voice_transcript.configure(text=f'"{transcript}"', fg="#FFFFFF", bg="#7F1D1D")
        elif voice_status == "CONFIRM_SAFE":
            self.voice_box.configure(bg="#14532D")
            self.lbl_voice_title.configure(text="✅ USER CONFIRMED SAFE", fg="#86EFAC", bg="#14532D")
            self.lbl_voice_transcript.configure(text=f'"{transcript}"', fg="#FFFFFF", bg="#14532D")
        else:
            self.voice_box.configure(bg="#334155")
            self.lbl_voice_title.configure(text="⚠️ NO RELIABLE VERBAL CONFIRMATION (ESCALATED)", fg="#FEF08A", bg="#334155")
            self.lbl_voice_transcript.configure(text=f'"{transcript or "[10.0s Timeout Expired without Reassurance]"}"', fg="#FFFFFF", bg="#334155")

        # 3. Metadata
        self.lbl_meta_time.configure(text=f"Detected: {detected}")
        self.lbl_meta_conf.configure(text=f"Confidence: {conf}%")
        self.lbl_meta_status.configure(
            text=f"Status: {status}",
            fg="#EF4444" if "EMERGENCY" in status else ("#38BDF8" if "ACK" in status else "#10B981"),
        )

        # 4. Evidentiary Photo Snapshot (Strictly 1 keyframe per incident, cached in memory)
        inc_id = inc.get("incidentId", "")
        if self._displayed_incident_id != inc_id:
            self._displayed_incident_id = inc_id
            evidence_b64 = inc.get("evidenceBase64", "")
            if evidence_b64:
                try:
                    img_data = base64.b64decode(evidence_b64)
                    pil_img = Image.open(io.BytesIO(img_data))
                    pil_img.thumbnail((500, 260), Image.Resampling.LANCZOS)
                    self._current_photo = ImageTk.PhotoImage(pil_img)
                    self.lbl_photo.configure(image=self._current_photo, text="")
                except Exception as e:
                    self.lbl_photo.configure(image="", text=f"[Photo Decode Error: {e}]", fg="#94A3B8")
            else:
                self.lbl_photo.configure(image="", text="[No Evidentiary Snapshot Available]", fg="#64748B")

        # 5. Button states
        is_active_emergency = status in ("EMERGENCY_PUBLISHED", "EMERGENCY_TRIGGERED")
        is_acknowledged = status == "ACKNOWLEDGED"

        self.btn_ack.configure(state=tk.NORMAL if is_active_emergency else tk.DISABLED)
        self.btn_call.configure(state=tk.NORMAL if (is_active_emergency or is_acknowledged) else tk.DISABLED)
        self.btn_resolve.configure(state=tk.NORMAL if (is_active_emergency or is_acknowledged) else tk.DISABLED)

    def _acknowledge_selected(self):
        if not self.selected_incident:
            return
        inc_id = self.selected_incident["incidentId"]

        def _worker():
            try:
                url = f"{self.backend_url}/incidents/{inc_id}/acknowledge"
                payload = {"actorName": CAREGIVER_NAME, "actorUid": CAREGIVER_UID}
                r = requests.post(url, json=payload, timeout=3.0)
                if r.status_code == 200:
                    self.root.after(0, lambda: messagebox.showinfo("Acknowledged", f"Incident #{inc_id} has been marked as Acknowledged by {CAREGIVER_NAME}."))
                    self.root.after(0, self._fetch_incidents)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Could not acknowledge incident: {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _resolve_selected_dialog(self):
        if not self.selected_incident:
            return
        inc_id = self.selected_incident["incidentId"]

        note = simpledialog.askstring(
            "Resolve Incident",
            f"Enter resolution note for Incident #{inc_id}:\n(e.g., 'Checked in person, Grandma is okay.')",
            parent=self.root,
        )
        if not note:
            return

        def _worker():
            try:
                url = f"{self.backend_url}/incidents/{inc_id}/resolve"
                payload = {"actorName": CAREGIVER_NAME, "actorUid": CAREGIVER_UID, "resolutionNote": note}
                r = requests.post(url, json=payload, timeout=3.0)
                if r.status_code == 200:
                    self.root.after(0, lambda: messagebox.showinfo("Resolved", f"Incident #{inc_id} has been marked as RESOLVED.\nNote: {note}"))
                    self.root.after(0, self._fetch_incidents)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Could not resolve incident: {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _clear_resolved_incidents(self):
        """Purges resolved incidents from backend and clears feed."""
        def _worker():
            try:
                r = requests.delete(f"{self.backend_url}/incidents/resolved", timeout=3.0)
                if r.status_code == 200:
                    purged = r.json().get("purgedCount", 0)
                    self.root.after(0, lambda: messagebox.showinfo("Cleaned", f"Purged {purged} resolved incident(s)."))
                    self.root.after(0, self._fetch_incidents)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Could not purge resolved incidents: {e}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _call_emergency_dialog(self):
        if not self.selected_incident:
            return
        inc = self.selected_incident
        loc = inc.get("locationLabel", "Residence")
        transcript = inc.get("voiceTranscript", "None")

        msg = (
            f"Initiating Emergency Dispatch Call\n\n"
            f"Target Location: {loc} (Demo Family Residence)\n"
            f"Incident: Confirmed Fall with Help Cry ('{transcript}')\n\n"
            f"Dialing 911 / 112...\n\n"
            f"In a real deployment, this triggers native cellular call / VoIP emergency dispatch."
        )
        messagebox.showwarning("Emergency Dispatch Initiated", msg)

    def _simulate_help_incident(self):
        """Simulates an urgent fall where the subject actively said 'Help me please!'."""
        def _worker():
            # Generate a synthetic evidentiary image (Dark canvas with alert text)
            im = Image.new("RGB", (320, 240), color=(30, 20, 25))
            buf = io.BytesIO()
            im.save(buf, format="JPEG")
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

            payload = {
                "incidentId": f"inc_demo_{int(time.time()) % 10000}",
                "familyId": "fam_demo_101",
                "deviceId": "dev_win_01",
                "deviceName": "Windows Living Room Monitor",
                "locationLabel": "Living Room",
                "lastKnownPosture": "LYING",
                "transitionType": "STANDING_TO_LYING",
                "confidence": 0.98,
                "voiceResponseStatus": "REQUEST_HELP",
                "voiceTranscript": "Help me! I fell down and cannot get up!",
                "incidentStatus": "EMERGENCY_PUBLISHED",
                "evidenceBase64": b64_img,
            }
            try:
                requests.post(f"{self.backend_url}/incidents", json=payload, timeout=3.0)
            except Exception as e:
                print(f"[FamilyApp] Simulation post error: {e}")

        threading.Thread(target=_worker, daemon=True).start()


def main():
    root = tk.Tk()
    app = SafeFallFamilyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
