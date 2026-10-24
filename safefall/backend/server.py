"""
SafeFall Shared Unified Backend Server.
Provides real-time REST and push notification streaming for both the
Windows Monitoring Station and the Family Safety Applications.
Zero external dependencies (uses standard library http.server).
"""

import os
import sys
import json
import time
import uuid
import queue
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List

# Ensure Windows console encoding compatibility
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# In-memory storage with thread safety
DB_LOCK = threading.Lock()
INCIDENTS: Dict[str, Dict[str, Any]] = {}
DEVICES: Dict[str, Dict[str, Any]] = {
    "dev_win_01": {
        "deviceId": "dev_win_01",
        "deviceName": "Windows Living Room Monitor",
        "platform": "windows",
        "locationLabel": "Living Room",
        "status": "online",
        "fps": 24.0,
        "lastHeartbeat": time.time(),
    }
}
FAMILY_MEMBERS: List[Dict[str, Any]] = [
    {"uid": "usr_sarah", "name": "Sarah (Daughter)", "role": "caregiver", "phone": "+1-555-0192", "isOnline": True},
    {"uid": "usr_david", "name": "David (Son)", "role": "caregiver", "phone": "+1-555-0193", "isOnline": True},
    {"uid": "usr_nurse", "name": "Nurse Evelyn", "role": "medical_admin", "phone": "+1-555-0199", "isOnline": True},
]

# Subscribers for real-time push notifications (Family Apps)
NOTIFICATION_SUBSCRIBERS: List[queue.Queue] = []
SUBSCRIBERS_LOCK = threading.Lock()


def broadcast_notification(event_type: str, data: Dict[str, Any]):
    """Pushes notification payload to all active Family App long-poll/SSE listeners."""
    msg = {
        "eventType": event_type,
        "timestamp": time.time(),
        "data": data,
    }
    with SUBSCRIBERS_LOCK:
        dead = []
        for q in NOTIFICATION_SUBSCRIBERS:
            try:
                q.put_nowait(msg)
            except Exception:
                dead.append(q)
        for d in dead:
            NOTIFICATION_SUBSCRIBERS.remove(d)


class SafeFallRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean logging
        try:
            sys.stderr.write(f"[SafeFall Backend] {self.command} {self.path} - {args[0]}\n")
        except Exception:
            pass

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, status_code: int, data: Any):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        # 1. Family Safety Web Dashboard & Health Check
        if path in ("", "/index.html"):
            html_paths = [
                os.path.join(os.path.dirname(__file__), "public", "index.html"),
                os.path.join(os.path.dirname(__file__), "..", "..", "public", "index.html"),
                os.path.join(os.getcwd(), "public", "index.html"),
            ]
            for hp in html_paths:
                if os.path.exists(hp):
                    try:
                        with open(hp, "rb") as f:
                            content = f.read()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(content)))
                        self._send_cors_headers()
                        self.end_headers()
                        self.wfile.write(content)
                        return
                    except Exception:
                        pass
            self._send_json(200, {"status": "healthy", "service": "SafeFall Cloud Backend", "timestamp": time.time()})
            return

        if path in ("/health", "/api/v1/health"):
            self._send_json(200, {"status": "healthy", "service": "SafeFall Cloud Backend", "timestamp": time.time()})
            return

        # 2. List all incidents
        if path == "/api/v1/incidents":
            with DB_LOCK:
                incidents_list = list(INCIDENTS.values())
                incidents_list.sort(key=lambda x: x.get("detectedAtTs", 0), reverse=True)
            self._send_json(200, {"incidents": incidents_list, "count": len(incidents_list)})
            return

        # 3. Get single incident
        if path.startswith("/api/v1/incidents/"):
            inc_id = path.split("/")[-1]
            with DB_LOCK:
                inc = INCIDENTS.get(inc_id)
            if inc:
                self._send_json(200, inc)
            else:
                self._send_json(404, {"error": "Incident not found", "incidentId": inc_id})
            return

        # 4. List devices & telemetry
        if path == "/api/v1/devices":
            with DB_LOCK:
                devs = list(DEVICES.values())
            self._send_json(200, {"devices": devs, "count": len(devs)})
            return

        # 5. List family info
        if path in ("/api/v1/family", "/api/v1/family/members"):
            self._send_json(200, {
                "familyId": "fam_demo_101",
                "familyName": "Demo Residence - SafeFall",
                "members": FAMILY_MEMBERS,
            })
            return

        # 6. Real-time Notifications SSE Stream
        if path == "/api/v1/notifications/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._send_cors_headers()
            self.end_headers()

            q = queue.Queue(maxsize=50)
            with SUBSCRIBERS_LOCK:
                NOTIFICATION_SUBSCRIBERS.append(q)

            try:
                self.wfile.write(b"data: {\"event\":\"CONNECTED\",\"message\":\"SafeFall Family Alert Stream Active\"}\n\n")
                self.wfile.flush()

                while True:
                    try:
                        msg = q.get(timeout=20.0)
                        data_str = f"data: {json.dumps(msg)}\n\n"
                        self.wfile.write(data_str.encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
            except Exception:
                pass
            finally:
                with SUBSCRIBERS_LOCK:
                    if q in NOTIFICATION_SUBSCRIBERS:
                        NOTIFICATION_SUBSCRIBERS.remove(q)
            return

        # 7. Long-polling notifications
        if path == "/api/v1/notifications/poll":
            since = float(query.get("since", [0])[0])
            timeout = min(15.0, float(query.get("timeout", [10.0])[0]))

            deadline = time.time() + timeout
            while time.time() < deadline:
                with DB_LOCK:
                    recent = [i for i in INCIDENTS.values() if i.get("updatedAtTs", i.get("detectedAtTs", 0)) > since]
                if recent:
                    self._send_json(200, {"events": recent, "timestamp": time.time()})
                    return
                time.sleep(0.5)

            self._send_json(200, {"events": [], "timestamp": time.time()})
            return

        self._send_json(404, {"error": "Not Found", "path": path})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            req_data = json.loads(body.decode("utf-8"))
        except Exception:
            req_data = {}

        # 1. Publish Incident (from Monitoring Station)
        if path == "/api/v1/incidents":
            inc_id = req_data.get("incidentId") or f"inc_{uuid.uuid4().hex[:10]}"
            now_ts = time.time()
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts))

            with DB_LOCK:
                existing = INCIDENTS.get(inc_id)
                if existing:
                    # Incident already exists: update status/voice without duplicating keyframe
                    existing["updatedAtTs"] = now_ts
                    if "voiceResponseStatus" in req_data:
                        existing["voiceResponseStatus"] = req_data["voiceResponseStatus"]
                    if "voiceTranscript" in req_data:
                        existing["voiceTranscript"] = req_data["voiceTranscript"]
                    if "incidentStatus" in req_data:
                        existing["incidentStatus"] = req_data["incidentStatus"]
                    # Retain original evidentiary keyframe strictly (1 image per incident)
                    if not existing.get("evidenceBase64") and req_data.get("evidenceBase64"):
                        existing["evidenceBase64"] = req_data["evidenceBase64"]
                    incident_record = existing
                    status_code = 200
                    event_type = "INCIDENT_UPDATED"
                else:
                    incident_record = {
                        "incidentId": inc_id,
                        "familyId": req_data.get("familyId", "fam_demo_101"),
                        "deviceId": req_data.get("deviceId", "dev_win_01"),
                        "deviceName": req_data.get("deviceName", "Windows Living Room Monitor"),
                        "locationLabel": req_data.get("locationLabel", "Living Room"),
                        "detectedAt": req_data.get("detectedAt", now_iso),
                        "detectedAtTs": now_ts,
                        "updatedAtTs": now_ts,
                        "lastKnownPosture": req_data.get("lastKnownPosture", "LYING"),
                        "transitionType": req_data.get("transitionType", "STANDING_TO_LYING"),
                        "confidence": req_data.get("confidence", 0.95),
                        "voiceResponseStatus": req_data.get("voiceResponseStatus", "NO_RESPONSE"),
                        "voiceTranscript": req_data.get("voiceTranscript", ""),
                        "incidentStatus": req_data.get("incidentStatus", "EMERGENCY_PUBLISHED"),
                        "evidenceBase64": req_data.get("evidenceBase64", ""),
                        "acknowledgedBy": None,
                        "acknowledgedAt": None,
                        "resolvedBy": None,
                        "resolvedAt": None,
                        "resolutionNote": None,
                    }
                    INCIDENTS[inc_id] = incident_record
                    status_code = 201
                    event_type = "INCIDENT_PUBLISHED"

            print(f"\n[SafeFall Backend] [ALERT] EMERGENCY INCIDENT RECORDED: {inc_id}")
            print(f"  -> Location: {incident_record['locationLabel']}")
            print(f"  -> Voice Response: '{incident_record['voiceTranscript']}' ({incident_record['voiceResponseStatus']})")
            print(f"  -> Single Keyframe Image: {'PRESENT (1 Keyframe)' if incident_record.get('evidenceBase64') else 'NONE'}")
            print(f"  -> Broadcasting push alert to Family Safety Applications...")

            # Broadcast instant notification to all connected Family Apps!
            broadcast_notification(event_type, incident_record)

            self._send_json(status_code, {
                "success": True,
                "incidentId": inc_id,
                "status": incident_record["incidentStatus"],
                "notifiedFamilyCount": len(FAMILY_MEMBERS),
            })
            return

        # 2. Acknowledge Incident (from Family App)
        if path.endswith("/acknowledge") and "/api/v1/incidents/" in path:
            inc_id = path.split("/")[-2]
            actor_name = req_data.get("actorName", "Family Caregiver")

            with DB_LOCK:
                inc = INCIDENTS.get(inc_id)
                if not inc:
                    self._send_json(404, {"error": "Incident not found", "incidentId": inc_id})
                    return

                inc["incidentStatus"] = "ACKNOWLEDGED"
                inc["acknowledgedBy"] = actor_name
                inc["acknowledgedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                inc["updatedAtTs"] = time.time()

            print(f"[SafeFall Backend] [ACK] Incident {inc_id} ACKNOWLEDGED by {actor_name}")
            broadcast_notification("INCIDENT_ACKNOWLEDGED", inc)

            self._send_json(200, {
                "success": True,
                "incidentId": inc_id,
                "status": "ACKNOWLEDGED",
                "acknowledgedBy": actor_name,
            })
            return

        # 3. Resolve Incident (from Family App)
        if path.endswith("/resolve") and "/api/v1/incidents/" in path:
            inc_id = path.split("/")[-2]
            actor_name = req_data.get("actorName", "Family Admin")
            note = req_data.get("resolutionNote", "Checked in person, confirmed safe.")

            with DB_LOCK:
                inc = INCIDENTS.get(inc_id)
                if not inc:
                    self._send_json(404, {"error": "Incident not found", "incidentId": inc_id})
                    return

                inc["incidentStatus"] = "RESOLVED"
                inc["resolvedBy"] = actor_name
                inc["resolvedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                inc["resolutionNote"] = note
                inc["updatedAtTs"] = time.time()

            print(f"[SafeFall Backend] [RESOLVE] Incident {inc_id} RESOLVED by {actor_name}: '{note}'")
            broadcast_notification("INCIDENT_RESOLVED", inc)

            self._send_json(200, {
                "success": True,
                "incidentId": inc_id,
                "status": "RESOLVED",
                "resolvedBy": actor_name,
                "note": note,
            })
            return

        # 4. Device Heartbeat
        if "/devices/" in path and path.endswith("/heartbeat"):
            parts = path.split("/")
            dev_id = parts[-2]
            with DB_LOCK:
                if dev_id not in DEVICES:
                    DEVICES[dev_id] = {
                        "deviceId": dev_id,
                        "deviceName": req_data.get("deviceName", dev_id),
                        "platform": req_data.get("platform", "windows"),
                        "locationLabel": req_data.get("locationLabel", "Home"),
                    }
                DEVICES[dev_id]["status"] = req_data.get("monitoringStatus", "online")
                DEVICES[dev_id]["lastHeartbeat"] = time.time()

            self._send_json(200, {"success": True, "deviceId": dev_id, "status": "acknowledged"})
            return

        # 5. Device Pairing
        if path == "/api/v1/devices/pair":
            dev_id = req_data.get("deviceId") or f"dev_{uuid.uuid4().hex[:6]}"
            with DB_LOCK:
                DEVICES[dev_id] = {
                    "deviceId": dev_id,
                    "deviceName": req_data.get("deviceName", "New Monitor"),
                    "platform": req_data.get("platform", "generic"),
                    "locationLabel": req_data.get("locationLabel", "Room"),
                    "status": "paired",
                    "lastHeartbeat": time.time(),
                }
            self._send_json(200, {
                "success": True,
                "deviceId": dev_id,
                "apiKey": f"key_{uuid.uuid4().hex}",
                "status": "paired",
            })
            return

        self._send_json(404, {"error": "Endpoint Not Found", "path": path})

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/v1/incidents/resolved":
            with DB_LOCK:
                to_remove = [k for k, v in INCIDENTS.items() if v.get("incidentStatus") == "RESOLVED"]
                for k in to_remove:
                    del INCIDENTS[k]
            print(f"[SafeFall Backend] Purged {len(to_remove)} resolved incidents.")
            self._send_json(200, {"success": True, "purgedCount": len(to_remove)})
            return

        if path == "/api/v1/incidents":
            with DB_LOCK:
                count = len(INCIDENTS)
                INCIDENTS.clear()
            print(f"[SafeFall Backend] Purged all {count} incidents.")
            self._send_json(200, {"success": True, "purgedCount": count})
            return

        self._send_json(404, {"error": "Not Found", "path": path})


def start_server(host: str = "0.0.0.0", port: int = 8080):
    server = HTTPServer((host, port), SafeFallRequestHandler)
    print("=" * 65)
    print(f"  [SafeFall] Shared Backend API Gateway & Notification Hub")
    print(f"  Listening on: http://localhost:{port}")
    print(f"  Endpoints:    /api/v1/incidents | /api/v1/notifications/stream")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SafeFall Backend] Server shutting down.")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    start_server(port=port)
