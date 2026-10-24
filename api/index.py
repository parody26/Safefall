"""
SafeFall Cloud Backend — Vercel Serverless Function & REST API Gateway.
Provides serverless endpoints for the SafeFall Family Safety Dashboard
and Windows AI Monitoring Station.
Zero external dependencies (uses standard library http.server).
"""

import os
import sys
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List

# Temporary persistent store path for serverless lambdas
STORE_PATH = os.path.join("/tmp", "safefall_incidents.json")

# In-memory incident cache with seed device
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


def _load_persisted_state():
    global INCIDENTS
    try:
        if os.path.exists(STORE_PATH):
            with open(STORE_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    INCIDENTS.update(saved)
    except Exception:
        pass

    # Also load queued incidents if present
    try:
        q_paths = [
            os.path.join(os.path.dirname(__file__), "..", "safefall", "windows_monitor", "queued_incidents.json"),
            os.path.join(os.getcwd(), "safefall", "windows_monitor", "queued_incidents.json"),
        ]
        for qp in q_paths:
            if os.path.exists(qp):
                with open(qp, "r", encoding="utf-8") as f:
                    q_list = json.load(f)
                    if isinstance(q_list, list):
                        for inc in q_list:
                            if isinstance(inc, dict) and "incidentId" in inc:
                                if inc["incidentId"] not in INCIDENTS:
                                    INCIDENTS[inc["incidentId"]] = inc
                break
    except Exception:
        pass


def _save_persisted_state():
    try:
        with open(STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(INCIDENTS, f)
    except Exception:
        pass


_load_persisted_state()


class handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        try:
            sys.stderr.write(f"[SafeFall Vercel] {self.command} {self.path} - {args[0]}\n")
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

    def _normalize_path(self) -> str:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if "__path" in query:
            return query["__path"][0].rstrip("/")

        for h in ("x-matched-path", "x-vercel-matched-path", "x-forwarded-uri"):
            val = self.headers.get(h)
            if val and val != "/api/index.py":
                return urlparse(val).path.rstrip("/")

        path = parsed.path.rstrip("/")
        if not path or path == "/api/index.py":
            return "/"
        return path

    def do_GET(self):
        _load_persisted_state()
        path = self._normalize_path()
        query = parse_qs(urlparse(self.path).query)

        # 1. Health check
        if path in ("", "/", "/health", "/api/health", "/api/v1/health", "/api"):
            self._send_json(200, {
                "status": "healthy",
                "service": "SafeFall Cloud Gateway (Vercel Serverless)",
                "timestamp": time.time(),
                "activeIncidents": len(INCIDENTS),
            })
            return

        # 2. List all incidents
        if path in ("/api/v1/incidents", "/api/incidents"):
            incidents_list = list(INCIDENTS.values())
            incidents_list.sort(key=lambda x: x.get("detectedAtTs", 0), reverse=True)
            self._send_json(200, {"incidents": incidents_list, "count": len(incidents_list)})
            return

        # 3. Get single incident
        if "/incidents/" in path and not path.endswith(("/acknowledge", "/resolve")):
            inc_id = path.split("/")[-1]
            inc = INCIDENTS.get(inc_id)
            if inc:
                self._send_json(200, inc)
            else:
                self._send_json(404, {"error": "Incident not found", "incidentId": inc_id})
            return

        # 4. List devices
        if path in ("/api/v1/devices", "/api/devices"):
            devs = list(DEVICES.values())
            self._send_json(200, {"devices": devs, "count": len(devs)})
            return

        # 5. List family members
        if path in ("/api/v1/family", "/api/family", "/api/v1/family/members", "/api/family/members"):
            self._send_json(200, {
                "familyId": "fam_demo_101",
                "familyName": "Demo Residence - SafeFall",
                "members": FAMILY_MEMBERS,
            })
            return

        # 6. Notifications poll
        if path in ("/api/v1/notifications/poll", "/api/notifications/poll"):
            since = float(query.get("since", [0])[0])
            recent = [i for i in INCIDENTS.values() if i.get("updatedAtTs", i.get("detectedAtTs", 0)) > since]
            self._send_json(200, {"events": recent, "timestamp": time.time()})
            return

        self._send_json(404, {"error": "Endpoint Not Found", "path": path})

    def do_POST(self):
        _load_persisted_state()
        path = self._normalize_path()

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            req_data = json.loads(body.decode("utf-8"))
        except Exception:
            req_data = {}

        # 1. Publish Incident
        if path in ("/api/v1/incidents", "/api/incidents"):
            inc_id = req_data.get("incidentId") or f"inc_{uuid.uuid4().hex[:10]}"
            now_ts = time.time()
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts))

            existing = INCIDENTS.get(inc_id)
            if existing:
                existing["updatedAtTs"] = now_ts
                if "voiceResponseStatus" in req_data:
                    existing["voiceResponseStatus"] = req_data["voiceResponseStatus"]
                if "voiceTranscript" in req_data:
                    existing["voiceTranscript"] = req_data["voiceTranscript"]
                if "incidentStatus" in req_data:
                    existing["incidentStatus"] = req_data["incidentStatus"]
                if not existing.get("evidenceBase64") and req_data.get("evidenceBase64"):
                    existing["evidenceBase64"] = req_data["evidenceBase64"]
                incident_record = existing
                status_code = 200
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

            _save_persisted_state()
            self._send_json(status_code, {
                "success": True,
                "incidentId": inc_id,
                "status": incident_record["incidentStatus"],
                "notifiedFamilyCount": len(FAMILY_MEMBERS),
            })
            return

        # 2. Acknowledge Incident
        if path.endswith("/acknowledge") and "/incidents/" in path:
            parts = path.split("/")
            inc_id = parts[parts.index("incidents") + 1]
            actor_name = req_data.get("actorName", "Family Caregiver")

            inc = INCIDENTS.get(inc_id)
            if not inc:
                self._send_json(404, {"error": "Incident not found", "incidentId": inc_id})
                return

            inc["incidentStatus"] = "ACKNOWLEDGED"
            inc["acknowledgedBy"] = actor_name
            inc["acknowledgedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            inc["updatedAtTs"] = time.time()
            _save_persisted_state()

            self._send_json(200, {
                "success": True,
                "incidentId": inc_id,
                "status": "ACKNOWLEDGED",
                "acknowledgedBy": actor_name,
            })
            return

        # 3. Resolve Incident
        if path.endswith("/resolve") and "/incidents/" in path:
            parts = path.split("/")
            inc_id = parts[parts.index("incidents") + 1]
            actor_name = req_data.get("actorName", "Family Admin")
            note = req_data.get("resolutionNote", "Checked in person, confirmed safe.")

            inc = INCIDENTS.get(inc_id)
            if not inc:
                self._send_json(404, {"error": "Incident not found", "incidentId": inc_id})
                return

            inc["incidentStatus"] = "RESOLVED"
            inc["resolvedBy"] = actor_name
            inc["resolvedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            inc["resolutionNote"] = note
            inc["updatedAtTs"] = time.time()
            _save_persisted_state()

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
            dev_id = parts[parts.index("devices") + 1]
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
        if path in ("/api/v1/devices/pair", "/api/devices/pair"):
            dev_id = req_data.get("deviceId") or f"dev_{uuid.uuid4().hex[:6]}"
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
        _load_persisted_state()
        path = self._normalize_path()

        if path in ("/api/v1/incidents/resolved", "/api/incidents/resolved"):
            to_remove = [k for k, v in INCIDENTS.items() if v.get("incidentStatus") == "RESOLVED"]
            for k in to_remove:
                del INCIDENTS[k]
            _save_persisted_state()
            self._send_json(200, {"success": True, "purgedCount": len(to_remove)})
            return

        if path in ("/api/v1/incidents", "/api/incidents"):
            count = len(INCIDENTS)
            INCIDENTS.clear()
            _save_persisted_state()
            self._send_json(200, {"success": True, "purgedCount": count})
            return

        self._send_json(404, {"error": "Not Found", "path": path})
