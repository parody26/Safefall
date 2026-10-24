"""
SafeFall Backend & Cloud Client.
Handles real-time incident publication, device heartbeats, evidence vaulting,
and resilient offline store-and-forward queueing.
"""

import os
import json
import time
import base64
import requests
from typing import Optional, Dict, Any
from ..detection.state_machine import IncidentRecord


class CloudClient:
    def __init__(
        self,
        api_base_url: str = "http://localhost:8080/api/v1",
        device_api_key: Optional[str] = None,
        queue_file_path: str = "queued_incidents.json",
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.device_api_key = device_api_key or "dev_token_default"
        self.queue_file_path = queue_file_path
        self._is_online = True

    def publish_incident(self, incident: IncidentRecord) -> bool:
        """
        Publishes incident to cloud backend. If offline or upload fails,
        durably queues to disk without interrupting the local alarm or emergency flow.
        """
        evidence_b64 = ""
        if incident.evidence_image_bytes:
            evidence_b64 = base64.b64encode(incident.evidence_image_bytes).decode("utf-8")

        payload = {
            "incidentId": incident.incident_id,
            "familyId": incident.family_id,
            "deviceId": incident.device_id,
            "deviceName": incident.device_name,
            "locationLabel": incident.location_label,
            "detectedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(incident.detected_at)),
            "lastKnownPosture": incident.last_known_posture,
            "transitionType": incident.transition_type,
            "confidence": incident.confidence,
            "voiceResponseStatus": incident.voice_response_status,
            "voiceTranscript": incident.voice_transcript,
            "incidentStatus": incident.state.value,
            "evidenceBase64": evidence_b64,
        }

        try:
            url = f"{self.api_base_url}/incidents"
            headers = {"Authorization": f"Bearer {self.device_api_key}", "Content-Type": "application/json"}
            resp = requests.post(url, json=payload, headers=headers, timeout=5.0)
            if resp.status_code in (200, 201):
                self._is_online = True
                self._flush_queue()
                return True
            else:
                self._is_online = False
                self._enqueue_offline(payload)
                return False
        except Exception:
            self._is_online = False
            self._enqueue_offline(payload)
            return False

    def send_heartbeat(self, device_id: str, family_id: str, status: str = "monitoring_active") -> bool:
        """Sends device presence & monitoring state to backend."""
        try:
            url = f"{self.api_base_url}/devices/{device_id}/heartbeat"
            headers = {"Authorization": f"Bearer {self.device_api_key}", "Content-Type": "application/json"}
            payload = {"familyId": family_id, "monitoringStatus": status, "timestamp": time.time()}
            resp = requests.post(url, json=payload, headers=headers, timeout=3.0)
            self._is_online = (resp.status_code == 200)
            return self._is_online
        except Exception:
            self._is_online = False
            return False

    def is_online(self) -> bool:
        return self._is_online

    def get_incident_status(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Queries backend for incident lifecycle updates (e.g. Acknowledged or Resolved by family)."""
        try:
            url = f"{self.api_base_url}/incidents/{incident_id}"
            headers = {"Authorization": f"Bearer {self.device_api_key}"}
            resp = requests.get(url, headers=headers, timeout=2.5)
            if resp.status_code == 200:
                self._is_online = True
                return resp.json()
        except Exception:
            pass
        return None

    def _enqueue_offline(self, payload: Dict[str, Any]):
        """Persists failed upload payloads locally until connectivity is restored."""
        try:
            queue = []
            if os.path.exists(self.queue_file_path):
                with open(self.queue_file_path, "r", encoding="utf-8") as f:
                    queue = json.load(f)
            queue.append(payload)
            with open(self.queue_file_path, "w", encoding="utf-8") as f:
                json.dump(queue, f, indent=2)
        except Exception as e:
            print(f"[CloudClient] Offline queue error: {e}")

    def _flush_queue(self):
        """Attempts to sync previously queued incidents upon network restoration."""
        if not os.path.exists(self.queue_file_path):
            return
        try:
            with open(self.queue_file_path, "r", encoding="utf-8") as f:
                queue = json.load(f)
            if not queue:
                return

            remaining = []
            url = f"{self.api_base_url}/incidents"
            headers = {"Authorization": f"Bearer {self.device_api_key}", "Content-Type": "application/json"}

            for item in queue:
                try:
                    resp = requests.post(url, json=item, headers=headers, timeout=4.0)
                    if resp.status_code not in (200, 201):
                        remaining.append(item)
                except Exception:
                    remaining.append(item)

            with open(self.queue_file_path, "w", encoding="utf-8") as f:
                json.dump(remaining, f, indent=2)
        except Exception:
            pass
