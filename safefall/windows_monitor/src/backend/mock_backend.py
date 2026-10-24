"""
SafeFall Mock Backend & REST Test Server.
Provides a self-contained in-memory and JSON-backed server that implements
the exact SafeFall API contract for offline demonstrations and automated tests.
"""

import time
import uuid
from typing import Dict, Any, List, Optional


class MockBackendService:
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {
            "usr_demo_admin": {
                "uid": "usr_demo_admin",
                "displayName": "Sarah Jenkins",
                "email": "sarah@example.com",
                "accountStatus": "active",
            }
        }
        self.families: Dict[str, Dict[str, Any]] = {
            "fam_demo_101": {
                "familyId": "fam_demo_101",
                "name": "Jenkins Family Residence",
                "description": "SafeFall Home Monitoring for Grandma Alice",
                "createdBy": "usr_demo_admin",
                "createdAt": "2026-10-23T18:00:00Z",
                "settings": {"emergencyDialNumber": "112"},
            }
        }
        self.memberships: Dict[str, Dict[str, Any]] = {
            "fam_demo_101_usr_demo_admin": {
                "membershipId": "fam_demo_101_usr_demo_admin",
                "familyId": "fam_demo_101",
                "uid": "usr_demo_admin",
                "role": "admin",
                "membershipStatus": "active",
                "notificationPreferences": {"pushEnabled": True, "soundEnabled": True},
            }
        }
        self.invitations: Dict[str, Dict[str, Any]] = {
            "SAFE-7711": {
                "invitationId": "inv_001",
                "familyId": "fam_demo_101",
                "familyName": "Jenkins Family Residence",
                "inviteCode": "SAFE-7711",
                "createdBy": "usr_demo_admin",
                "role": "caregiver",
                "expiresAt": "2026-10-28T18:00:00Z",
                "usageLimit": 5,
                "usageCount": 0,
                "revokedAt": None,
            }
        }
        self.devices: Dict[str, Dict[str, Any]] = {
            "dev_win_01": {
                "deviceId": "dev_win_01",
                "familyId": "fam_demo_101",
                "deviceName": "Living Room Monitor",
                "platform": "windows",
                "locationLabel": "Living Room (Main Area)",
                "lastSeenAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "monitoringStatus": "monitoring_active",
                "registrationStatus": "paired",
            }
        }
        self.incidents: Dict[str, Dict[str, Any]] = {}
        self.incident_events: List[Dict[str, Any]] = []

    def pair_device(self, pairing_code: str, device_name: str, platform: str, location_label: str) -> Dict[str, Any]:
        """Pairing a new monitoring device using family pairing code."""
        # Validate code (we accept SAFE-xxxx or pairing code)
        family_id = "fam_demo_101"
        device_id = f"dev_{platform}_{uuid.uuid4().hex[:6]}"
        record = {
            "deviceId": device_id,
            "familyId": family_id,
            "deviceName": device_name,
            "platform": platform,
            "locationLabel": location_label,
            "lastSeenAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "monitoringStatus": "paired",
            "registrationStatus": "paired",
        }
        self.devices[device_id] = record
        return {"deviceId": device_id, "apiKey": f"key_{device_id}", "status": "paired", "familyId": family_id}

    def publish_incident(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Publishes an incident and dispatches family notifications."""
        inc_id = payload.get("incidentId") or f"inc_{uuid.uuid4().hex[:8]}"
        payload["incidentId"] = inc_id
        if "incidentStatus" not in payload:
            payload["incidentStatus"] = "EMERGENCY_PUBLISHED"
        self.incidents[inc_id] = payload

        # Audit event
        self.incident_events.append({
            "eventId": f"evt_{uuid.uuid4().hex[:8]}",
            "incidentId": inc_id,
            "eventType": "EMERGENCY_PUBLISHED",
            "timestamp": time.time(),
            "metadata": {"source": payload.get("deviceName")},
        })
        return {"incidentId": inc_id, "status": "published", "notifiedMembers": 1}

    def acknowledge_incident(self, incident_id: str, actor_uid: str, actor_name: str) -> Dict[str, Any]:
        """Acknowledges an incident."""
        if incident_id not in self.incidents:
            raise KeyError(f"Incident {incident_id} not found")
        inc = self.incidents[incident_id]
        inc["incidentStatus"] = "ACKNOWLEDGED"
        inc["acknowledgedBy"] = actor_uid
        inc["acknowledgedByName"] = actor_name
        inc["acknowledgedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.incident_events.append({
            "eventId": f"evt_{uuid.uuid4().hex[:8]}",
            "incidentId": incident_id,
            "eventType": "ACKNOWLEDGED",
            "actorUid": actor_uid,
            "timestamp": time.time(),
            "metadata": {"actorName": actor_name},
        })
        return inc

    def resolve_incident(self, incident_id: str, actor_uid: str, actor_name: str, note: str = "") -> Dict[str, Any]:
        """Resolves an incident."""
        if incident_id not in self.incidents:
            raise KeyError(f"Incident {incident_id} not found")
        inc = self.incidents[incident_id]
        inc["incidentStatus"] = "RESOLVED"
        inc["resolvedBy"] = actor_uid
        inc["resolvedByName"] = actor_name
        inc["resolvedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        inc["resolutionNote"] = note
        self.incident_events.append({
            "eventId": f"evt_{uuid.uuid4().hex[:8]}",
            "incidentId": incident_id,
            "eventType": "RESOLVED",
            "actorUid": actor_uid,
            "timestamp": time.time(),
            "metadata": {"actorName": actor_name, "note": note},
        })
        return inc

    def redeem_invitation(self, invite_code: str, uid: str, display_name: str) -> Dict[str, Any]:
        """Validates and redeems an invitation code."""
        code = invite_code.strip().upper()
        if code not in self.invitations:
            raise ValueError("Invalid invitation code")
        inv = self.invitations[code]
        if inv.get("usageCount", 0) >= inv.get("usageLimit", 1):
            raise ValueError("Invitation limit reached")
        inv["usageCount"] = inv.get("usageCount", 0) + 1
        membership_id = f"{inv['familyId']}_{uid}"
        self.memberships[membership_id] = {
            "membershipId": membership_id,
            "familyId": inv["familyId"],
            "uid": uid,
            "role": inv.get("role", "caregiver"),
            "membershipStatus": "active",
            "notificationPreferences": {"pushEnabled": True, "soundEnabled": True},
        }
        return {"success": True, "familyId": inv["familyId"], "role": inv["role"]}
