"""
Unit Tests for Backend Synchronization, Device Pairing, and Security Policies.
"""

import os
import unittest
from src.backend.mock_backend import MockBackendService
from src.backend.cloud_client import CloudClient
from src.detection.state_machine import IncidentRecord, EmergencyState


class TestBackendSync(unittest.TestCase):
    def setUp(self):
        self.mock_backend = MockBackendService()
        self.queue_file = "test_queue.json"
        if os.path.exists(self.queue_file):
            os.remove(self.queue_file)
        self.client = CloudClient(api_base_url="http://invalid-offline-host:9999", queue_file_path=self.queue_file)

    def tearDown(self):
        if os.path.exists(self.queue_file):
            os.remove(self.queue_file)

    def test_invitation_code_redemption(self):
        """Scenario 20: Valid invitation code accepted, usage count incremented, expired/depleted rejected."""
        # Valid code
        res = self.mock_backend.redeem_invitation("SAFE-7711", "usr_caregiver_bob", "Bob")
        self.assertTrue(res["success"])
        self.assertEqual(res["familyId"], "fam_demo_101")

        # Exceed limit
        self.mock_backend.invitations["SAFE-7711"]["usageLimit"] = 1
        with self.assertRaises(ValueError):
            self.mock_backend.redeem_invitation("SAFE-7711", "usr_other", "Alice")

    def test_device_pairing(self):
        """Scenario 22: Device registered securely to family."""
        res = self.mock_backend.pair_device(
            pairing_code="SAFE-7711",
            device_name="Kitchen Camera",
            platform="windows",
            location_label="Kitchen",
        )
        self.assertEqual(res["status"], "paired")
        self.assertTrue(res["deviceId"].startswith("dev_windows_"))
        self.assertIn(res["deviceId"], self.mock_backend.devices)

    def test_offline_incident_enqueue(self):
        """Scenario 13 & 15: Network failure enqueues incident locally without crash."""
        inc = IncidentRecord(
            incident_id="inc_offline_test",
            family_id="fam_demo_101",
            device_id="dev_win_01",
            device_name="Living Room Monitor",
            location_label="Living Room",
            detected_at=100.0,
            last_known_posture="LYING",
            transition_type="STANDING_TO_LYING",
            confidence=0.91,
            state=EmergencyState.EMERGENCY_PUBLISHED,
        )

        success = self.client.publish_incident(inc)
        self.assertFalse(success, "Host is invalid, so publish should fail gracefully")
        self.assertTrue(os.path.exists(self.queue_file), "Incident must be stored in offline queue")


if __name__ == "__main__":
    unittest.main()
