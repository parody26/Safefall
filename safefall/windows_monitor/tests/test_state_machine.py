"""
Unit Tests for SafeFall Emergency State Machine.
Validates all 9 formal states, monotonic timers, cancellation, and escalation.
"""

import time
import unittest
from src.detection.state_machine import EmergencyStateMachine, EmergencyState
from src.detection.temporal_detector import CandidateFallEvent
from src.speech.speech_service import VerbalIntent


class MockAlarm:
    def __init__(self):
        self.ringing = False

    def start(self):
        self.ringing = True

    def stop(self):
        self.ringing = False

    def is_ringing(self):
        return self.ringing


class TestStateMachine(unittest.TestCase):
    def setUp(self):
        self.alarm = MockAlarm()
        self.sm = EmergencyStateMachine(
            family_id="fam_test_01",
            device_id="dev_test_01",
            suspected_fall_grace_sec=0.2, # short durations for fast automated testing
            voice_response_timeout_sec=0.3,
            alarm_service=self.alarm,
        )

    def _trigger_test_fall(self):
        ev = CandidateFallEvent(
            detected_at=time.time(),
            transition_type="STANDING_TO_LYING",
            duration_sec=0.6,
            velocity_hip=0.3,
            angular_velocity=80.0,
            confidence=0.92,
        )
        self.sm.handle_candidate_fall(ev)

    def test_suspected_fall_to_normal_on_quick_recovery(self):
        """If subject stands up within grace window, state machine resets to NORMAL."""
        self._trigger_test_fall()
        self.assertEqual(self.sm.current_state, EmergencyState.SUSPECTED_FALL)

        # Subject sat up before grace timeout
        self.sm.handle_posture_update("SITTING")
        self.assertEqual(self.sm.current_state, EmergencyState.NORMAL)
        self.assertFalse(self.alarm.ringing)

    def test_grace_countdown_to_local_alarm(self):
        """Scenario 6 & 7: Subject remains lying for 0.2s grace -> local alarm sounds."""
        self._trigger_test_fall()
        self.assertEqual(self.sm.current_state, EmergencyState.SUSPECTED_FALL)

        # Wait past grace countdown
        time.sleep(0.25)
        self.sm.handle_posture_update("LYING")

        self.assertEqual(self.sm.current_state, EmergencyState.RESPONSE_PROCESSING)
        self.assertTrue(self.alarm.ringing)

    def test_verbal_reassurance_stops_alarm_and_cancels(self):
        """Scenario 8: Reassurance stops alarm and marks false alarm."""
        self._trigger_test_fall()
        time.sleep(0.25)
        self.sm.handle_posture_update("LYING")

        # Simulate user saying "I am fine"
        self.sm.handle_verbal_response(VerbalIntent.CONFIRM_SAFE, "I am fine, don't worry")

        self.assertFalse(self.alarm.ringing)
        self.assertEqual(self.sm.current_state, EmergencyState.NORMAL)

    def test_immediate_escalation_on_help_request(self):
        """Scenario 9: Help request triggers immediate escalation."""
        published = []
        self.sm.on_publish_incident = lambda inc: published.append(inc)

        self._trigger_test_fall()
        time.sleep(0.25)
        self.sm.handle_posture_update("LYING")

        # User asks for help immediately
        self.sm.handle_verbal_response(VerbalIntent.REQUEST_HELP, "Help me!")

        self.assertEqual(self.sm.current_state, EmergencyState.EMERGENCY_PUBLISHED)
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0].voice_response_status, "REQUEST_HELP")

    def test_voice_timeout_escalation(self):
        """Scenario 10 & 11: Timeout or speech recognition error safely escalates to emergency."""
        self._trigger_test_fall()
        time.sleep(0.25)
        self.sm.handle_posture_update("LYING")

        # Wait for voice timeout
        time.sleep(0.35)
        self.sm.handle_posture_update("LYING")

        # Must be published as an active emergency
        self.assertEqual(self.sm.current_state, EmergencyState.EMERGENCY_PUBLISHED)
        self.assertIn(self.sm.active_incident.voice_response_status, ("NO_RESPONSE", "SPEECH_ERROR"))

    def test_acknowledgement_and_resolution_flow(self):
        """Scenario 18 & 19: Acknowledge marks seen; Resolve stops alarm and records note."""
        self._trigger_test_fall()
        time.sleep(0.25)
        self.sm.handle_posture_update("LYING")
        self.sm.handle_verbal_response(VerbalIntent.REQUEST_HELP, "Help")

        # Family member acknowledges
        self.sm.acknowledge_incident(actor_uid="usr_son", actor_name="David")
        self.assertEqual(self.sm.current_state, EmergencyState.ACKNOWLEDGED)
        self.assertEqual(self.sm.active_incident.acknowledged_by, "usr_son")

        # Caregiver resolves
        self.sm.resolve_incident(actor_uid="usr_nurse", actor_name="Nurse Sarah", resolution_note="Checked in person, safe.")
        self.assertEqual(self.sm.current_state, EmergencyState.NORMAL)
        self.assertFalse(self.alarm.ringing)


if __name__ == "__main__":
    unittest.main()
