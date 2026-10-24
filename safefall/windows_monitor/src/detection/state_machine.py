"""
SafeFall Emergency State Machine.
Encapsulates all 9 formal states, monotonic timers, transition rules,
and audit trail generation independent of the UI layer.
Includes audio acoustic feedback prevention and natural posture recovery checks.
"""

import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable

from ..alarm.alarm_service import AlarmService
from ..speech.speech_service import SpeechService, VerbalIntent
from .temporal_detector import CandidateFallEvent


class EmergencyState(str, Enum):
    NORMAL = "NORMAL"
    SUSPECTED_FALL = "SUSPECTED_FALL"
    LOCAL_ALARM = "LOCAL_ALARM"
    RESPONSE_PROCESSING = "RESPONSE_PROCESSING"
    CANCELLED_OR_FALSE_ALARM = "CANCELLED_OR_FALSE_ALARM"
    EMERGENCY_TRIGGERED = "EMERGENCY_TRIGGERED"
    EMERGENCY_PUBLISHED = "EMERGENCY_PUBLISHED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


@dataclass
class IncidentRecord:
    incident_id: str
    family_id: str
    device_id: str
    device_name: str
    location_label: str
    detected_at: float           # timestamp
    last_known_posture: str
    transition_type: str
    confidence: float
    state: EmergencyState
    voice_response_status: str = "PENDING"
    voice_transcript: str = ""
    evidence_image_bytes: Optional[bytes] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[float] = None
    resolution_note: Optional[str] = None
    events_log: List[Dict[str, Any]] = field(default_factory=list)


class EmergencyStateMachine:
    def __init__(
        self,
        family_id: str = "fam_default",
        device_id: str = "dev_win_01",
        device_name: str = "Windows Monitor",
        location_label: str = "Living Room",
        suspected_fall_grace_sec: float = 3.0,
        voice_response_timeout_sec: float = 10.0,
        alarm_service: Optional[AlarmService] = None,
        speech_service: Optional[SpeechService] = None,
        on_state_change: Optional[Callable[[EmergencyState, Optional[IncidentRecord]], None]] = None,
        on_publish_incident: Optional[Callable[[IncidentRecord], None]] = None,
        on_voice_status: Optional[Callable[[str], None]] = None,
        on_voice_audio_level: Optional[Callable[[float, bool], None]] = None,
    ):
        self.family_id = family_id
        self.device_id = device_id
        self.device_name = device_name
        self.location_label = location_label
        self.suspected_fall_grace_sec = suspected_fall_grace_sec
        self.voice_response_timeout_sec = voice_response_timeout_sec

        self.alarm_service = alarm_service or AlarmService()
        self.speech_service = speech_service or SpeechService(response_timeout_sec=voice_response_timeout_sec)
        self.on_state_change = on_state_change
        self.on_publish_incident = on_publish_incident
        self.on_voice_status = on_voice_status
        self.on_voice_audio_level = on_voice_audio_level

        self.current_state: EmergencyState = EmergencyState.NORMAL
        self.active_incident: Optional[IncidentRecord] = None

        self._state_start_time: float = time.monotonic()
        self._voice_session_active = False
        self._is_prompting = False

    def get_remaining_grace_sec(self) -> float:
        """Returns remaining seconds for the 3.0s suspected fall countdown."""
        if self.current_state != EmergencyState.SUSPECTED_FALL:
            return 0.0
        elapsed = time.monotonic() - self._state_start_time
        return max(0.0, self.suspected_fall_grace_sec - elapsed)

    def get_remaining_voice_sec(self) -> float:
        """Returns remaining seconds for the 10.0s voice response window."""
        if self.current_state != EmergencyState.RESPONSE_PROCESSING:
            return 0.0
        if self._is_prompting:
            return self.voice_response_timeout_sec
        elapsed = time.monotonic() - self._state_start_time
        return max(0.0, self.voice_response_timeout_sec - elapsed)

    def handle_candidate_fall(self, event: CandidateFallEvent, frame_bytes: Optional[bytes] = None):
        """Called when temporal fall detector identifies a potential fall."""
        if self.current_state != EmergencyState.NORMAL:
            # Prevent overlapping incident creation
            return

        incident_id = f"inc_{uuid.uuid4().hex[:10]}"
        self.active_incident = IncidentRecord(
            incident_id=incident_id,
            family_id=self.family_id,
            device_id=self.device_id,
            device_name=self.device_name,
            location_label=self.location_label,
            detected_at=time.time(),
            last_known_posture="LYING",
            transition_type=event.transition_type,
            confidence=event.confidence,
            state=EmergencyState.SUSPECTED_FALL,
            evidence_image_bytes=frame_bytes,
        )
        self._transition_to(EmergencyState.SUSPECTED_FALL, "Candidate fall detected by temporal detector")

    def handle_posture_update(self, current_posture_str: str):
        """Called periodically with latest posture classification."""
        now = time.monotonic()

        # In SUSPECTED_FALL: check if person recovered or 3-second grace timer elapsed
        if self.current_state == EmergencyState.SUSPECTED_FALL:
            if current_posture_str in ("STANDING", "SITTING"):
                # Recovered on their own within grace period
                self._log_event("RECOVERED_NATURALLY", {"posture": current_posture_str})
                self._transition_to(EmergencyState.NORMAL, "Subject stood up/sat up within 3-second grace period")
                self.active_incident = None
                return

            elapsed = now - self._state_start_time
            if elapsed >= self.suspected_fall_grace_sec:
                # 3 seconds elapsed and still lying -> Sound local alarm and process voice
                self._transition_to(EmergencyState.LOCAL_ALARM, "3.0s grace window elapsed with subject still lying")
                self._trigger_voice_verification()

        # In RESPONSE_PROCESSING: check physical recovery or timeout
        elif self.current_state == EmergencyState.RESPONSE_PROCESSING:
            if current_posture_str in ("STANDING", "SITTING"):
                # Subject stood back up on their own during verbal check window
                self._log_event("RECOVERED_NATURALLY", {"posture": current_posture_str})
                self.alarm_service.stop()
                if hasattr(self.speech_service, "cancel_listening"):
                    self.speech_service.cancel_listening()
                self._transition_to(EmergencyState.NORMAL, "Subject stood up/sat up during response verification")
                self.active_incident = None
                return

            if self._is_prompting:
                return

            elapsed = now - self._state_start_time
            if elapsed >= self.voice_response_timeout_sec and self._voice_session_active:
                # 10s voice timeout expired without reassurance -> escalate to emergency
                self._handle_voice_result(VerbalIntent.NO_RESPONSE, "[10.0s Timeout Expired]")

    def handle_verbal_response(self, intent: VerbalIntent, transcript: str):
        """Explicit entrypoint for voice input evaluation (e.g. from speech recognizer or test mock)."""
        if self.current_state == EmergencyState.RESPONSE_PROCESSING:
            self._handle_voice_result(intent, transcript)
        elif self.current_state in (EmergencyState.EMERGENCY_TRIGGERED, EmergencyState.EMERGENCY_PUBLISHED):
            # If the emergency was already triggered by timeout, but user says "help" or "i am fine":
            if intent == VerbalIntent.CONFIRM_SAFE:
                self.alarm_service.stop()
                if hasattr(self.speech_service, "speak_confirmation"):
                    self.speech_service.speak_confirmation("Alarm cancelled. Confirmed safe.")
                self._transition_to(EmergencyState.CANCELLED_OR_FALSE_ALARM, f"User confirmed safe post-trigger: '{transcript}'")
                self._transition_to(EmergencyState.NORMAL, "Monitoring resumed following false alarm resolution")
                self.active_incident = None
            elif intent == VerbalIntent.REQUEST_HELP and self.active_incident:
                self.active_incident.voice_response_status = intent.value
                self.active_incident.voice_transcript = transcript
                self._log_event("VOICE_EVALUATED_POST_TRIGGER", {"intent": intent.value, "transcript": transcript})
                if self.on_publish_incident:
                    self.on_publish_incident(self.active_incident)

    def _trigger_voice_verification(self):
        """Starts acoustic alarm in verification mode and opens microphone listening window."""
        self.alarm_service.start()
        if hasattr(self.alarm_service, "set_verification_mode"):
            self.alarm_service.set_verification_mode(True)

        self._transition_to(EmergencyState.RESPONSE_PROCESSING, "Alarm active, prompting verbal response")
        self._voice_session_active = True
        self._state_start_time = time.monotonic()

        # Pause alarm sound while listening so microphone captures voice without acoustic feedback!
        if hasattr(self.alarm_service, "pause"):
            self.alarm_service.pause()

        def _start_mic_listening():
            self._is_prompting = False
            self._state_start_time = time.monotonic()
            self.speech_service.listen_for_response(
                on_result=self.handle_verbal_response,
                timeout_sec=self.voice_response_timeout_sec,
                on_status=self.on_voice_status,
                on_audio_level=self.on_voice_audio_level,
            )

        if self.voice_response_timeout_sec < 1.0:
            # Fast test mode: bypass TTS prompt so automated tests run instantaneously
            self._is_prompting = False
            _start_mic_listening()
        else:
            self._is_prompting = True
            if self.on_voice_status:
                self.on_voice_status("SafeFall: Are you okay? (Prompting...)")
            self.speech_service.prompt_verbal_check(
                prompt_text="SafeFall alert. Are you okay?",
                on_finished=_start_mic_listening,
            )

    def _handle_voice_result(self, intent: VerbalIntent, transcript: str):
        self._voice_session_active = False

        if not self.active_incident:
            return

        self.active_incident.voice_response_status = intent.value
        self.active_incident.voice_transcript = transcript
        self._log_event("VOICE_EVALUATED", {"intent": intent.value, "transcript": transcript})

        if intent == VerbalIntent.CONFIRM_SAFE:
            # Person clearly confirmed they are fine -> Stop alarm immediately
            self.alarm_service.stop()
            if hasattr(self.speech_service, "speak_confirmation"):
                self.speech_service.speak_confirmation("Alarm cancelled. Confirmed safe.")
            self._transition_to(EmergencyState.CANCELLED_OR_FALSE_ALARM, f"User confirmed safe: '{transcript}'")
            # Reset to normal
            self._transition_to(EmergencyState.NORMAL, "Monitoring resumed following false alarm resolution")
            self.active_incident = None

        elif intent == VerbalIntent.REQUEST_HELP:
            # Immediate escalation without waiting for timeout -> Full emergency alarm
            if hasattr(self.alarm_service, "set_verification_mode"):
                self.alarm_service.set_verification_mode(False)
            self._transition_to(EmergencyState.EMERGENCY_TRIGGERED, f"Immediate escalation on help request: '{transcript}'")
            self._publish_emergency()

        else:
            # NO_RESPONSE, AMBIGUOUS, SPEECH_ERROR -> Safe failure escalation
            if hasattr(self.alarm_service, "set_verification_mode"):
                self.alarm_service.set_verification_mode(False)
            reason = f"Safe escalation on {intent.value}: '{transcript}'"
            self._transition_to(EmergencyState.EMERGENCY_TRIGGERED, reason)
            self._publish_emergency()

    def _publish_emergency(self):
        """Publishes the incident to backend / family dispatch (strictly once per incident)."""
        if not self.active_incident or getattr(self.active_incident, "_is_published", False):
            return
        setattr(self.active_incident, "_is_published", True)
        self._transition_to(EmergencyState.EMERGENCY_PUBLISHED, "Incident published to Cloud Backend")
        if self.on_publish_incident:
            self.on_publish_incident(self.active_incident)

    def acknowledge_incident(self, actor_uid: str, actor_name: str = "Family Member"):
        """Called when a family member taps 'Acknowledge'."""
        if not self.active_incident or self.current_state != EmergencyState.EMERGENCY_PUBLISHED:
            return
        self.active_incident.acknowledged_by = actor_uid
        self.active_incident.acknowledged_at = time.time()
        self._log_event("ACKNOWLEDGED", {"actorUid": actor_uid, "actorName": actor_name})
        self._transition_to(EmergencyState.ACKNOWLEDGED, f"Acknowledged by {actor_name} ({actor_uid})")

    def resolve_incident(self, actor_uid: str, actor_name: str = "Family Admin", resolution_note: str = ""):
        """Called when an authorized family member taps 'Resolve'."""
        if not self.active_incident or self.current_state not in (EmergencyState.EMERGENCY_PUBLISHED, EmergencyState.ACKNOWLEDGED):
            return
        self.alarm_service.stop()
        self.active_incident.resolved_by = actor_uid
        self.active_incident.resolved_at = time.time()
        self.active_incident.resolution_note = resolution_note
        self._log_event("RESOLVED", {"actorUid": actor_uid, "actorName": actor_name, "note": resolution_note})
        self._transition_to(EmergencyState.RESOLVED, f"Resolved by {actor_name}: {resolution_note}")
        
        # Reset to NORMAL after recording resolution
        self._transition_to(EmergencyState.NORMAL, "Monitoring restored after authorized emergency resolution")
        self.active_incident = None

    def _transition_to(self, new_state: EmergencyState, reason: str):
        old_state = self.current_state
        self.current_state = new_state
        self._state_start_time = time.monotonic()
        if self.active_incident:
            self.active_incident.state = new_state
        self._log_event("STATE_TRANSITION", {"from": old_state.value, "to": new_state.value, "reason": reason})
        if self.on_state_change:
            self.on_state_change(new_state, self.active_incident)

    def _log_event(self, event_type: str, metadata: Dict[str, Any]):
        if self.active_incident:
            self.active_incident.events_log.append({
                "timestamp": time.time(),
                "eventType": event_type,
                "metadata": metadata,
            })
