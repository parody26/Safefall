import 'package:flutter_test/flutter_test.dart';
import 'package:safefall/monitoring/state_machine/emergency_state_machine.dart';
import 'package:safefall/monitoring/detector/temporal_detector.dart';

void main() {
  group('Flutter Emergency State Machine Tests', () {
    late FlutterEmergencyStateMachine sm;

    setUp(() {
      sm = FlutterEmergencyStateMachine(
        suspectedFallGraceSec: 0.1,
        voiceResponseTimeoutSec: 0.2,
      );
    });

    test('Candidate fall enters suspectedFall state', () {
      final ev = FallCandidateEvent(
        detectedAt: 100.0,
        transitionType: 'STANDING_TO_LYING',
        durationSec: 0.5,
        confidence: 0.94,
      );
      sm.handleCandidateFall(ev);

      expect(sm.currentState, equals(EmergencyState.suspectedFall));
      expect(sm.currentIncidentId, isNotNull);
    });

    test('Help request escalates immediately', () {
      final ev = FallCandidateEvent(
        detectedAt: 100.0,
        transitionType: 'STANDING_TO_LYING',
        durationSec: 0.5,
        confidence: 0.94,
      );
      sm.handleCandidateFall(ev);
      // simulate grace expiration
      sm.handlePostureTick('LYING');

      sm.handleVerbalResponse(VerbalIntent.requestHelp, 'Help me please');
      expect(sm.currentState, equals(EmergencyState.emergencyPublished));
      expect(sm.voiceStatus, equals('REQUESTHELP'));
    });

    test('Reassurance stops alarm and resets to normal', () {
      final ev = FallCandidateEvent(
        detectedAt: 100.0,
        transitionType: 'STANDING_TO_LYING',
        durationSec: 0.5,
        confidence: 0.94,
      );
      sm.handleCandidateFall(ev);
      sm.handlePostureTick('LYING');

      sm.handleVerbalResponse(VerbalIntent.confirmSafe, 'I am fine');
      expect(sm.currentState, equals(EmergencyState.normal));
      expect(sm.isAlarmRinging, isFalse);
    });
  });
}
