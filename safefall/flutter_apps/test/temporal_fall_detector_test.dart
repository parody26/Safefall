import 'package:flutter_test/flutter_test.dart';
import 'package:safefall/monitoring/pose/pose_detector.dart';
import 'package:safefall/monitoring/detector/temporal_detector.dart';

void main() {
  group('Flutter Temporal Fall Detector Tests', () {
    late FlutterTemporalFallDetector detector;

    setUp(() {
      detector = FlutterTemporalFallDetector(
        historyWindowSec: 4.0,
        maxTransitionDurationSec: 1.4,
        minLyingConfirmationSec: 0.3,
        cooldownSec: 10.0,
      );
    });

    test('Rapid Standing to Lying triggers fall event', () {
      detector.addObservation(
        posture: PostureState.standing,
        confidence: 0.92,
        timestamp: 10.0,
      );
      detector.addObservation(
        posture: PostureState.standing,
        confidence: 0.90,
        timestamp: 10.2,
      );
      detector.addObservation(
        posture: PostureState.lying,
        confidence: 0.95,
        timestamp: 10.6,
      );
      final ev = detector.addObservation(
        posture: PostureState.lying,
        confidence: 0.95,
        timestamp: 11.0,
      );

      expect(ev, isNotNull);
      expect(ev!.transitionType, equals('STANDING_TO_LYING'));
    });

    test('Slow transition over 3 seconds is not a fall', () {
      detector.addObservation(
        posture: PostureState.standing,
        confidence: 0.90,
        timestamp: 10.0,
      );
      detector.addObservation(
        posture: PostureState.lying,
        confidence: 0.92,
        timestamp: 13.0,
      );
      final ev = detector.addObservation(
        posture: PostureState.lying,
        confidence: 0.92,
        timestamp: 13.5,
      );

      expect(ev, isNull);
    });
  });
}
