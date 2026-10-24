import 'package:flutter_test/flutter_test.dart';
import 'package:safefall/monitoring/pose/pose_detector.dart';

void main() {
  group('Flutter Pose Classifier Tests', () {
    late FlutterPoseClassifier classifier;

    setUp(() {
      classifier = FlutterPoseClassifier();
    });

    test('Standing posture classification', () {
      final landmarks = List.generate(33, (i) => LandmarkPoint(0.5, 0.5, 0.0, 1.0));
      landmarks[11] = LandmarkPoint(0.46, 0.28, 0.0, 0.95);
      landmarks[12] = LandmarkPoint(0.54, 0.28, 0.0, 0.95);
      landmarks[23] = LandmarkPoint(0.47, 0.52, 0.0, 0.95);
      landmarks[24] = LandmarkPoint(0.53, 0.52, 0.0, 0.95);
      landmarks[25] = LandmarkPoint(0.47, 0.72, 0.0, 0.95);
      landmarks[26] = LandmarkPoint(0.53, 0.72, 0.0, 0.95);
      landmarks[27] = LandmarkPoint(0.47, 0.88, 0.0, 0.95);
      landmarks[28] = LandmarkPoint(0.53, 0.88, 0.0, 0.95);

      final feat = FlutterPoseClassifier.extractFeatures(landmarks);
      expect(feat, isNotNull);
      expect(feat!.torsoAngleDeg, lessThan(35.0));

      final res = classifier.classify(feat);
      expect(res.key, equals(PostureState.standing));
      expect(res.value, greaterThan(0.70));
    });

    test('Lying posture classification', () {
      final landmarks = List.generate(33, (i) => LandmarkPoint(0.5, 0.5, 0.0, 1.0));
      landmarks[11] = LandmarkPoint(0.20, 0.82, 0.0, 0.95);
      landmarks[12] = LandmarkPoint(0.24, 0.82, 0.0, 0.95);
      landmarks[23] = LandmarkPoint(0.42, 0.84, 0.0, 0.95);
      landmarks[24] = LandmarkPoint(0.46, 0.84, 0.0, 0.95);
      landmarks[25] = LandmarkPoint(0.62, 0.85, 0.0, 0.95);
      landmarks[26] = LandmarkPoint(0.66, 0.85, 0.0, 0.95);
      landmarks[27] = LandmarkPoint(0.80, 0.86, 0.0, 0.95);
      landmarks[28] = LandmarkPoint(0.84, 0.86, 0.0, 0.95);

      final feat = FlutterPoseClassifier.extractFeatures(landmarks);
      expect(feat, isNotNull);
      expect(feat!.torsoAngleDeg, greaterThan(55.0));

      final res = classifier.classify(feat);
      expect(res.key, equals(PostureState.lying));
      expect(res.value, greaterThan(0.70));
    });
  });
}
