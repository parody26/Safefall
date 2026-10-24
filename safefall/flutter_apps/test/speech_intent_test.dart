import 'package:flutter_test/flutter_test.dart';
import 'package:safefall/monitoring/state_machine/emergency_state_machine.dart';

void main() {
  group('Dart Verbal Intent Classification Tests', () {
    test('Reassurance phrases map to confirmSafe', () {
      expect(classifyTranscriptDart("I'm fine"), equals(VerbalIntent.confirmSafe));
      expect(classifyTranscriptDart("I am fine"), equals(VerbalIntent.confirmSafe));
      expect(classifyTranscriptDart("I'm okay, no worries"), equals(VerbalIntent.confirmSafe));
      expect(classifyTranscriptDart("I don't need help"), equals(VerbalIntent.confirmSafe));
    });

    test('Help cries map to requestHelp', () {
      expect(classifyTranscriptDart("Help me!"), equals(VerbalIntent.requestHelp));
      expect(classifyTranscriptDart("Call an ambulance"), equals(VerbalIntent.requestHelp));
      expect(classifyTranscriptDart("I'm in pain, help"), equals(VerbalIntent.requestHelp));
    });

    test('Contradictory phrases prioritize help', () {
      expect(classifyTranscriptDart("I am fine but help me please"), equals(VerbalIntent.requestHelp));
    });

    test('Silence maps to noResponse', () {
      expect(classifyTranscriptDart(""), equals(VerbalIntent.noResponse));
      expect(classifyTranscriptDart("   "), equals(VerbalIntent.noResponse));
    });
  });
}
