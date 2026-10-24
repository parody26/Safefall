"""
Unit Tests for Verbal Intent Classification & Safety Precedence Rules.
"""

import unittest
from src.speech.speech_service import classify_transcript, VerbalIntent


class TestSpeechIntent(unittest.TestCase):
    def test_reassurance_phrases(self):
        """Scenario 8: Reassurance confirms safety."""
        test_phrases = [
            "I'm fine",
            "I am fine",
            "I'm okay",
            "I am okay, don't worry",
            "I'm good",
            "Everything is fine",
            "I don't need help",
            "all good here",
        ]
        for phrase in test_phrases:
            intent = classify_transcript(phrase)
            self.assertEqual(intent, VerbalIntent.CONFIRM_SAFE, f"Failed for phrase: '{phrase}'")

    def test_help_request_phrases(self):
        """Scenario 9: Help requests trigger immediate escalation."""
        test_phrases = [
            "Help",
            "Help me",
            "help",
            "help me",
            "help me please",
            "Emergency",
            "emergency",
            "It is an emergency",
            "It's an emergency",
            "Please help me I fell",
            "I need help",
            "Call an ambulance",
            "Call my family",
            "I'm hurt and cannot get up",
        ]
        for phrase in test_phrases:
            intent = classify_transcript(phrase)
            self.assertEqual(intent, VerbalIntent.REQUEST_HELP, f"Failed for phrase: '{phrase}'")

    def test_contradictory_precedence(self):
        """Help request must take priority over reassurance in contradictory language."""
        mixed_phrase = "I think I am fine but help me please"
        intent = classify_transcript(mixed_phrase)
        self.assertEqual(intent, VerbalIntent.REQUEST_HELP, "Help must take precedence over reassurance")

        another = "I'm okay, no wait call an ambulance"
        self.assertEqual(classify_transcript(another), VerbalIntent.REQUEST_HELP)

    def test_silence_and_ambiguity(self):
        """Silence maps to NO_RESPONSE; groans map to AMBIGUOUS."""
        self.assertEqual(classify_transcript(""), VerbalIntent.NO_RESPONSE)
        self.assertEqual(classify_transcript("   "), VerbalIntent.NO_RESPONSE)
        self.assertEqual(classify_transcript("uhhh mmm arrgh"), VerbalIntent.AMBIGUOUS)


if __name__ == "__main__":
    unittest.main()
