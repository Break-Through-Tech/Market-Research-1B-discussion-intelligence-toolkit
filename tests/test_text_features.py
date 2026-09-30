import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'artifacts'))
from schema import Utterance
from text_features import extract_features

def make_utt(uid, text, **extra):
    return Utterance(
        id=uid, source_id=uid, conversation_id="c1", speaker_id="s1",
        text=text, text_clean=text, created_at=None, depth=0, **extra
    )

class TextFeatureTests(unittest.TestCase):
    def test_one_row_per_utterance(self):    
        u1 = make_utt("1", "hello")
        u2 = make_utt("2", "bye")
        u3 = make_utt("3", "hi")
        df = extract_features([u1, u2, u3])
        self.assertEqual(len(df), 3)

    def test_handles_empty_or_deleted(self):
        u1 = make_utt("1", "")
        u2 = make_utt("2", "[deleted]", is_deleted=True)
        df = extract_features([u1, u2])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["word_count"], 0)
        self.assertEqual(df.iloc[0]["caps_ratio"], 0)

    def test_value_checking(self):
        u1 = make_utt("1", "Is this RIGHT?!")
        df = extract_features([u1])
        self.assertEqual(df.iloc[0]["word_count"], 3)
        self.assertEqual(df.iloc[0]["char_count"], 15)
        self.assertTrue(df.iloc[0]["has_question"])
        self.assertEqual(df.iloc[0]["exclaim_count"], 1)
        self.assertAlmostEqual(df.iloc[0]["caps_ratio"], 6/11)

    def test_has_quote(self):
        df = extract_features([
            make_utt("1", "> old comment\nI agree"),
            make_utt("2", "I agree\n> old comment"),
            make_utt("3", "hello\n\nworld"),
        ])
        self.assertTrue(df.iloc[0]["has_quote"])
        self.assertTrue(df.iloc[1]["has_quote"])
        self.assertFalse(df.iloc[2]["has_quote"])

    def test_has_url(self):
        df = extract_features([
            make_utt("1", "see https://example.com"),
            make_utt("2", "no link here"),
        ])
        self.assertTrue(df.iloc[0]["has_url"])
        self.assertFalse(df.iloc[1]["has_url"])

    def test_is_deleted(self):
        df = extract_features([
            make_utt("1", "[deleted]", is_deleted=True),
            make_utt("2", "hello"),
        ])
        self.assertTrue(df.iloc[0]["is_deleted"])
        self.assertFalse(df.iloc[1]["is_deleted"])