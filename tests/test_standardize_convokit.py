"""Graph, missing-data, label preservation, and text regression tests."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'artifacts'))
from schema import Conversation
from standardize_convokit import clean_text, normalize, validate_conversation, canonical

CORPUS = 'reddit-coarse-discourse-corpus'


def row(uid='root', parent=None, text='Hello', **extra):
    r = {'id': uid, 'root': 'root', 'user': 'alice', 'reply-to': parent,
         'text': text, 'timestamp': None, 'meta': {}}
    r.update(extra)
    return r


class StandardizationTests(unittest.TestCase):
    def convert(self, rows, meta=None):
        return normalize(CORPUS, 'root', rows, meta or {}, {'alice': {}})

    def test_null_text_and_missing_time_are_not_invented(self):
        c, counts = self.convert([row(text=None)])
        self.assertEqual(c.root().text, '')
        self.assertTrue(c.root().metadata['text_was_null'])
        self.assertFalse(c.root().is_deleted)
        self.assertIsNone(c.created_at)
        self.assertEqual(counts['null_text'], 1)
        self.assertEqual(Conversation.model_validate_json(c.model_dump_json()), c)

    def test_missing_parent_preserves_reply_and_flags_synthetic_edge(self):
        c, counts = self.convert([row(), row('child', 'missing')])
        child = next(u for u in c.utterances if u.source_id == 'child')
        parent = next(u for u in c.utterances if u.id == child.parent_id)
        self.assertTrue(parent.metadata['parent_edge_is_synthetic'])
        self.assertEqual(child.metadata['source_parent_id'], 'missing')
        self.assertEqual(child.depth, 2)
        self.assertEqual(counts['synthetic_utterances'], 1)
        self.assertEqual(len(list(c.traverse())), 3)

    def test_shared_missing_parent_creates_only_one_placeholder(self):
        c, counts = self.convert([row(), row('a','missing'), row('b','missing')])
        self.assertEqual(counts['synthetic_utterances'], 1)
        self.assertEqual(counts['orphaned_replies'], 2)

    def test_cycle_is_rejected_without_hanging(self):
        with self.assertRaisesRegex(ValueError, 'cycle'):
            self.convert([row(), row('a','b'), row('b','a')])

    def test_multiple_roots_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'one source root'):
            self.convert([row(), row('another')])

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.convert([row(), row('root','root')])

    def test_deleted_author_does_not_mean_deleted_text(self):
        c, _ = self.convert([row(user='[deleted]'), row('a','root','[removed]')])
        self.assertFalse(c.root().is_deleted)
        self.assertTrue(c.speaker_of(c.root()).is_deleted)
        self.assertTrue(c.utterances[1].is_deleted)

    def test_cleaning_preserves_language_and_markdown(self):
        text = '  > I do NOT agree!\r\n\r\n\r\n[Evidence](https://example.org) &amp; **why**\t? '
        self.assertEqual(clean_text(text), '> I do NOT agree!\n\n[Evidence](https://example.org) & **why** ?')
        c, _ = self.convert([row(text=text)])
        self.assertEqual(c.root().text, text)

    def test_new_convokit_format_and_labels_are_preserved(self):
        r = {'id':'root', 'conversation_id':'root', 'speaker':'bob', 'text':'Hi',
             'reply-to':None, 'timestamp':1440446000,
             'meta':{'majority_type':'other','parsed':[{'huge':'annotation'}]}}
        meta = {'meta':{'pair_id':'pair','split':'test','has_removed_comment':True}, 'vectors':[]}
        c, _ = self.convert([r], meta)
        self.assertEqual(c.root().discourse_act, 'other')
        self.assertEqual(c.metadata['source_metadata'], meta['meta'])
        self.assertEqual(c.root().created_at.utcoffset().total_seconds(), 0)
        self.assertNotIn('parsed', c.root().metadata['source_metadata'])

    def test_validator_catches_wrong_depth_and_speaker(self):
        c, _ = self.convert([row(), row('a','root')])
        c.utterances[1].depth = 9
        with self.assertRaises(ValueError): validate_conversation(c)
        c.utterances[1].depth = 1
        c.utterances[1].speaker_id = 'missing'
        with self.assertRaises(ValueError): validate_conversation(c)

    def test_child_before_parent_and_deep_chain(self):
        rows = [row()] + [row(str(i), 'root' if i == 0 else str(i-1)) for i in range(1100)]
        c, _ = self.convert(list(reversed(rows)))
        self.assertEqual(c.utterances[-1].depth, 1100)
        self.assertEqual(len(list(c.traverse())), 1101)

    def test_corpus_ids_cannot_collide(self):
        self.assertNotEqual(canonical('a','utterance','x'), canonical('b','utterance','x'))


class PipelineTests(unittest.TestCase):
    def test_disk_pipeline_quarantines_invalid_thread_and_streams_valid_output(self):
        import tempfile
        import json
        import io
        from contextlib import redirect_stdout
        from standardize_convokit import run, iter_standardized
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / CORPUS
            source.mkdir()
            rows = [row(), row('a', 'b', root='broken'), row('b', 'a', root='broken')]
            (source/'utterances.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            for name, contents in [('conversations.json', {'root':{}, 'broken':{}}), ('users.json', {'alice':{}}), ('index.json',{}), ('corpus.json',{})]:
                (source/name).write_text(json.dumps(contents))
            with redirect_stdout(io.StringIO()):
                report = run(source, base/'out', base/'samples')
            self.assertEqual(report['counts']['rejected_conversations'], 1)
            self.assertEqual(report['counts']['rejected_utterances'], 2)
            self.assertEqual(report['counts']['output_conversations'], 1)
            self.assertEqual(len(list(iter_standardized(base/'out'/f'{CORPUS}.jsonl'))), 1)
            self.assertEqual(len((base/'out'/f'{CORPUS}.rejected.jsonl').read_text().splitlines()), 1)
            with self.assertRaisesRegex(ValueError, 'inside the source'):
                run(source, source/'out', base/'samples')


if __name__ == '__main__':
    unittest.main()
