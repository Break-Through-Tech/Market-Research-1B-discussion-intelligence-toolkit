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
        c, _ = normalize('conversations-gone-awry-cmv-corpus', 'root', [r], meta, {})
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



class ReviewRegressionTests(unittest.TestCase):
    def test_same_output_directory_rejected_without_touching_existing_file(self):
        import tempfile
        from standardize_convokit import run
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output = base/'out'
            output.mkdir()
            sentinel = output/f'{CORPUS}.jsonl'
            sentinel.write_text('existing full dataset')
            alias = base/'alias'
            alias.symlink_to(output, target_is_directory=True)
            for sample in [output, output/'.', alias]:
                with self.subTest(sample=sample), self.assertRaisesRegex(ValueError, 'must be different'):
                    run(base/CORPUS, output, sample)
                self.assertEqual(sentinel.read_text(), 'existing full dataset')

    def test_preflight_checks_sample_directory_against_every_source(self):
        import tempfile
        from standardize_convokit import validate_paths
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'inside the source'):
                validate_paths([base/'one', base/'two'], base/'out', base/'two'/'samples')
            self.assertFalse((base/'out').exists())

    def test_retention_checks_remain_enabled_in_optimized_python(self):
        import subprocess
        code = """
from collections import Counter
from standardize_convokit import validate_retention
for bad in [Counter(input_utterances=1), Counter(input_conversations=1)]:
    try:
        validate_retention(bad)
    except ValueError:
        pass
    else:
        raise SystemExit('retention mismatch was accepted')
validate_retention(Counter(input_utterances=2, source_utterances=1, rejected_utterances=1,
                           input_conversations=2, output_conversations=1, rejected_conversations=1))
"""
        result = subprocess.run([sys.executable, '-O', '-c', code], cwd=Path(__file__).resolve().parents[1]/'artifacts', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_coarse_discourse_parsed_metadata_is_preserved(self):
        c, _ = normalize(CORPUS, 'root', [row(meta={'parsed':[{'example':1}]})], {}, {})
        self.assertEqual(c.root().metadata['source_metadata']['parsed'], [{'example':1}])
        self.assertFalse(c.root().metadata['parsed_annotation_omitted'])

    def fixture(self, base, metadata_only=False):
        import json
        source = base/CORPUS
        source.mkdir()
        (source/'utterances.jsonl').write_text(json.dumps(row(meta={'ups':7,'custom':'keep'}))+'\n')
        conversations = {'root':{}}
        if metadata_only:
            conversations['empty'] = {'title':'metadata-only conversation'}
        for filename, contents in [('conversations.json',conversations),('users.json',{'alice':{'profile':'preserve'}}),('index.json',{}),('corpus.json',{})]:
            (source/filename).write_text(json.dumps(contents))
        return source

    def test_metadata_only_conversation_is_counted_and_quarantined(self):
        import tempfile, io, json
        from contextlib import redirect_stdout
        from standardize_convokit import run
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with redirect_stdout(io.StringIO()):
                result = run(self.fixture(base, True), base/'out', base/'samples')
            self.assertEqual(result['counts']['input_conversations'], 2)
            self.assertEqual(result['counts']['rejected_conversations'], 1)
            rejected = json.loads((base/'out'/f'{CORPUS}.rejected.jsonl').read_text())
            self.assertEqual(rejected['conversation_id'], 'empty')
            self.assertEqual(rejected['source_ids'], [])

    def test_independent_audit_rejects_metadata_tampering_even_with_updated_checksum(self):
        import tempfile, io, json
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from standardize_convokit import run, sha256
        from verify_standardized import verify
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with redirect_stdout(io.StringIO()):
                run(self.fixture(base), base/'out', base/'samples')
            out = base/'out'
            target = out/f'{CORPUS}.jsonl'
            original = json.loads(target.read_text())
            import copy
            for change in ['message', 'speaker', 'score', 'depth']:
                altered = copy.deepcopy(original)
                if change == 'message': altered['utterances'][0]['metadata']['source_metadata']['custom'] = 'lost'
                if change == 'speaker': altered['metadata']['source_speaker_metadata'] = {}
                if change == 'score': altered['utterances'][0]['score'] = 999
                if change == 'depth': altered['utterances'][0]['depth'] = 9
                target.write_text(json.dumps(altered)+'\n')
                mf = out/f'{CORPUS}.manifest.json'
                manifest = json.loads(mf.read_text())
                manifest['output']['sha256'] = sha256(target)
                mf.write_text(json.dumps(manifest))
                with self.subTest(change=change), patch('verify_standardized.CORPORA',(CORPUS,)):
                    with self.assertRaises(ValueError): verify(base, out)


if __name__ == '__main__':
    unittest.main()
