"""Independent full-file source-preservation audit for standardized corpora."""
import argparse
import json
from pathlib import Path

from standardize_convokit import CORPORA, canonical, iter_standardized, sha256, write_json


def verify(source_root: Path, output: Path):
    results = []
    for name in CORPORA:
        source = source_root/name
        manifest = json.loads((output/f'{name}.manifest.json').read_text())
        for filename, info in manifest['source']['files'].items():
            if sha256(source/filename) != info['sha256']:
                raise ValueError(f'source checksum changed: {name}/{filename}')
        target = output/manifest['output']['file']
        if sha256(target) != manifest['output']['sha256']:
            raise ValueError(f'output checksum changed: {name}')
        # Retain just the source fields needed for the audit, not bulky NLP parses.
        expected = {}
        with (source/'utterances.jsonl').open() as f:
            for line in f:
                r = json.loads(line)
                expected[r['id']] = (r.get('conversation_id', r.get('root')), r.get('text'),
                                     r.get('reply-to'), r.get('meta',{}).get('majority_type'),
                                     r.get('timestamp'))
        cm = json.loads((source/'conversations.json').read_text())
        seen = set()
        count = synthetic = 0
        for conv in iter_standardized(target):
            count += 1
            original_meta = cm[conv.source_id].get('meta', cm[conv.source_id])
            if conv.metadata['source_metadata'] != original_meta:
                raise ValueError(f'conversation labels/metadata changed: {conv.id}')
            by_id = {u.id:u for u in conv.utterances}
            for u in conv.utterances:
                if u.metadata['is_synthetic']:
                    synthetic += 1
                    continue
                if u.source_id in seen:
                    raise ValueError(f'duplicate output source message: {u.source_id}')
                seen.add(u.source_id)
                cid, text, parent, label, ts = expected[u.source_id]
                actual_ts = None if u.created_at is None else u.created_at.timestamp()
                if (conv.source_id, u.text, u.metadata['source_parent_id'], u.discourse_act, actual_ts) != (cid, text if text is not None else '', parent, label, ts):
                    raise ValueError(f'source fields changed: {u.source_id}')
                if u.metadata['text_was_null'] != (text is None):
                    raise ValueError(f'null text flag changed: {u.source_id}')
                if parent is None:
                    if u.parent_id is not None: raise ValueError('root changed')
                elif parent in expected and expected[parent][0] == cid:
                    if u.parent_id != canonical(name, 'utterance', parent):
                        raise ValueError(f'existing parent edge changed: {u.source_id}')
                elif by_id[u.parent_id].source_id != parent or not by_id[u.parent_id].metadata.get('is_synthetic'):
                    raise ValueError('missing parent not represented by correct placeholder')
        if seen != set(expected):
            raise ValueError('missing source utterances')
        if count != len(cm):
            raise ValueError('missing source conversations')
        results.append({'corpus':name, 'verified_conversations':count,
                        'verified_source_utterances':len(seen), 'synthetic_placeholders':synthetic,
                        'source_and_output_checksums_match':True,
                        'source_text_labels_timestamps_and_existing_edges_preserved':True})
    write_json(output/'source-preservation-check.json', results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, default=Path.home()/'.convokit/saved-corpora')
    ap.add_argument('--output-dir', type=Path, default=Path('data/local/standardized-v1'))
    args = ap.parse_args()
    verify(args.source_root, args.output_dir)
