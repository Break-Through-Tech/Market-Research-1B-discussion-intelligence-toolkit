"""Independent full-file source-preservation audit for standardized corpora."""
import argparse
import json
from pathlib import Path

from standardize_convokit import CORPORA, canonical, sha256, write_json
from schema import Conversation


def read_checked(path):
    """Validate tree reachability independently of producer traversal helpers."""
    with path.open() as f:
        for line in f:
            conv = Conversation.model_validate_json(line)
            nodes = {u.id: u for u in conv.utterances}
            roots = [u.id for u in conv.utterances if u.parent_id is None]
            if len(nodes) != len(conv.utterances) or len(roots) != 1:
                raise ValueError('invalid root or duplicate message ID')
            children = {}
            for u in conv.utterances:
                if u.conversation_id != conv.id or u.speaker_id not in conv.speakers:
                    raise ValueError('invalid message ownership')
                if u.parent_id is not None and u.parent_id not in nodes:
                    raise ValueError('unresolved parent')
                children.setdefault(u.parent_id, []).append(u.id)
            visited = set()
            stack = [(roots[0], 0)]
            while stack:
                uid, depth = stack.pop()
                if uid in visited or nodes[uid].depth != depth:
                    raise ValueError('cycle or invalid depth')
                visited.add(uid)
                stack.extend((child, depth+1) for child in children.get(uid, []))
            if visited != set(nodes):
                raise ValueError('unreachable messages')
            yield conv


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
                if r['id'] in expected:
                    raise ValueError('duplicate source message IDs')
                meta = dict(r.get('meta', {}))
                omitted = name == 'conversations-gone-awry-cmv-corpus' and meta.pop('parsed', None) is not None
                r['meta'] = meta
                r['_parsed_omitted'] = omitted
                expected[r['id']] = r
        cm = json.loads((source/'conversations.json').read_text())
        sf = source / ('users.json' if (source/'users.json').exists() else 'speakers.json')
        speakers = json.loads(sf.read_text())
        seen_conversations = set()
        seen = set()
        count = synthetic = 0
        for conv in read_checked(target):
            count += 1
            if conv.source_id in seen_conversations:
                raise ValueError('duplicate source conversation')
            seen_conversations.add(conv.source_id)
            wanted_speakers = {}
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
                r = expected[u.source_id]
                cid = r.get('conversation_id', r.get('root'))
                text, parent, label, ts = r.get('text'), r.get('reply-to'), r['meta'].get('majority_type'), r.get('timestamp')
                if u.metadata['source_metadata'] != r['meta']:
                    raise ValueError(f'message metadata changed: {u.source_id}')
                if u.score != r['meta'].get('score', r['meta'].get('ups')):
                    raise ValueError(f'voting score changed: {u.source_id}')
                if u.metadata['parsed_annotation_omitted'] != r['_parsed_omitted'] or u.metadata['source_vectors'] != r.get('vectors', []):
                    raise ValueError('annotation/vector metadata changed')
                sid = r.get('speaker', r.get('user'))
                spid = canonical(name, 'synthetic-speaker', cid) if sid is None else canonical(name, 'speaker', str(sid))
                handle = '[unknown]' if sid is None else str(sid)
                if u.speaker_id != spid or conv.speakers[spid].handle != handle:
                    raise ValueError('speaker identity changed')
                if sid in speakers:
                    wanted_speakers[spid] = speakers[sid]
                actual_ts = None if u.created_at is None else u.created_at.timestamp()
                if (conv.source_id, u.text, u.metadata['source_parent_id'], u.discourse_act, actual_ts) != (cid, text if text is not None else '', parent, label, ts):
                    raise ValueError(f'source fields changed: {u.source_id}')
                if u.metadata['text_was_null'] != (text is None):
                    raise ValueError(f'null text flag changed: {u.source_id}')
                if parent is None:
                    if u.parent_id is not None: raise ValueError('root changed')
                elif parent in expected and expected[parent].get('conversation_id', expected[parent].get('root')) == cid:
                    if u.parent_id != canonical(name, 'utterance', parent):
                        raise ValueError(f'existing parent edge changed: {u.source_id}')
                elif by_id[u.parent_id].source_id != parent or not by_id[u.parent_id].metadata.get('is_synthetic'):
                    raise ValueError('missing parent not represented by correct placeholder')
            if conv.metadata['source_speaker_metadata'] != wanted_speakers:
                raise ValueError('speaker profile metadata changed')
            if conv.metadata['source_vectors'] != cm[conv.source_id].get('vectors', []):
                raise ValueError('conversation vectors changed')
        if seen != set(expected):
            raise ValueError('missing source utterances')
        if seen_conversations != set(cm):
            raise ValueError('missing source conversations')
        results.append({'corpus':name, 'verified_conversations':count,
                        'verified_source_utterances':len(seen), 'synthetic_placeholders':synthetic,
                        'source_and_output_checksums_match':True,
                        'source_text_labels_timestamps_and_existing_edges_preserved':True,
                        'message_and_speaker_metadata_preserved':True})
    write_json(output/'source-preservation-check.json', results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, default=Path.home()/'.convokit/saved-corpora')
    ap.add_argument('--output-dir', type=Path, default=Path('data/local/standardized-v1'))
    args = ap.parse_args()
    verify(args.source_root, args.output_dir)
