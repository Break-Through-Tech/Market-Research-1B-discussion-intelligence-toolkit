"""Offline ConvoKit-to-canonical adapter. No downloads, training, or source edits.

Run from the repo root: python artifacts/standardize_convokit.py --help
Rows are staged in temporary SQLite storage and processed one conversation at a time.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sqlite3
import tempfile

from schema import Conversation, Speaker, Utterance, SCHEMA_VERSION

CORPORA = ('reddit-coarse-discourse-corpus', 'conversations-gone-awry-cmv-corpus')
VERSION = '1.0'


def clean_text(text: str) -> str:
    """Conservative normalization; preserve markdown, URLs, quotes and case."""
    text = html.unescape(text).replace('\r\n', '\n').replace('\r', '\n')
    text = '\n'.join(re.sub(r'[^\S\n]+', ' ', line).strip() for line in text.split('\n'))
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def canonical(corpus: str, kind: str, raw: str) -> str:
    # Corpus namespaces prevent collisions when datasets overlap.
    return f'reddit:{corpus}:{kind}:{raw}'


def timestamp(value):
    return None if value is None else datetime.fromtimestamp(float(value), timezone.utc)


def validate_conversation(conv: Conversation) -> None:
    """Check graph invariants before invoking traversal (which assumes a tree)."""
    by_id = {u.id: u for u in conv.utterances}
    if len(by_id) != len(conv.utterances):
        raise ValueError('duplicate utterance IDs')
    root = conv.root()
    if root.depth != 0:
        raise ValueError('root depth must be zero')
    for u in conv.utterances:
        if u.conversation_id != conv.id or u.speaker_id not in conv.speakers:
            raise ValueError('invalid conversation/speaker reference')
        if u.parent_id is not None:
            if u.parent_id not in by_id or u.depth != by_id[u.parent_id].depth + 1:
                raise ValueError('invalid parent or depth')
    # Positive depth increments make cycles impossible; check connectivity as well.
    visited = [u.id for u in conv.traverse()]
    if len(visited) != len(set(visited)) or set(visited) != set(by_id):
        raise ValueError('disconnected or cyclic conversation')


def normalize(corpus: str, cid: str, rows: list[dict], source_meta: dict,
              source_speakers: dict) -> tuple[Conversation, Counter]:
    counts = Counter()
    roots = [r for r in rows if r.get('reply-to') is None]
    if len(roots) != 1:
        raise ValueError(f'expected one source root, found {len(roots)}')
    raw_ids = [r['id'] for r in rows]
    if len(set(raw_ids)) != len(rows):
        raise ValueError('duplicate source IDs')
    ids = set(raw_ids)
    conv_id = canonical(corpus, 'conversation', cid)
    root_id = canonical(corpus, 'utterance', roots[0]['id'])
    speakers = {}
    speaker_meta = {}
    utterances = []
    missing = sorted({r['reply-to'] for r in rows if r.get('reply-to') is not None and r['reply-to'] not in ids})
    unknown_id = canonical(corpus, 'synthetic-speaker', cid)
    for r in rows:
        sid = r.get('speaker', r.get('user'))
        if sid is None:
            spid = unknown_id
            speaker = Speaker(id=spid, handle='[unknown]', source='reddit')
            counts['missing_speaker'] += 1
        else:
            spid = canonical(corpus, 'speaker', str(sid))
            speaker = Speaker(id=spid, handle=str(sid), source='reddit', is_deleted=sid in ('[deleted]', '[removed]'))
            if sid in source_speakers:
                speaker_meta[spid] = source_speakers[sid]
        speakers[spid] = speaker
        raw_text = r.get('text')
        if raw_text is not None and not isinstance(raw_text, str):
            raise ValueError('unexpected non-string text; review rather than silently coerce')
        text = raw_text if raw_text is not None else ''
        cleaned = clean_text(text)
        deleted = text.strip() in ('[deleted]', '[removed]')
        counts['null_text'] += raw_text is None
        counts['empty_clean_text'] += not bool(cleaned)
        counts['deleted_text'] += deleted
        counts['missing_timestamp'] += r.get('timestamp') is None
        meta = dict(r.get('meta', {}))
        parsed = meta.pop('parsed', None)
        counts['omitted_parsed_annotations'] += parsed is not None
        par = r.get('reply-to')
        if par in missing:
            parent = canonical(corpus, 'missing-parent', f'{cid}:{par}')
            counts['orphaned_replies'] += 1
        else:
            parent = None if par is None else canonical(corpus, 'utterance', par)
        source_depth = meta.get('post_depth')
        utterances.append(Utterance(
            id=canonical(corpus, 'utterance', r['id']), source_id=r['id'],
            conversation_id=conv_id, parent_id=parent, speaker_id=spid,
            text=text, text_clean=cleaned, created_at=timestamp(r.get('timestamp')), depth=0,
            score=meta.get('score', meta.get('ups')), is_deleted=deleted,
            discourse_act=meta.get('majority_type'),
            metadata={'source_metadata': meta, 'source_parent_id': par,
                      'source_order': len(utterances), 'source_depth': source_depth,
                      'text_was_null': raw_text is None,
                      'is_synthetic': False,
                      'source_vectors': r.get('vectors', []),
                      'parsed_annotation_omitted': parsed is not None}))
    if missing:
        speakers.setdefault(unknown_id, Speaker(id=unknown_id, handle='[unknown]', source='reddit'))
    for parent in missing:
        # Its attachment to the root is a structural scaffold, NOT a recovered edge.
        utterances.append(Utterance(
            id=canonical(corpus, 'missing-parent', f'{cid}:{parent}'), source_id=parent,
            conversation_id=conv_id, parent_id=root_id, speaker_id=unknown_id,
            text='', text_clean='', created_at=None, depth=0,
            metadata={'is_synthetic': True, 'reason': 'missing_parent',
                      'parent_edge_is_synthetic': True, 'source_parent_id': None}))
    by_id = {u.id: u for u in utterances}
    children = defaultdict(list)
    for u in utterances:
        if u.parent_id is not None:
            children[u.parent_id].append(u.id)
    ordered = []
    visited = set()
    stack = [(root_id, 0)]
    while stack:
        uid, depth = stack.pop()
        if uid in visited:
            raise ValueError('cycle or duplicate traversal')
        visited.add(uid)
        u = by_id[uid]
        u.depth = depth
        if u.metadata.get('source_depth') is not None:
            counts['source_depth_differences'] += u.metadata['source_depth'] != depth
        ordered.append(u)
        stack.extend((child, depth + 1) for child in reversed(children[uid]))
    if len(visited) != len(utterances):
        raise ValueError('unreachable nodes / cycle in source replies')
    meta = source_meta.get('meta', source_meta)
    conv = Conversation(
        id=conv_id, source_id=cid, source='reddit', title=meta.get('title'), url=meta.get('url'),
        created_at=timestamp(roots[0].get('timestamp')), utterances=ordered, speakers=speakers,
        metadata={'corpus': corpus, 'source_metadata': meta, 'source_speaker_metadata': speaker_meta,
                  'source_vectors': source_meta.get('vectors', []), 'pipeline_version': VERSION,
                  'has_synthetic_structure': bool(missing)})
    validate_conversation(conv)
    counts['synthetic_utterances'] = len(missing)
    counts['source_utterances'] = len(rows)
    counts['output_utterances'] = len(ordered)
    counts['conversations_with_repairs'] = bool(missing)
    return conv, counts


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def iter_standardized(path: str | Path):
    """Stream validated Conversation objects without loading the full dataset."""
    with Path(path).open() as f:
        for number, line in enumerate(f, 1):
            try:
                conv = Conversation.model_validate_json(line)
                validate_conversation(conv)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{number}: {exc}') from exc
            yield conv


def run(source: Path, output: Path, sample_dir: Path):
    corpus = source.name
    if corpus not in CORPORA:
        raise ValueError(f'unsupported corpus: {corpus}')
    for destination in (output, sample_dir):
        if destination.resolve().is_relative_to(source.resolve()):
            raise ValueError('outputs must not be inside the source corpus')
    output.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    conv_meta = json.loads((source / 'conversations.json').read_text())
    speaker_file = source / ('users.json' if (source / 'users.json').exists() else 'speakers.json')
    source_speakers = json.loads(speaker_file.read_text())
    totals = Counter({'rejected_conversations': 0, 'rejected_utterances': 0,
                      'orphaned_replies': 0, 'source_depth_differences': 0})
    unique_speakers = set()
    source_used_speakers = set()
    labels = Counter()
    splits = Counter()
    sample = []
    selected_kinds = set()
    outpath = output / f'{corpus}.jsonl'
    rejected = output / f'{corpus}.rejected.jsonl'
    with tempfile.TemporaryDirectory(prefix='convokit-standardize-') as tmp:
        db = sqlite3.connect(str(Path(tmp) / 'rows.sqlite'))
        db.execute('CREATE TABLE rows (seq INTEGER PRIMARY KEY, cid TEXT, uid TEXT UNIQUE, payload TEXT)')
        with (source / 'utterances.jsonl').open() as f:
            for seq, line in enumerate(f):
                r = json.loads(line)
                cid = r.get('conversation_id', r.get('root'))
                if not isinstance(cid, str) or not isinstance(r.get('id'), str):
                    raise ValueError(f'invalid IDs on line {seq+1}')
                db.execute('INSERT INTO rows VALUES (?, ?, ?, ?)', (seq, cid, r['id'], line))
                totals['input_utterances'] += 1
        db.execute('CREATE INDEX by_conversation ON rows(cid)')
        db.commit()
        cids = [r[0] for r in db.execute('SELECT cid FROM rows GROUP BY cid ORDER BY cid')]
        totals['input_conversations'] = len(cids)
        with outpath.open('w') as out, rejected.open('w') as errors:
            for cid in cids:
                rows = [json.loads(r[0]) for r in db.execute('SELECT payload FROM rows WHERE cid=? ORDER BY seq', (cid,))]
                try:
                    if cid not in conv_meta:
                        raise ValueError('missing conversation metadata')
                    conv, counts = normalize(corpus, cid, rows, conv_meta[cid], source_speakers)
                    serialized = conv.model_dump_json()
                    # Validate the actual serialized artifact, not just in-memory construction.
                    validate_conversation(Conversation.model_validate_json(serialized))
                except (ValueError, TypeError, OverflowError) as exc:
                    totals['rejected_conversations'] += 1
                    totals['rejected_utterances'] += len(rows)
                    errors.write(json.dumps({'conversation_id': cid, 'reason': str(exc), 'source_ids': [r['id'] for r in rows]}) + '\n')
                    continue
                out.write(serialized + '\n')
                totals.update(counts)
                totals['output_conversations'] += 1
                unique_speakers.update(conv.speakers)
                source_used_speakers.update(str(r.get('speaker', r.get('user'))) for r in rows if r.get('speaker', r.get('user')) is not None)
                labels.update(u.discourse_act for u in conv.utterances if u.discourse_act is not None)
                sm = conv.metadata['source_metadata']
                if 'split' in sm:
                    splits[str(sm['split'])] += 1
                kind = 'repaired' if counts['synthetic_utterances'] else 'ordinary'
                if kind not in selected_kinds and len(rows) <= 15:
                    # Public fixtures avoid publishing author handles/profile metadata.
                    public = conv.model_copy(deep=True)
                    mapping = {sid: f'sample:speaker:{i}' for i, sid in enumerate(public.speakers)}
                    public.speakers = {mapping[sid]: Speaker(id=mapping[sid], handle=f'participant-{i}', source='reddit', is_deleted=s.is_deleted) for i, (sid, s) in enumerate(public.speakers.items())}
                    for u in public.utterances:
                        u.speaker_id = mapping[u.speaker_id]
                    public.metadata['source_speaker_metadata'] = {}
                    public.metadata['sample_speaker_ids_anonymized'] = True
                    validate_conversation(public)
                    sample.append(public.model_dump_json())
                    selected_kinds.add(kind)
        db.close()
    totals['unique_output_speaker_ids'] = len(unique_speakers)
    totals['unique_source_speaker_ids'] = len(source_used_speakers)
    assert totals['input_utterances'] == totals['source_utterances'] + totals['rejected_utterances']
    assert totals['input_conversations'] == totals['output_conversations'] + totals['rejected_conversations']
    (sample_dir / f'{corpus}.jsonl').write_text('\n'.join(sample) + '\n')
    source_files = ['utterances.jsonl', 'conversations.json', speaker_file.name, 'corpus.json', 'index.json']
    manifest = {
        'corpus': corpus, 'pipeline_version': VERSION, 'schema_version': SCHEMA_VERSION,
        'implementation_sha256': {name: sha256(Path(__file__).parent/name)
                                  for name in ('standardize_convokit.py', 'schema.py')},
        'processed_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': {'download_name': corpus,
                   'archive_url': f'http://zissou.infosci.cornell.edu/convokit/datasets/{corpus}/{corpus}.zip',
                   'documentation': 'https://convokit.cornell.edu/documentation/',
                   'download_record': json.loads((source/'local-download-record.json').read_text()) if (source/'local-download-record.json').exists() else None,
                   'files': {f: {'sha256': sha256(source/f), 'bytes': (source/f).stat().st_size} for f in source_files}},
        'selection': 'All source conversations, no sampling; invalid conversations quarantined',
        'split_policy': 'Preserve source split and pair_id metadata. No new train/test split.',
        'counts': dict(totals), 'discourse_labels': dict(labels), 'source_splits': dict(splits),
        'output': {'file': outpath.name, 'sha256': sha256(outpath), 'bytes': outpath.stat().st_size},
        'rejections': rejected.name,
        'validation': 'Every retained conversation passed JSON round-trip, schema, root, parent, speaker, depth, connectivity and traversal checks.',
        'terms': 'Upstream source terms apply. Redistribution must follow the source documentation; processing grants no new rights.'}
    write_json(output / f'{corpus}.manifest.json', manifest)
    print(json.dumps({'corpus': corpus, 'counts': dict(totals)}, indent=2), flush=True)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, default=Path.home()/'.convokit/saved-corpora')
    ap.add_argument('--output-dir', type=Path, default=Path('data/local/standardized-v1'))
    ap.add_argument('--sample-dir', type=Path, default=Path('data/standardized-samples'))
    args = ap.parse_args()
    for corpus in CORPORA:
        source = args.source_root / corpus
        if args.output_dir.resolve().is_relative_to(source.resolve()):
            ap.error('output directory must not be inside the source corpus')
    reports = [run(args.source_root/n, args.output_dir, args.sample_dir) for n in CORPORA]
    write_json(args.output_dir/'validation-summary.json', reports)
    if any(r['counts'].get('rejected_conversations', 0) for r in reports):
        raise SystemExit('Review quarantined conversations before publishing outputs.')


if __name__ == '__main__':
    main()
