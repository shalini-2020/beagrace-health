"""
Independent nurse evaluation, kept separate from ordinary pilot use and feedback.

Every evaluator rates the SAME notes, made once from the SAME approved recordings, so
results can be compared across nurses and agreement between them can be measured.

  1. Scenarios (eval_scenarios.json): standardised, simulated clinical scenarios. Only
     entries with "approved": true are used. Reference answers are NOT stored here.
  2. Recordings: consenting speakers record each approved scenario. Name each file
     <scenario_id>.<ext> or <scenario_id>__<speaker code>.<ext> (e.g. ANC-01__S1.m4a)
     and keep them in a private folder (consent must cover evaluators listening).
  3. Generate notes once, on your own computer (needs the API keys in .env and
     `hf auth login` with write access to the private usage dataset):
         python nurse_eval.py generate <recordings folder> [--with-baseline]
     Each recording goes through the same pipeline nurses use; --with-baseline also
     makes an LLM-only note from the same transcript. Notes and recordings are saved to
     the PRIVATE dataset only. Each note has a random label; its condition is stored for
     the administrator only.
  4. Evaluators (codes in the EVALUATOR_CODES secret, e.g. EV01) open the Evaluation
     tab. Blinded: they see the scenario, the original recording and the note, never the
     transcript, glossary match, confidence or condition. They answer the same five
     questions for each note, in their own shuffled order, once each. Question 1
     (meaning) only counts from evaluators who understand Ekiti Yoruba; nurses assess
     clinical completeness, usefulness and safety.
  5. Reference answers: qualified Yoruba-speaking and clinical reviewers write them in
     eval_references.csv (template: eval_references_template.csv) WITHOUT seeing any
     output. The file is git-ignored: keep it private.
  6. Score:  python nurse_eval.py score eval_references.csv
     -> danger-sign detection against the references (counted once per note), evaluator
        answers, agreement between evaluators (Krippendorff's alpha), and
        eval_out/nurse_evaluation_results.csv for term-level adjudication.

Evaluation runs are never counted in the pilot usage, feedback or Impact totals.
Never use a real patient encounter as a scenario or recording.
"""
import csv
import hashlib
import json
import random
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from usage_log import LOG_DIR, LOG_REPO, append_record, load_records, redact_comment

SCENARIOS_FILE = Path('eval_scenarios.json')
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.aac', '.ogg', '.opus', '.mp4', '.3gp', '.amr',
                    '.webm', '.flac'}
UNABLE = 'unable_to_assess'

# The five questions: same wording and options for every participant.
QUESTIONS = [
    ('meaning', '1. Does the English handoff note preserve the original symptom meaning?',
     [('Accurate', 'accurate'), ('Partially accurate', 'partially_accurate'),
      ('Incorrect', 'incorrect'), ('Unable to assess', UNABLE)]),
    ('usefulness', '2. Is the generated note useful for nurse handoff?',
     [('Useful', 'useful'), ('Partially useful', 'partially_useful'), ('Not useful', 'not_useful')]),
    ('issues', '3. Does the note contain missing or misleading clinical information?',
     [('No important issues', 'none'), ('Minor issues', 'minor'), ('Important issues', 'important'),
      ('Unable to assess', UNABLE)]),
    ('danger_signs', '4. Are danger signs represented appropriately?',
     [('Correctly handled', 'correct'), ('Important warning missed', 'missed'),
      ('Incorrect warning', 'incorrect_warning'), ('No danger sign in scenario', 'not_applicable'),
      ('Unable to assess', UNABLE)]),
    ('would_use', '5. Would you use the note after reviewing it?',
     [('Yes', 'yes'), ('Only after corrections', 'after_corrections'), ('No', 'no')]),
]
QUESTION_KEYS = [q[0] for q in QUESTIONS]


# ---------------------------------------------------------------- scenarios

def load_scenarios() -> list:
    """Approved scenarios only."""
    if not SCENARIOS_FILE.exists():
        return []
    try:
        items = json.loads(SCENARIOS_FILE.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'[evaluation] could not read {SCENARIOS_FILE}: {e}')
        return []
    return [s for s in items if s.get('approved') is True and s.get('scenario_id')]


def scenario_by_id(scenario_id: str):
    return next((s for s in load_scenarios() if s['scenario_id'] == scenario_id), None)


def scenario_version(scenario: dict) -> str:
    """Fingerprint of the scenario text, so results are tied to the exact wording used."""
    data = json.dumps(scenario, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(data.encode('utf-8')).hexdigest()[:8]


# ---------------------------------------------------------------- generate (run once, locally)

def _recordings(folder: Path, scenarios: dict) -> list:
    found = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        scenario_id, _, speaker = path.stem.partition('__')
        if scenario_id in scenarios:
            found.append((path, scenario_id, speaker or None))
        else:
            print(f'  skipped {path.name}: no approved scenario "{scenario_id}"')
    return found


CONDITIONS = ('clinguard', 'llm_only')


def _llm_only(app, transcript: str, lang: str) -> dict:
    """Baseline: the same LLM and note prompt, but no ClinGuard glossary matching and no
    rule-based danger check. Uses the same transcript, so speech recognition is held constant."""
    cfg = app.lang_cfg(lang)
    prompt = (f"Translate this {cfg['name']} speech ({cfg.get('dialect_note', 'local dialect')}, "
              'transcribed by speech recognition, may contain errors) into plain English.\n'
              'Reply with the English translation only, one short sentence. If it cannot be '
              f"understood, reply exactly: UNCLEAR\n\n{cfg['name']}: {transcript}")
    english = app.call_groq(prompt, max_tokens=400)
    if english.strip().upper().startswith('UNCLEAR'):
        english = 'Unclear - please ask the patient to repeat'
    note = app.generate_clinical_note(english, '', lang)
    return {'english': english, 'meaning': '', 'note': note, 'urgency': app.extract_urgency(note),
            'danger_sign': None, 'danger_rule_ids': [], 'confidence': None, 'confidence_level': None,
            'match_id': None, 'glossary_phrase': ''}


def generate(folder: str, upload: bool = True, baseline: bool = False) -> str:
    """Make one batch of notes from the approved recordings. baseline=True also makes an
    LLM-only note from the same transcript, stored with condition 'llm_only' (admin only)."""
    import app   # the same pipeline nurses use
    scenarios = {s['scenario_id']: s for s in load_scenarios()}
    if not scenarios:
        print('No approved scenarios in eval_scenarios.json.')
        return ''
    recordings = _recordings(Path(folder), scenarios)
    if not recordings:
        print(f'No recordings in {folder} named after an approved scenario.')
        return ''
    batch_id = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    out_file = LOG_DIR / f'eval_outputs_{batch_id}.jsonl'
    audio_dir = LOG_DIR / 'eval_audio'
    audio_dir.mkdir(exist_ok=True)
    print(f"Batch {batch_id}: {len(recordings)} recordings"
          f"{' (ClinGuard and LLM-only)' if baseline else ''}")
    made, audio_files = 0, []
    for path, scenario_id, speaker in recordings:
        sc = scenarios[scenario_id]
        details = {}
        app.process_audio(str(path), sc.get('department', 'Not specified'), app.DEFAULT_LANG,
                          record=False, details=details)
        if not details:
            print(f'  {path.name}: NO NOTE (no speech or an error) - check the recording')
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        audio_name = f'{sha}{path.suffix.lower()}'   # no scenario id in the name
        (audio_dir / audio_name).write_bytes(path.read_bytes())
        audio_files.append(audio_dir / audio_name)
        common = {'batch_id': batch_id, 'scenario_id': scenario_id,
                  'scenario_version': scenario_version(sc), 'speaker': speaker,
                  'recording_sha256': sha, 'audio_file': audio_name,
                  'department': sc.get('department'), 'language': app.DEFAULT_LANG}
        versions = {k: details.get(k) for k in ('app_version', 'code_fingerprint', 'clinguard_version',
                                                'glossary_version', 'rules_version', 'speech_model')}
        outputs = [{**common, **details, 'condition': 'clinguard'}]
        if baseline:
            outputs.append({**common, **versions, 'transcript': details.get('transcript'),
                            **_llm_only(app, details.get('transcript', ''), app.DEFAULT_LANG),
                            'condition': 'llm_only'})
        for o in outputs:
            o['output_id'] = uuid.uuid4().hex[:12]   # random label: reveals nothing about the condition
            o['time_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
            with out_file.open('a', encoding='utf-8') as f:
                f.write(json.dumps(o, ensure_ascii=False) + '\n')
            made += 1
        print(f"  {path.name}: {details.get('english')} ({details.get('confidence_level')})"
              f"{' + LLM-only note' if baseline else ''}")
    print(f'\n{made} notes saved to {out_file}')
    if made and upload:
        try:
            from huggingface_hub import CommitOperationAdd, HfApi
            ops = [CommitOperationAdd(path_in_repo=f'logs/{out_file.name}', path_or_fileobj=str(out_file))]
            ops += [CommitOperationAdd(path_in_repo=f'eval_audio/{f.name}', path_or_fileobj=str(f))
                    for f in audio_files]
            HfApi().create_commit(repo_id=LOG_REPO, repo_type='dataset', operations=ops,
                                  commit_message=f'Evaluation batch {batch_id}')
            print(f'Uploaded notes and recordings to the PRIVATE dataset {LOG_REPO}. '
                  'Evaluators can start rating.')
        except Exception as e:
            print(f'Could not upload ({e}). Run `hf auth login` with a write token and try again.')
    return batch_id


def audio_for(output: dict):
    """Local path of the approved recording for this note, or None. On the Space it is
    fetched from the private dataset; it is never part of the public repo."""
    name = (output or {}).get('audio_file')
    if not name:
        return None
    local = LOG_DIR / 'eval_audio' / name
    if local.exists():
        return str(local)
    import os
    if os.environ.get('HF_TOKEN'):
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(LOG_REPO, f'eval_audio/{name}', repo_type='dataset',
                                   token=os.environ['HF_TOKEN'])
        except Exception as e:
            print(f'[evaluation] could not fetch recording: {e}')
    return None


# ---------------------------------------------------------------- rating (Evaluation tab)

def latest_batch() -> list:
    outputs = load_records('eval_outputs')
    if not outputs:
        return []
    batch = max(o.get('batch_id', '') for o in outputs)
    return [o for o in outputs if o.get('batch_id') == batch]


def next_for(evaluator_id: str):
    """(next unrated output or None, number rated, total) for this evaluator, in an order
    shuffled per evaluator so position effects average out across nurses."""
    batch = latest_batch()
    done = {e.get('output_id') for e in load_records('evaluations')
            if e.get('evaluator_id') == evaluator_id}
    order = sorted(batch, key=lambda o: o['output_id'])
    random.Random(evaluator_id).shuffle(order)
    remaining = [o for o in order if o['output_id'] not in done]
    return (remaining[0] if remaining else None), len(batch) - len(remaining), len(batch)


def already_rated(evaluator_id: str, output_id: str) -> bool:
    return any(e.get('evaluator_id') == evaluator_id and e.get('output_id') == output_id
               for e in load_records('evaluations'))


def save_evaluation(evaluator_id: str, reads_yoruba: bool, output: dict, answers: dict,
                    comment: str) -> str:
    """answers: {question key: option value}. Question 1 is stored as 'unable to assess'
    for evaluators who do not understand Ekiti Yoruba."""
    clean = {}
    for key, _, options in QUESTIONS:
        value = answers.get(key)
        clean[key] = value if value in {v for _, v in options} else None
    if not reads_yoruba:
        clean['meaning'] = UNABLE
    evaluation_id = uuid.uuid4().hex[:12]
    append_record('evaluations', {
        'evaluation_id': evaluation_id, 'evaluator_id': evaluator_id,
        'evaluator_reads_yoruba': bool(reads_yoruba),
        # which note was rated (the full output stays in its eval_outputs record)
        **{k: output.get(k) for k in ('output_id', 'batch_id', 'condition', 'scenario_id',
                                      'scenario_version', 'speaker', 'english', 'confidence_level', 'urgency',
                                      'danger_sign', 'app_version', 'code_fingerprint',
                                      'clinguard_version', 'glossary_version', 'rules_version')},
        **clean,
        'comment': redact_comment(comment),
    })
    return evaluation_id


def responses_text(evaluator_id: str, output: dict, answers: dict, reads_yoruba: bool) -> str:
    """Plain-text copy of one evaluator's answers, for their own records."""
    lines = [f"Evaluator {evaluator_id} · note {output.get('output_id')}"]
    for key, question, options in QUESTIONS:
        value = answers.get(key) if (key != 'meaning' or reads_yoruba) else UNABLE
        label = next((l for l, v in options if v == value), '-')
        lines.append(f'{question} {label}')
    return '\n'.join(lines)


# ---------------------------------------------------------------- scoring

def _wilson(k: int, n: int) -> str:
    from compare_baselines import wilson
    return wilson(k, n)


def _yes(v) -> bool:
    return str(v or '').strip().lower() in ('yes', 'y', 'true', '1')


def krippendorff_alpha_nominal(units: dict):
    """units: {item: [ratings from different evaluators]}. Items with fewer than two
    ratings are ignored. Returns None when it cannot be computed."""
    pairs = Counter()
    for values in units.values():
        values = [v for v in values if v is not None]
        m = len(values)
        if m < 2:
            continue
        counts = Counter(values)
        for a in counts:
            for b in counts:
                pairs[(a, b)] += counts[a] * (counts[b] - (1 if a == b else 0)) / (m - 1)
    n = sum(pairs.values())
    if n <= 1:
        return None
    marg = Counter()
    for (a, _), v in pairs.items():
        marg[a] += v
    observed = sum(v for (a, b), v in pairs.items() if a != b) / n
    expected = sum(marg[a] * marg[b] for a in marg for b in marg if a != b) / (n * (n - 1))
    return None if expected == 0 else 1 - observed / expected


def score(references_path: str, batch_id: str = None) -> None:
    with open(references_path, encoding='utf-8-sig', newline='') as f:
        refs = {r['scenario_id'].strip(): r for r in csv.DictReader(f)
                if (r.get('scenario_id') or '').strip()
                and not r['scenario_id'].upper().startswith('EXAMPLE')}
    outputs = load_records('eval_outputs')
    batch_id = batch_id or (max(o.get('batch_id', '') for o in outputs) if outputs else None)
    outputs = {o['output_id']: o for o in outputs
               if o.get('batch_id') == batch_id and o.get('scenario_id') in refs}
    evals = [e for e in load_records('evaluations') if e.get('output_id') in outputs]
    if not outputs:
        print('No generated notes match the scenario_ids in the references file.')
        return
    print(f'Batch {batch_id}: {len(outputs)} notes, {len(evals)} assessments by '
          f"{len({e['evaluator_id'] for e in evals})} evaluators "
          f"({len({e['evaluator_id'] for e in evals if e.get('evaluator_reads_yoruba')})} understand Ekiti Yoruba)")
    conditions = sorted({o.get('condition') or 'clinguard' for o in outputs.values()},
                        key=lambda c: CONDITIONS.index(c) if c in CONDITIONS else 99)
    for condition in conditions:
        outs = {k: o for k, o in outputs.items() if (o.get('condition') or 'clinguard') == condition}
        evs = [e for e in evals if e['output_id'] in outs]
        _report(condition, outs, evs, refs)
    if len(conditions) > 1:
        print('\nNote: blinding is partial. ClinGuard notes can carry the safety-check banner, '
              'which the LLM-only notes never do.')
    _write_results(outputs, evals, refs)


def _report(condition: str, outputs: dict, evals: list, refs: dict) -> None:
    title = {'clinguard': 'BeaGrace with ClinGuard', 'llm_only': 'LLM only (no ClinGuard)'}.get(condition, condition)
    print(f'\n=== {title}: {len(outputs)} notes, {len(evals)} assessments ===')

    # 1. Output against independent references: one count per note
    danger = [o for o in outputs.values() if _yes(refs[o['scenario_id']].get('reference_danger'))]
    safe = [o for o in outputs.values() if not _yes(refs[o['scenario_id']].get('reference_danger'))]
    high = lambda o: (o.get('urgency') or '').lower() == 'high'
    print('Output against independent reference answers (one count per note):')
    print(f"  Danger sign -> note urgency High:  {_wilson(sum(1 for o in danger if high(o)), len(danger))}")
    print(f"  No danger sign -> urgency High:    {_wilson(sum(1 for o in safe if high(o)), len(safe))}")
    if condition == 'clinguard':
        print(f"  Rule-based alert on danger signs:  {_wilson(sum(1 for o in danger if o.get('danger_sign')), len(danger))}")
        print(f"  Rule-based false alerts:           {_wilson(sum(1 for o in safe if o.get('danger_sign')), len(safe))}")
    ref_urg = [o for o in outputs.values() if refs[o['scenario_id']].get('reference_urgency', '').strip()]
    if ref_urg:
        same = sum(1 for o in ref_urg if (o.get('urgency') or '').lower()
                   == refs[o['scenario_id']]['reference_urgency'].strip().lower())
        print(f'  Urgency matches reference:         {_wilson(same, len(ref_urg))}')

    # 2. Evaluator answers: one count per assessment, 'unable to assess' left out
    def share(key, good, rows):
        rated = [e for e in rows if e.get(key) and e[key] != UNABLE]
        return _wilson(sum(1 for e in rated if e[key] in good), len(rated))
    yoruba = [e for e in evals if e.get('evaluator_reads_yoruba')]
    print('Evaluator answers (one count per assessment; "unable to assess" left out):')
    print(f"  Q1 Meaning accurate (Yoruba-speaking evaluators only): {share('meaning', {'accurate'}, yoruba)}")
    print(f"  Q2 Useful for handoff:             {share('usefulness', {'useful'}, evals)}")
    print(f"  Q3 No important issues:            {share('issues', {'none', 'minor'}, evals)}")
    danger_rated = [e for e in evals if e.get('danger_signs') not in (None, UNABLE, 'not_applicable')]
    print(f"  Q4 Danger signs correctly handled: "
          f"{_wilson(sum(1 for e in danger_rated if e['danger_signs'] == 'correct'), len(danger_rated))}"
          f" | warnings missed: {sum(1 for e in evals if e.get('danger_signs') == 'missed')}"
          f" | incorrect warnings: {sum(1 for e in evals if e.get('danger_signs') == 'incorrect_warning')}")
    print(f"  Q5 Would use (as is):              {share('would_use', {'yes'}, evals)}")
    print(f"     Would use, incl. after corrections: {share('would_use', {'yes', 'after_corrections'}, evals)}")

    # 3. Agreement between evaluators on the same notes
    print("Agreement between evaluators (Krippendorff's alpha, nominal; 1 = perfect, 0 = chance):")
    for key, _, _ in QUESTIONS:
        units = defaultdict(list)
        for e in (yoruba if key == 'meaning' else evals):
            if e.get(key) and e[key] != UNABLE:
                units[e['output_id']].append(e[key])
        alpha = krippendorff_alpha_nominal(units)
        multi = sum(1 for v in units.values() if len(v) >= 2)
        print(f"  {key:<14} {'n/a' if alpha is None else f'{alpha:.2f}'} ({multi} notes rated by 2+ evaluators)")


def _write_results(outputs: dict, evals: list, refs: dict) -> None:
    rows = []
    for e in evals:
        o, ref = outputs[e['output_id']], refs[outputs[e['output_id']]['scenario_id']]
        rows.append({
            'evaluation_id': e.get('evaluation_id'), 'time_utc': e.get('time_utc'),
            'evaluator_id': e.get('evaluator_id'),
            'evaluator_reads_yoruba': 'yes' if e.get('evaluator_reads_yoruba') else 'no',
            'output_id': o['output_id'], 'condition': o.get('condition') or 'clinguard',
            'scenario_id': o['scenario_id'], 'scenario_version': o.get('scenario_version'),
            'speaker': o.get('speaker'), 'recording_sha256': o.get('recording_sha256'),
            'transcript': o.get('transcript'), 'app_symptom': o.get('english'),
            'glossary_phrase': o.get('glossary_phrase'), 'confidence_level': o.get('confidence_level'),
            'app_urgency': o.get('urgency'),
            'rule_alert': '' if o.get('danger_sign') is None else ('yes' if o.get('danger_sign') else 'no'),
            'reference_meaning': ref.get('reference_meaning'), 'reference_term': ref.get('reference_term'),
            'reference_danger': 'yes' if _yes(ref.get('reference_danger')) else 'no',
            'reference_urgency': ref.get('reference_urgency', ''),
            **{f'q_{k}': e.get(k) for k in QUESTION_KEYS},
            'comment': e.get('comment'),
            'term_matches_reference': '',   # filled in by a qualified reviewer
            'app_version': o.get('app_version'), 'code_fingerprint': o.get('code_fingerprint'),
            'clinguard_version': o.get('clinguard_version'),
            'glossary_version': o.get('glossary_version'), 'rules_version': o.get('rules_version'),
        })
    if rows:
        out = Path('eval_out')
        out.mkdir(exist_ok=True)
        path = out / 'nurse_evaluation_results.csv'
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\nSaved {path} (admin only: it shows each note\'s condition). A qualified reviewer '
              'should fill in term_matches_reference (yes/no) by comparing app_symptom with reference_term.')


if __name__ == '__main__':
    if len(sys.argv) in (3, 4) and sys.argv[1] == 'generate':
        generate(sys.argv[2], baseline='--with-baseline' in sys.argv[3:])
    elif len(sys.argv) in (3, 4) and sys.argv[1] == 'score':
        score(sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None)
    else:
        print('Usage:\n  python nurse_eval.py generate <recordings folder> [--with-baseline]\n'
              '  python nurse_eval.py score eval_references.csv [batch_id]')
