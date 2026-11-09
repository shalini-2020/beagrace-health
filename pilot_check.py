"""
Pilot health check: one command to see how the pilot is going, from the private logs.

    python pilot_check.py              # everything since the pilot started
    python pilot_check.py 2026-11-09   # only from this date (UTC)

Run it on your own computer after `hf auth login`. It only reads; it changes nothing and
shows no patient information (no transcripts, notes or comments). Logs reach the dataset
about every 10 minutes, so the last few minutes may be missing.
"""
import os
import sys
from collections import Counter, defaultdict

import usage_log   # import first: the Space's log syncing must not start on this computer
from impact import PILOT_START, TEST_DEPARTMENT


def main(since: str) -> None:
    from huggingface_hub import get_token
    token = get_token()
    if not token:
        print('Not logged in to Hugging Face. Run:  hf auth login')
        return
    os.environ['SPACE_ID'], os.environ['HF_TOKEN'] = 'pilot-check', token   # read-only access

    usage = sorted([r for r in usage_log.load_records('usage') if r.get('time_utc', '') >= since],
                   key=lambda r: r['time_utc'])
    tests = [r for r in usage if r.get('department') == TEST_DEPARTMENT]
    real = [r for r in usage if r.get('department') != TEST_DEPARTMENT]
    notes = [r for r in real if r.get('outcome') == 'note']
    feedback = {}
    for f in sorted(usage_log.load_records('feedback'), key=lambda f: f.get('time_utc', '')):
        feedback[f.get('note_id')] = f

    print(f'Pilot check from {since} (UTC)   |   latest record: '
          f"{usage[-1]['time_utc'][:16].replace('T', ' ') if usage else 'none'}")
    print(f'Attempts: {len(real)} real use, {len(tests)} testing (excluded)')
    outcomes = Counter(r.get('outcome') for r in real)
    print(f"  notes {outcomes['note']} | no speech {outcomes['no_speech']} | errors {outcomes['error']}")
    errors = Counter(r.get('error_stage') or '?' for r in real if r.get('outcome') == 'error')
    if errors:
        print('  errors by stage: ' + ', '.join(f'{k} {v}' for k, v in errors.most_common()))

    rated = [r for r in notes if r.get('note_id') in feedback]
    ratings = Counter(feedback[r['note_id']].get('rating') for r in rated)
    print(f'\nRatings: {len(rated)} of {len(notes)} notes rated | '
          f"correct {ratings['correct']} | needs changes {ratings['needs_changes']} | not sure {ratings['not_sure']}")
    comments = sum(1 for r in rated if feedback[r['note_id']].get('comment'))
    print(f'  with a written comment: {comments}')

    missing = Counter()
    for r in notes:
        for field, label in (('nurse_code', 'nurse code'), ('facility_code', 'facility code')):
            if not r.get(field):
                missing[label] += 1
        if r.get('department') in (None, '', 'Not specified'):
            missing['department'] += 1
    print('\nMissing on notes: ' + (', '.join(f'{k} {v}' for k, v in missing.items()) if missing else 'nothing'))

    per_nurse = defaultdict(lambda: [0, 0])
    for r in notes:
        per_nurse[r.get('nurse_code') or '(no code)'][0] += 1
        per_nurse[r.get('nurse_code') or '(no code)'][1] += r.get('note_id') in feedback
    print('\nBy nurse (notes / rated):')
    for nurse, (n, k) in sorted(per_nurse.items()):
        print(f'  {nurse:<10} {n} / {k}')
    print('By facility: ' + ', '.join(f'{k} {v}' for k, v in
                                      Counter(r.get('facility_code') or '(none)' for r in notes).most_common()))
    print('By department: ' + ', '.join(f'{k} {v}' for k, v in
                                        Counter(r.get('department') or '(none)' for r in notes).most_common()))
    print('Type of use: ' + ', '.join(f'{k} {v}' for k, v in
                                      Counter(r.get('use_mode') or '(not recorded)' for r in notes).most_common()))
    print('App versions: ' + ', '.join(f'{k} {v}' for k, v in
                                       Counter(r.get('app_version') or '(not recorded)' for r in real).most_common()))

    unrated = [r for r in notes if r.get('note_id') not in feedback]
    if unrated:
        print('\nNot rated:')
        for r in unrated:
            print(f"  {r['time_utc'][:16].replace('T', ' ')}  nurse {r.get('nurse_code') or '-'}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else PILOT_START)
