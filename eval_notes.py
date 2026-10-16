"""
Test the clinical note step on the BeaGrace team's labelled examples.

Runs every example in clinical_examples.json that is NOT used inside the prompt
through the app's note generator, and writes eval_results.csv for a nurse to
review (fill in the nurse_correct and nurse_comment columns).

Usage:  python eval_notes.py
Needs GROQ_API_KEY in .env. About 31 calls; stays within Groq's free limits.
"""
import csv
import json
import re
import time
from pathlib import Path

from app import generate_clinical_note

EXAMPLES_FILE = Path('clinical_examples.json')
OUTPUT_FILE = Path('eval_results.csv')


def field(note: str, name: str) -> str:
    match = re.search(rf'{name}\s*:\s*(.+)', note, re.IGNORECASE)
    return match.group(1).strip().strip('*').strip() if match else ''


def main() -> None:
    examples = json.loads(EXAMPLES_FILE.read_text(encoding='utf-8'))
    test_set = [e for e in examples if not e.get('use_in_prompt')]
    print(f'Testing {len(test_set)} examples...')

    rows = []
    for i, ex in enumerate(test_set, 1):
        print(f'[{i}/{len(test_set)}] {ex["patient_says"][:60]}')
        try:
            note = generate_clinical_note(ex['patient_says'], '')
        except Exception as e:
            note = f'ERROR: {e}'
        rows.append({
            'department': ex['department'],
            'patient_says': ex['patient_says'],
            'expected_term': ex['clinical_term'],
            'app_symptom': field(note, 'Symptom'),
            'app_urgency': field(note, 'Urgency'),
            'full_note': note,
            'nurse_correct': '',
            'nurse_comment': '',
        })
        time.sleep(2)  # stay well under the 30 requests/minute limit

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    high = sum(1 for r in rows if 'high' in r['app_urgency'].lower())
    print(f'\nSaved {OUTPUT_FILE} ({len(rows)} rows, {high} marked High urgency).')
    print('Open it in Excel and ask a nurse to fill in nurse_correct (Y/N).')


if __name__ == '__main__':
    main()
