"""
Blinded three-way comparison: general translator vs base LLM vs BeaGrace + ClinGuard.

All three systems get the SAME text for each case (speech recognition is held
constant), so differences come from interpretation, not transcription.

  A  general     - general-purpose translator (Google Translate via deep-translator)
  B  llm         - the same LLM the app uses, asked to translate and flag danger,
                   with no glossary and no ClinGuard
  C  clinguard   - the BeaGrace pipeline: ClinGuard glossary match first, LLM
                   fallback only when nothing matches, rule-based danger check

Step 1 - prepare cases (eval_cases.csv, see eval_cases_template.csv)
    Use consented, de-identified recordings' transcripts or standard test
    phrases, never routine patient encounters. Cover common terms, dialect
    variants, clinically important symptoms, negative (non-danger) cases and
    ambiguous phrasing. Qualified Yoruba-speaking and clinical reviewers write
    reference_meaning, reference_term and reference_danger INDEPENDENTLY of any
    system output. Keep some cases out of the glossary on purpose.

Step 2 - run the systems
    python compare_baselines.py run eval_cases.csv
    -> eval_out/blinded_review.csv   (shuffled, system names hidden: give to reviewers)
    -> eval_out/key.csv              (which system made each output: do NOT share)

Step 3 - reviewers fill in blinded_review.csv
    meaning_score     correct / partly / wrong
    important_omission yes / no   (a clinically important detail lost or changed)
    reviewer_code     their pseudonymous code
    Two reviewers per output is best; record disagreements and the agreed result
    in adjudicated_meaning_score.

Step 4 - score
    python compare_baselines.py score eval_cases.csv eval_out/blinded_review.csv
    -> eval_out/results.csv and a printed summary per system: meaning accuracy,
       important-omission rate, danger-sign recall, false-alert rate, and for
       ClinGuard the split between glossary-matched and fallback cases. 95%
       Wilson intervals are shown because pilot samples are small.

Needs GROQ_API_KEY in .env, and:  pip install deep-translator
"""
import csv
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

OUT = Path('eval_out')
SYSTEMS = ('general', 'llm', 'clinguard')


def read_csv(path) -> list:
    with open(path, encoding='utf-8-sig', newline='') as f:
        return [r for r in csv.DictReader(f) if (r.get('case_id') or '').strip()
                and not r['case_id'].upper().startswith('EXAMPLE')]


def write_csv(path, rows) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def yes(value) -> bool:
    return str(value).strip().lower() in ('yes', 'y', 'true', '1')


# ---------- the three systems ----------

def run_general(text: str, lang: str) -> dict:
    from deep_translator import GoogleTranslator
    english = GoogleTranslator(source=lang, target='en').translate(text) or ''
    # A general translator gives no danger decision; it is scored as "not flagged"
    return {'meaning': english, 'danger': 'no', 'detail': ''}


def run_llm(text: str, lang: str) -> dict:
    from app import call_groq, lang_cfg
    cfg = lang_cfg(lang)
    prompt = f"""This is {cfg['name']} speech ({cfg.get('dialect_note', 'local dialect')}) from a patient, transcribed by speech recognition.
Answer in exactly two lines:
Meaning: <plain English meaning, one short sentence, or UNCLEAR>
Danger: <yes if this could be a clinical danger sign needing urgent escalation, otherwise no>

{cfg['name']}: {text}"""
    reply = call_groq(prompt, max_tokens=400)
    meaning, danger = '', 'no'
    for line in reply.splitlines():
        if line.lower().startswith('meaning:'):
            meaning = line.split(':', 1)[1].strip()
        elif line.lower().startswith('danger:'):
            danger = 'yes' if 'yes' in line.lower() else 'no'
    return {'meaning': meaning or reply.strip(), 'danger': danger, 'detail': ''}


def run_clinguard(text: str, lang: str) -> dict:
    from app import translate_detailed, confidence_level, lang_cfg
    from safety import check_danger
    r = translate_detailed(text, lang)
    reasons = check_danger(glossary_term=r['english'] if r['confidence'] > 0 else '',
                           transcript=text,
                           keywords_key=lang_cfg(lang).get('danger_keywords', ''),
                           english_text=f"{r['english']} {r['meaning']}")
    meaning = r['english'] + (f" ({r['meaning']})" if r['meaning'] else '')
    return {'meaning': meaning, 'danger': 'yes' if reasons else 'no',
            'detail': confidence_level(r)}


RUNNERS = {'general': run_general, 'llm': run_llm, 'clinguard': run_clinguard}


def run(cases_path: str) -> None:
    cases = read_csv(cases_path)
    print(f'{len(cases)} cases x {len(SYSTEMS)} systems')
    outputs = []
    for i, case in enumerate(cases, 1):
        lang = (case.get('language') or 'yo').strip()
        for system in SYSTEMS:
            try:
                result = RUNNERS[system](case['input_text'], lang)
            except Exception as e:
                result = {'meaning': f'ERROR: {e}', 'danger': 'error', 'detail': 'error'}
            outputs.append({'case_id': case['case_id'], 'system': system, **result})
            if system != 'clinguard' or result['detail'] == 'low':
                time.sleep(2)   # stay under the API rate limits
        print(f'  [{i}/{len(cases)}] {case["case_id"]}')

    random.seed(20261102)
    random.shuffle(outputs)
    review, key = [], []
    by_case = {c['case_id']: c for c in cases}
    for n, o in enumerate(outputs, 1):
        output_id = f'O{n:04d}'
        review.append({
            'output_id': output_id,
            'case_id': o['case_id'],
            'input_text': by_case[o['case_id']]['input_text'],
            'reference_meaning': by_case[o['case_id']].get('reference_meaning', ''),
            'system_meaning': o['meaning'],
            'meaning_score': '',              # correct / partly / wrong
            'important_omission': '',         # yes / no
            'reviewer_code': '',
            'adjudicated_meaning_score': '',  # filled only when reviewers disagree
            'reviewer_notes': '',
        })
        key.append({'output_id': output_id, 'case_id': o['case_id'], 'system': o['system'],
                    'danger_flag': o['danger'], 'confidence_level': o['detail']})
    write_csv(OUT / 'blinded_review.csv', review)
    write_csv(OUT / 'key.csv', key)
    print(f'\nGive {OUT / "blinded_review.csv"} to the reviewers. Keep {OUT / "key.csv"} private.')


# ---------- scoring ----------

def wilson(k: int, n: int) -> str:
    if not n:
        return 'n/a'
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f'{100 * p:.0f}% ({k}/{n}; 95% CI {100 * max(0, centre - half):.0f}-{100 * min(1, centre + half):.0f}%)'


def score(cases_path: str, review_path: str) -> None:
    cases = {c['case_id']: c for c in read_csv(cases_path)}
    key = {k['output_id']: k for k in read_csv_any(OUT / 'key.csv')}
    reviews = read_csv_any(review_path)
    stats = defaultdict(lambda: defaultdict(int))
    rows = []
    for r in reviews:
        k = key.get(r['output_id'])
        if not k:
            continue
        case = cases.get(k['case_id'], {})
        system = k['system']
        label = (r.get('adjudicated_meaning_score') or r.get('meaning_score') or '').strip().lower()
        groups = [system]
        if system == 'clinguard' and k.get('confidence_level'):
            groups.append('clinguard:' + ('glossary' if k['confidence_level'] in ('high', 'partial') else 'fallback'))
        for g in groups:
            s = stats[g]
            if label in ('correct', 'partly', 'wrong'):
                s['scored'] += 1
                s['correct'] += label == 'correct'
            if (r.get('important_omission') or '').strip():
                s['omission_scored'] += 1
                s['omission'] += yes(r['important_omission'])
            if (case.get('reference_danger') or '').strip():
                if yes(case['reference_danger']):
                    s['true_danger'] += 1
                    s['caught'] += yes(k['danger_flag'])
                else:
                    s['non_danger'] += 1
                    s['false_alert'] += yes(k['danger_flag'])
            if system == 'clinguard' and g == 'clinguard':
                s['fallback'] += k.get('confidence_level') == 'low'
                s['outputs'] += 1
        rows.append({**r, **k, 'reference_danger': case.get('reference_danger', ''),
                     'case_type': case.get('case_type', '')})
    if rows:
        write_csv(OUT / 'results.csv', rows)
    print('Note: the general translator gives no danger decision, so its recall is 0 by design.\n')
    for g in [*SYSTEMS, 'clinguard:glossary', 'clinguard:fallback']:
        s = stats.get(g)
        if not s:
            continue
        print(f'== {g}')
        print('  meaning accuracy      ', wilson(s['correct'], s['scored']))
        print('  important omissions   ', wilson(s['omission'], s['omission_scored']))
        print('  danger-sign recall    ', wilson(s['caught'], s['true_danger']))
        print('  false-alert rate      ', wilson(s['false_alert'], s['non_danger']))
        if g == 'clinguard':
            print('  fallback rate         ', wilson(s['fallback'], s['outputs']))
    print(f'\nPer-output details: {OUT / "results.csv"}. Report denominators, missing scores and limitations.')


def read_csv_any(path) -> list:
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'run':
        run(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == 'score':
        score(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
