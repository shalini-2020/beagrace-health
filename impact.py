"""
Public Impact page: live totals from the anonymous usage log.

Only real pilot use is counted:
  - records on or after PILOT_START, in UTC: the start of real nurse use,
    2026-11-06T19:00 (everything before was testing; see CHANGELOG.md). It can be
    overridden with a PILOT_START variable in the Space settings (YYYY-MM-DD or
    YYYY-MM-DDTHH:MM)
  - the 'Testing (not real use)' department is always left out
Only totals are shown. No individual notes, phrases or comments.

The page has two parts:
  - Operational pilot: what happened in day-to-day use (counts, with their denominators)
  - Independent evaluation: accuracy against reference answers. Left unreported until
    qualified reviewers have labelled cases and run the blinded comparison.
"""
import hashlib
import os
import time
from collections import Counter
from datetime import datetime, timezone

from usage_log import load_records

PILOT_START = os.environ.get('PILOT_START', '2026-11-06T19:00')
TEST_DEPARTMENT = 'Testing (not real use)'
_CACHE = {'at': 0.0, 'stats': None}
CACHE_SECONDS = 600
# How the app is being used (also recorded with every interaction as use_mode):
# 'simulated' = nurses acting out patient scenarios, no real patients; 'clinical' = real patients.
PILOT_MODE = os.environ.get('PILOT_MODE', 'simulated').strip().lower()
MODE_TEXT = {
    'simulated': ('Simulated-use pilot',
                  'Nurses acting out patient scenarios at the participating facility. No real patients.'),
    'clinical': ('Clinical pilot', 'Day-to-day use by nurses with patients.'),
}.get(PILOT_MODE, ('Pilot', ''))
# Below this many rated notes, show the count only, not a percentage
MIN_RATED_FOR_PCT = 20

# ClinGuard interpretation pathways (confidence_level in the usage log)
PATHWAYS = [
    ('high', 'Glossary match (≥90%)'),
    ('partial', 'Partial glossary match'),
    ('low', 'AI fallback: translated'),
    ('none', 'AI fallback: not understood'),
]


def _is_real(record: dict) -> bool:
    return (record.get('time_utc', '') >= PILOT_START
            and record.get('department') != TEST_DEPARTMENT)


def _eval_progress(in_period) -> dict:
    recs = [r for r in load_records('evaluations') if in_period(r)]
    return {'eval_assessments': len(recs),
            'eval_evaluators': len({r.get('evaluator_id') for r in recs if r.get('evaluator_id')}),
            'eval_scenarios': len({r.get('scenario_id') for r in recs if r.get('scenario_id')})}


def _pct(part: int, whole: int):
    return round(100 * part / whole) if whole else None


def compute_stats(glossary_size: int, fresh: bool = False, until: str = None) -> dict:
    """until: optional ISO UTC time; only records up to then are counted (used to
    re-check a dated report's fingerprint later). Implies fresh."""
    fresh = fresh or bool(until)
    if not fresh and _CACHE['stats'] and time.time() - _CACHE['at'] < CACHE_SECONDS:
        stats = dict(_CACHE['stats'])
        stats['glossary_size'] = glossary_size
        return stats

    until = until or datetime.now(timezone.utc).isoformat(timespec='seconds')
    in_period = lambda r: r.get('time_utc', '') <= until
    usage = [r for r in load_records('usage') if _is_real(r) and in_period(r)]
    outcomes = Counter(r.get('outcome') for r in usage)
    notes = [r for r in usage if r.get('outcome') == 'note']
    note_by_id = {r['note_id']: r for r in notes if r.get('note_id')}

    # One rating per note: keep the latest if a note was somehow rated twice
    latest = {}
    for f in sorted(load_records('feedback'), key=lambda f: f.get('time_utc', '')):
        if f.get('note_id') in note_by_id and in_period(f):
            latest[f['note_id']] = f
    feedback = list(latest.values())
    ratings = Counter(f.get('rating') for f in feedback)

    # Ratings by interpretation pathway
    by_path = {}
    with_level = [r for r in notes if r.get('confidence_level')]
    for key, label in PATHWAYS:
        ids = {r['note_id'] for r in with_level if r.get('confidence_level') == key}
        path_fb = [f for f in feedback if f['note_id'] in ids]
        c = Counter(f.get('rating') for f in path_fb)
        by_path[key] = {'label': label, 'notes': len(ids),
                        'correct': c['correct'], 'needs_changes': c['needs_changes'],
                        'not_sure': c['not_sure']}

    reviews = [r for r in load_records('reviews')
               if r.get('decision') == 'approved' and r.get('time_utc', '') >= PILOT_START
               and in_period(r)]
    departments = Counter(r.get('department') for r in notes
                          if r.get('department') not in (None, '', 'Not specified'))
    days = sorted({r['time_utc'][:10] for r in notes if r.get('time_utc')})
    nurse_codes_using = {r.get('nurse_code') for r in notes if r.get('nurse_code')}
    nurse_codes_rating = {f.get('nurse_code') for f in feedback if f.get('nurse_code')}
    facilities = Counter(r.get('facility_code') for r in notes if r.get('facility_code'))

    # Software versions as RECORDED WITH EACH INTERACTION (not the version deployed now),
    # with how many attempts each produced and over which dates. Latest first.
    def _versions(key):
        found = {}
        for r in usage:
            v, t = r.get(key), r.get('time_utc', '')[:10]
            if not v:
                continue
            f = found.setdefault(v, {'version': v, 'attempts': 0, 'first': t, 'last': t})
            f['attempts'] += 1
            f['first'], f['last'] = min(f['first'], t), max(f['last'], t)
        return sorted(found.values(), key=lambda f: f['last'], reverse=True)

    # Fingerprint of exactly which records were counted, so a dated report can later be
    # checked against the private logs by an authorised reviewer (ids only, no content)
    counted = sorted(f"u:{r.get('note_id')}:{r.get('outcome')}" for r in usage)
    counted += sorted(f"f:{f.get('note_id')}:{f.get('rating')}" for f in feedback)
    fingerprint = hashlib.sha256('\n'.join(counted).encode('utf-8')).hexdigest()[:16]

    stats = {
        'fingerprint': fingerprint,
        'counted_until': until,
        'feedback_records': len(feedback),
        'period_start': PILOT_START.replace('T', ' '),
        'period_end': until[:10],
        'attempts': len(usage),
        'notes': len(notes),
        'no_speech': outcomes['no_speech'],
        'errors': outcomes['error'],
        'days_active': len(days),
        'first_day': days[0] if days else None,
        'departments': departments.most_common(),
        'rated': ratings['correct'] + ratings['needs_changes'],   # 'not sure' counted separately
        'accepted': ratings['correct'],
        'corrections': ratings['needs_changes'],
        'not_sure': ratings['not_sure'],
        'nurses_using': len(nurse_codes_using),
        'nurses_rating': len(nurse_codes_rating),
        'facilities': len(facilities),
        'facility_notes': sum(facilities.values()),
        'notes_with_level': len(with_level),
        'pathways': by_path,
        'danger_notes': sum(1 for r in notes if r.get('danger_sign')),
        'danger_rules_fired': sum(len(r.get('danger_rule_ids') or []) for r in notes),
        'phrases_learned': len(reviews),
        # Independent evaluation progress: counts only, never results (see nurse_eval.py)
        **_eval_progress(in_period),
        'versions': {k: _versions(k) for k in
                     ('app_version', 'code_fingerprint', 'clinguard_version', 'glossary_version',
                      'rules_version')},
        'versions_unrecorded': sum(1 for r in usage if not r.get('app_version')),
    }
    _CACHE.update(at=time.time(), stats=stats)
    stats = dict(stats)
    stats['glossary_size'] = glossary_size
    return stats


def _card(number, label, note=''):
    note_html = f'<div class="impact-note">{note}</div>' if note else ''
    return (f'<div class="impact-card"><div class="impact-number">{number}</div>'
            f'<div class="impact-label">{label}</div>{note_html}</div>')


def _section(title, sub, body):
    return (f'<div class="impact-section"><div class="impact-section-title">{title}</div>'
            f'<div class="impact-sub">{sub}</div>{body}</div>')


def _acceptance_card(stats):
    rated, accepted = stats['rated'], stats['accepted']
    if not rated:
        return _card('–', 'Accepted by nurses', f"no ratings yet · {stats['notes']} notes generated")
    note = f"{rated} rated of {stats['notes']} notes generated"
    if stats['not_sure']:
        note += f" · {stats['not_sure']} marked not sure"
    if rated >= MIN_RATED_FOR_PCT:
        note += f" · {_pct(accepted, rated)}% accepted"
    else:
        note += f" · too few ratings for a percentage (shown from {MIN_RATED_FOR_PCT})"
    return _card(f'{accepted} of {rated}', 'Rated notes accepted by nurses',
                 note + ' · nurse view, not verified accuracy')


def _pathway_table(stats):
    total = stats['notes_with_level']
    if not total:
        return ('<div class="impact-note">No notes with interpretation data yet '
                '(recorded for notes from 2 Nov 2026).</div>')
    rows = ''
    for key, _ in PATHWAYS:
        p = stats['pathways'][key]
        rated = p['correct'] + p['needs_changes']
        rating_txt = f"{p['correct']} of {rated} accepted" if rated else 'not rated yet'
        if p['not_sure']:
            rating_txt += f", {p['not_sure']} not sure"
        rows += (f"<tr><td>{p['label']}</td><td>{p['notes']}</td>"
                 f"<td>{_pct(p['notes'], total)}%</td><td>{rating_txt}</td></tr>")
    return (f'<table class="impact-table"><thead><tr><th>How the phrase was interpreted</th>'
            f'<th>Notes</th><th>Share</th><th>Nurse ratings</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
            f'<div class="impact-note">Out of {total} notes with interpretation data.</div>')


VERSION_NAMES = {'app_version': 'App', 'code_fingerprint': 'Code fingerprint',
                 'clinguard_version': 'ClinGuard', 'glossary_version': 'Glossary',
                 'rules_version': 'Danger rules'}


def version_text(entries: list) -> str:
    return '; '.join(f"{e['version']} ({e['attempts']} attempts, {e['first']} to {e['last']})"
                     for e in entries) or 'not recorded'


def _versions_line(stats):
    parts = [f"<br>{VERSION_NAMES[k]}: {version_text(v)}" for k, v in stats['versions'].items() if v]
    if stats.get('versions_unrecorded'):
        parts.append(f"<br>{stats['versions_unrecorded']} earlier attempts have no version recorded")
    return ''.join(parts) if parts else 'not recorded yet'


def impact_html(stats: dict) -> str:
    dash = '–'
    nurses_note = ('from pseudonymous nurse codes, never names' if stats['nurses_using']
                   else 'not recorded yet · shown once nurses enter their codes')
    attempts_note = (f"{stats['notes']} notes · {stats['no_speech']} no speech · "
                     f"{stats['errors']} errors")

    pilot_cards = ''.join([
        _card(stats['notes'], 'Clinical notes generated',
              f"over {stats['days_active']} day(s) · {stats['attempts']} attempts in total"
              if stats['notes'] else 'pilot starting'),
        _acceptance_card(stats),
        _card(stats['corrections'], 'Notes marked "needs changes"'),
        _card(stats['nurses_using'] or dash, 'Nurses using the app', nurses_note),
        _card(stats['nurses_rating'] or dash, 'Nurses who rated notes', 'unique nurse codes'),
        _card(stats['facilities'] or dash, 'Participating facilities',
              f"from facility codes · {stats['facility_notes']} of {stats['notes']} notes have a code"
              if stats['facilities'] else 'not recorded yet · shown once facility codes are entered'),
        _card(len(stats['departments']) or dash, 'Clinical departments',
              'with a department selected' if stats['departments'] else 'not recorded yet'),
        _card(stats['glossary_size'], 'Phrases in the glossary',
              f"{stats['phrases_learned']} learned from nurses, reviewer-approved"),
        _card(stats['attempts'], 'All attempts', attempts_note),
    ])
    if stats['departments']:
        dept_rows = ''.join(f'<li><b>{name}</b> · {count} note{"s" if count != 1 else ""}</li>'
                            for name, count in stats['departments'])
        pilot_cards_extra = f'<div class="impact-depts"><b>By department</b><ul>{dept_rows}</ul></div>'
    else:
        pilot_cards_extra = ''

    clinguard_cards = ''.join([
        _card(stats['danger_notes'], 'Notes with a danger-sign alert',
              f"{stats['danger_rules_fired']} rule matches · safety check runs without the AI"),
    ])

    pilot = _section(
        f'Operational pilot: {MODE_TEXT[0].lower()}',
        f"{MODE_TEXT[1]} From {stats['period_start']} to {stats['period_end']}. "
        'Test use is excluded. These show activity and nurse opinion, not clinical accuracy.',
        f'<div class="impact-grid">{pilot_cards}</div>{pilot_cards_extra}')

    clinguard = _section(
        'How phrases were interpreted (ClinGuard)',
        'Which path each note took, and how nurses rated notes on each path.',
        f'{_pathway_table(stats)}<div class="impact-grid" style="margin-top:12px">{clinguard_cards}</div>')

    evaluation = _section(
        'Independent evaluation',
        '',
        '<div class="impact-pending"><b>Not yet reported.</b> Accuracy, missed danger signs and '
        'comparison with a general translator and an LLM alone will be reported only after '
        'qualified Yoruba-speaking and clinical reviewers have set reference answers and run a '
        'blinded, controlled comparison.'
        f"<br><br><b>In progress:</b> {stats['eval_assessments']} assessments by "
        f"{stats['eval_evaluators']} nurse evaluators across {stats['eval_scenarios']} standardised "
        'scenarios. These are kept separate from pilot use and are not counted above.</div>')

    definitions = f"""
        <details class="impact-defs"><summary>Definitions and versions</summary>
        <ul>
          <li><b>Attempt</b>: one audio submitted (note made, no speech heard, or an error).</li>
          <li><b>Note generated</b>: a clinical note was produced.</li>
          <li><b>Rated</b>: a nurse tapped 👍 or 👎 on the note. 🤷 Not sure is shown separately
              and is not counted either way. Percentages are shown only from {MIN_RATED_FOR_PCT} rated notes.</li>
          <li><b>Interpretation path</b>: each note has exactly one path, set by the best glossary
              match for the whole transcript: <b>glossary match</b> (score ≥90%), <b>partial match</b>
              (75-89%), <b>AI fallback</b> (no glossary phrase scored 75% or more, so the AI translated
              the whole transcript), or <b>AI fallback: not understood</b> (the AI fallback was tried but could not
              translate it, so the nurse was asked to repeat). A note is
              never counted in two paths, so the shares add up to 100%. When a glossary phrase matches,
              the AI translation is not used for the rest of what was said.</li>
          <li><b>Danger-sign alert</b>: the rule-based safety check (no AI) flagged the note for escalation.</li>
          <li><b>Nurses</b>: distinct pseudonymous nurse codes. Notes without a code are not counted.</li>
          <li><b>Facilities</b>: distinct facility codes (e.g. FAC001) issued by the pilot lead, who keeps
              the code-to-facility list privately. Notes without a code are not counted.</li>
          <li><b>Software versions</b> are those recorded with each interaction at the time it
              happened, not the version deployed now: {_versions_line(stats)}</li>
        </ul></details>"""

    return f"""
    <div class="impact-wrap">
        <div class="impact-head">
            <div class="impact-title">📊 Impact so far</div>
            <div class="impact-sub">Reporting period {stats['period_start']} to {stats['period_end']}.
            Updated about every 10 minutes. No patient information is shown.</div>
        </div>
        {pilot}
        {clinguard}
        {evaluation}
        {definitions}
    </div>"""
