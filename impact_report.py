"""
Dated PDF snapshot of the Impact page, for sharing with reviewers.

Built only from the same anonymous totals as the Impact page (impact.compute_stats):
counts, denominators, interpretation pathways, software versions and definitions.
No transcripts, notes, comments or patient information are ever included.
"""
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF

from impact import MIN_RATED_FOR_PCT, MODE_TEXT, PATHWAYS, TEST_DEPARTMENT, _pct

GREEN = (45, 106, 79)
DARK = (26, 58, 42)
GREY = (90, 90, 90)
PALE = (232, 245, 237)

# The built-in PDF fonts only cover Latin-1, so swap the few symbols we use
_SAFE = {'≥': '>=', '–': '-', '—': '-', '→': '->', '“': '"', '”': '"', '’': "'", '‘': "'", '…': '...'}


def _t(text) -> str:
    text = str(text)
    for a, b in _SAFE.items():
        text = text.replace(a, b)
    return text.encode('latin-1', 'replace').decode('latin-1')


class _Report(FPDF):
    def __init__(self, generated: str, fingerprint: str):
        super().__init__(format='A4')
        self.generated, self.fingerprint = generated, fingerprint
        self.set_margins(18, 16, 18)
        self.set_auto_page_break(True, margin=18)

    def footer(self):
        self.set_y(-13)
        self.set_font('Helvetica', '', 7.5)
        self.set_text_color(*GREY)
        self.cell(0, 5, _t(f'BeaGrace Health pilot report - generated {self.generated} - '
                           f'data fingerprint {self.fingerprint} - page {self.page_no()}/{{nb}}'),
                  align='C')

    # helpers
    def h1(self, text):
        self.set_font('Helvetica', 'B', 18)
        self.set_text_color(*DARK)
        self.multi_cell(0, 9, _t(text), new_x='LMARGIN', new_y='NEXT')

    def h2(self, text):
        self.ln(4)
        self.set_font('Helvetica', 'B', 12.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 7, _t(text), new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(*PALE)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def para(self, text, size=9.5, color=(30, 30, 30), style=''):
        self.set_font('Helvetica', style, size)
        self.set_text_color(*color)
        self.multi_cell(0, 5, _t(text), new_x='LMARGIN', new_y='NEXT')
        self.ln(1)

    def table(self, header, rows, widths):
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(*PALE)
        self.set_text_color(30, 30, 30)
        for h, w in zip(header, widths):
            self.cell(w, 7, _t(h), border=0, fill=True)
        self.ln()
        self.set_font('Helvetica', '', 9)
        for row in rows:
            for v, w in zip(row, widths):
                self.cell(w, 6.5, _t(v), border='B')
            self.ln()
        self.ln(2)

    def callout(self, title, text):
        self.set_fill_color(*PALE)
        self.set_draw_color(*GREEN)
        self.set_line_width(0.8)
        y = self.get_y()
        self.set_x(self.l_margin + 3)
        self.set_font('Helvetica', 'B', 9.5)
        self.set_text_color(*DARK)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 3, 5.5, _t(title), fill=True, new_x='LMARGIN', new_y='NEXT')
        self.set_x(self.l_margin + 3)
        self.set_font('Helvetica', '', 9)
        self.set_text_color(30, 30, 30)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 3, 5, _t(text), fill=True, new_x='LMARGIN', new_y='NEXT')
        self.line(self.l_margin + 1, y, self.l_margin + 1, self.get_y())
        self.ln(3)


def build_report(stats: dict) -> str:
    """Write the PDF to a temp file and return its path."""
    now = datetime.now(timezone.utc)
    generated = now.strftime('%Y-%m-%d %H:%M UTC')
    pdf = _Report(generated, stats.get('fingerprint', 'n/a'))
    pdf.alias_nb_pages()
    pdf.add_page()

    pdf.h1('BeaGrace Health - Pilot Impact Report')
    pdf.para('Ekiti Yoruba speech to structured clinical nurse handoff note, '
             'with ClinGuard glossary matching and rule-based danger-sign checks.', color=GREY)
    pdf.table(['Item', 'Value'], [
        ['Reporting period', f"{stats['period_start']} to {stats['period_end']} (UTC)"],
        ['Report generated', generated],
        ['Type of use', f'{MODE_TEXT[0]}. {MODE_TEXT[1]}'],
        ['Data included', f"Pilot use only. The '{TEST_DEPARTMENT}' department is excluded."],
        ['Records counted up to', stats.get('counted_until', '')],
        ['Data fingerprint', f"{stats.get('fingerprint', 'n/a')} (identifies the exact records counted)"],
    ], [45, 129])

    pdf.callout('Nurse acceptance is not verified accuracy',
                'The ratings in this report are nurses\' own views of each note (thumbs up / '
                'thumbs down). They show whether nurses found the notes acceptable in use. They '
                'have not been checked against independent reference answers and must not be read '
                'as clinical accuracy. Accuracy is reported only in Section 4, once an independent '
                'evaluation has been carried out.')

    # 1. Usage
    pdf.h2(f'1. Operational pilot ({MODE_TEXT[0].lower()}): usage')
    depts = ', '.join(f'{n} ({c})' for n, c in stats['departments']) or 'not recorded yet'
    pdf.table(['Measure', 'Count', 'Note'], [
        ['Attempts (audio submitted)', stats['attempts'], 'all outcomes'],
        ['Clinical notes generated', stats['notes'], f"over {stats['days_active']} day(s)"],
        ['No speech detected', stats['no_speech'], ''],
        ['Errors', stats['errors'], 'technical failures'],
        ['Nurses using the app', stats['nurses_using'] or 'not recorded', 'pseudonymous codes'],
        ['Nurses who rated notes', stats['nurses_rating'] or 'not recorded', 'pseudonymous codes'],
        ['Participating facilities', stats['facilities'] or 'not recorded',
         f"codes only; {stats['facility_notes']} of {stats['notes']} notes coded"],
        ['Clinical departments', len(stats['departments']) or 'not recorded', ''],
        ['Phrases in the glossary', stats.get('glossary_size', ''),
         f"{stats['phrases_learned']} reviewer-approved from nurses"],
    ], [70, 30, 74])
    pdf.para(f'Departments: {depts}', size=8.5, color=GREY)

    # 2. Feedback
    pdf.h2('2. Nurse feedback (with denominators)')
    rated = stats['rated']
    pct = (f"{_pct(stats['accepted'], rated)}%" if rated >= MIN_RATED_FOR_PCT
           else f'not calculated (fewer than {MIN_RATED_FOR_PCT} rated notes)')
    pdf.table(['Measure', 'Value'], [
        ['Notes generated', stats['notes']],
        ['Notes rated (thumbs up or down)', f"{rated} of {stats['notes']}"],
        ['Accepted (thumbs up)', f"{stats['accepted']} of {rated} rated"],
        ['Needs changes (thumbs down)', f"{stats['corrections']} of {rated} rated"],
        ['Not sure (not counted either way)', stats['not_sure']],
        ['Acceptance rate', pct],
    ], [90, 84])

    # 3. Pathways
    pdf.h2('3. How phrases were interpreted (ClinGuard)')
    total = stats['notes_with_level']
    if total:
        rows = []
        for key, _ in PATHWAYS:
            p = stats['pathways'][key]
            r = p['correct'] + p['needs_changes']
            rows.append([p['label'], p['notes'], f"{_pct(p['notes'], total)}%",
                         f"{p['correct']} of {r} accepted" if r else 'not rated yet'])
        pdf.table(['Interpretation path', 'Notes', 'Share', 'Nurse ratings'], rows, [64, 20, 20, 70])
        pdf.para(f'Out of {total} notes with interpretation data (recorded from 2 Nov 2026).',
                 size=8.5, color=GREY)
    else:
        pdf.para('No notes with interpretation data yet (recorded from 2 Nov 2026).')
    pdf.table(['Safety check (rule-based, runs without the AI)', 'Count'], [
        ['Notes with a danger-sign alert', f"{stats['danger_notes']} of {stats['notes']}"],
        ['Danger rule matches', stats['danger_rules_fired']],
    ], [130, 44])

    # 4. Independent evaluation
    pdf.h2('4. Independent evaluation')
    pdf.para('Not yet reported. Accuracy, clinically significant errors, missed danger signs and the '
             'comparison with a general translator and an LLM-only baseline will be reported only '
             'after qualified Yoruba-speaking and clinical reviewers have set reference answers and '
             'a blinded, controlled comparison has been run.', style='')
    pdf.para(f"In progress: {stats['eval_assessments']} assessments by {stats['eval_evaluators']} nurse "
             f"evaluators across {stats['eval_scenarios']} standardised scenarios. These are recorded "
             'separately from pilot use and are not included in Sections 1-3.', size=9)

    # 5. Versions
    pdf.h2('5. Software versions recorded with each interaction')
    pdf.para('Each interaction stores the versions in use when it happened, so results are '
             'attributed to the release that produced them, not to the version deployed today.',
             size=8.5, color=GREY)
    names = {'app_version': 'BeaGrace app', 'code_fingerprint': 'Code fingerprint',
             'clinguard_version': 'ClinGuard package', 'glossary_version': 'Glossary fingerprint',
             'rules_version': 'Danger rules'}
    rows = []
    for k, entries in stats['versions'].items():
        if not entries:
            rows.append([names[k], 'not recorded', '', ''])
        for i, e in enumerate(entries):
            rows.append([names[k] if i == 0 else '', e['version'], e['attempts'],
                         f"{e['first']} to {e['last']}"])
    pdf.table(['Component', 'Version', 'Attempts', 'Dates'], rows, [42, 52, 22, 58])
    if stats.get('versions_unrecorded'):
        pdf.para(f"{stats['versions_unrecorded']} earlier attempts have no version recorded.",
                 size=8.5, color=GREY)

    # 6. Definitions
    pdf.h2('6. Definitions')
    for term, text in [
        ('Attempt', 'One audio submitted: a note made, no speech heard, or an error.'),
        ('Note generated', 'A clinical handoff note was produced.'),
        ('Rated', f'A nurse tapped thumbs up or thumbs down. "Not sure" is shown separately and not '
                  f'counted either way. A percentage is shown only from {MIN_RATED_FOR_PCT} rated notes.'),
        ('Interpretation path', 'Each note has exactly one path, set by the best glossary match for '
                                'the whole transcript: glossary match (score >= 90%), partial match '
                                '(75-89%), AI fallback (no glossary phrase scored 75% or more, so the AI '
                                'translated the whole transcript), or AI fallback: not understood (the AI fallback was tried but could not '
                                'translate it). A note is never counted in two paths, so the shares add '
                                'up to 100%. When a glossary phrase matches, the AI translation is not '
                                'used for the rest of what was said.'),
        ('Danger-sign alert', 'The rule-based safety check (no AI) flagged the note for escalation.'),
        ('Nurses', 'Distinct pseudonymous nurse codes. Notes without a code are not counted.'),
        ('Facilities', 'Distinct facility codes (e.g. FAC001) issued by the pilot lead, who keeps the '
                       'code-to-facility list privately. Facility names are never stored.'),
        ('Data fingerprint', 'A short hash of the record identifiers and outcomes counted in this '
                             'report. An authorised reviewer can recompute it from the private logs '
                             '(python impact_report.py --verify <records counted up to>) to confirm '
                             'the report matches them. It contains no patient information.'),
    ]:
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(38, 5, _t(term))
        pdf.set_font('Helvetica', '', 9)
        pdf.multi_cell(0, 5, _t(text), new_x='LMARGIN', new_y='NEXT')
        pdf.ln(0.5)

    pdf.ln(3)
    pdf.para('Privacy: this report contains only anonymous totals. No audio, transcripts, notes, '
             'comments, names or other patient or nurse information are included. The underlying '
             'records are kept in a private dataset and are available to authorised reviewers on request.',
             size=8.5, color=GREY)

    path = Path(tempfile.mkdtemp()) / f"beagrace-pilot-report-{now.strftime('%Y-%m-%d')}.pdf"
    pdf.output(str(path))
    return str(path)


if __name__ == '__main__':
    # Re-check a report: python impact_report.py --verify 2026-11-06T17:09:41+00:00
    import sys
    from impact import compute_stats
    if len(sys.argv) == 3 and sys.argv[1] == '--verify':
        stats = compute_stats(0, until=sys.argv[2])
        print(f"Records up to {sys.argv[2]}: fingerprint {stats['fingerprint']}, "
              f"{stats['attempts']} attempts, {stats['notes']} notes, {stats['rated']} rated")
    else:
        print('Usage: python impact_report.py --verify <records counted up to>')
