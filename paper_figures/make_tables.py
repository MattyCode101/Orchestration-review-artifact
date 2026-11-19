# make_tables.py: lays out Tables III-VI of the paper from the tables written by final/final_tables.py (paper_figures/out/final/).
# Layout: policies grouped (Reference | orchestration policies | Direct LLM Control),
# single-line rotated headers, vertical rules between groups, bold on every row (highest gain, lowest cost), a tool-runs row
# (budget made explicit), start values next to their brief's gains, rewritten notes.
# Values are never retyped: gain, +- and cost cells are copied from the generated tables (or, for tool runs, from the cost csv)
# and checked cell by cell at the end.
# Run from the release root:  python paper_figures/make_tables.py [<generated tables> [<cost csv folder>]]
import re, sys, os, csv
SRC = os.path.join(sys.argv[1], '') if len(sys.argv) > 1 else 'paper_figures/out/final/'
COST = os.path.join(sys.argv[2], '') if len(sys.argv) > 2 else 'paper_figures/inputs/'
OUT = 'paper/tables/'
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- parsing the agent's tables
def cells_of(row):
    return [c.strip() for c in row.split('&')]

def main_and_pm(cell):
    m = re.match(r'\\shortstack\[[lc]\]\{(.*?)\\\\\{\\fontsize\{[\d.]+\}\{[\d.]+\}\\selectfont (.*?)\}\}$', cell.strip(), re.S)
    if not m: raise ValueError(cell[:120])
    main, sub = m.group(1), m.group(2)
    return main, ('' if 'phantom' in sub else sub)

def strip_fmt(s):
    s = s.replace('$^{2}$', '').replace('$^{1}$', '').replace('$^{a}$', '')
    while True:
        t = re.sub(r'\\text(?:bf|it)\{([^{}]*)\}', r'\1', s)
        if t == s: return s
        s = t

def num(s):
    s = strip_fmt(s).replace(',', '').replace('$-$', '-').replace('$\\pm$', '').strip()
    try: return float(s)
    except ValueError: return None

def parse_summary(stem):
    t = open(SRC + stem + '.tex').read()
    body = t.split('\\midrule', 1)[1].split('\\bottomrule')[0]
    rows = []
    for r in body.split('\\\\\n'):
        r = r.replace('\\midrule', '').strip()
        if not r: continue
        cs = cells_of(r.rstrip('\\').strip())
        rows.append((main_and_pm(cs[0]), [main_and_pm(c) for c in cs[1:]]))
    return rows   # source column order: Exhaustive, Direct, LLM, Rule, Fixed, Bandit, Grid

def parse_breakdown(stem):
    t = open(SRC + stem + '.tex').read()
    body = t.split('\\midrule', 1)[1].split('\\bottomrule')[0]
    out = []
    for r in body.split('\\\\\n'):
        r = r.strip()
        if r.startswith('\\midrule'):
            out.append(('rule',)); r = r[len('\\midrule'):].strip()
        if not r: continue
        cs = cells_of(r)
        if cs[0].startswith('\\multicolumn{2}'):
            out.append(('total', '', [main_and_pm(c) for c in cs[1:]])); continue
        g = re.match(r'\\multirow\{(\d+)\}\{\*\}\{\\rotatebox\[origin=c\]\{90\}\{(\w+)\}\}', cs[0])
        out.append(('row', g.group(2) if g else '', main_and_pm(cs[1])[0], [main_and_pm(c) for c in cs[2:]]))
    return out   # per row: start values, then per brief: Exhaustive, Direct, LLM, Rule, Fixed

# ---------------------------------------------------------------- shared layout pieces
PM = '\\providecommand{\\pmf}[1]{{\\fontsize{5.5}{5.5}\\selectfont\\,#1}}'
RH = '\\providecommand{\\rh}[1]{\\rotatebox{90}{\\fontsize{6.5}{7}\\selectfont\\shortstack[l]{#1}}}'
SEP = '!{\\vrule width 0.4pt}'          # thin vertical rule between policy groups
BSEP = '!{\\hspace{3pt}\\vrule width 0.8pt\\hspace{3pt}}'   # heavier rule between brief blocks
RULESEP = '\\setlength{\\aboverulesep}{0pt}\\setlength{\\belowrulesep}{0pt}\\setlength{\\extrarowheight}{1.6pt}'

def cell(main, pm, bold=False):
    m = strip_fmt(main) if bold is not None else main
    if bold: m = '\\textbf{%s}' % m
    return m + ('\\pmf{%s}' % pm if pm else '')

def bold_flags(values, better):
    """values: list of (display string or None); returns bools marking the best (ties at display precision)."""
    nums = [num(v) if v is not None else None for v in values]
    valid = [x for x in nums if x is not None]
    if not valid: return [False] * len(values)
    best = max(valid) if better == 'max' else min(valid)
    if better == 'max' and best <= 0: return [False] * len(values)
    return [x is not None and x == best for x in nums]

# ---------------------------------------------------------------- summary tables (III, V)
NEW = [0, 6, 5, 4, 3, 2, 1]   # Exhaustive | Grid, Bandit, Fixed, Rule-Based, LLM | Direct

def cost_runs(fname, keys):
    rows = {r['key']: r for r in csv.DictReader(open(COST + fname))}
    out = []
    for k in keys:
        v = rows[k]['tool_runs_per_task'] if k in rows else ''
        out.append('%.1f' % float(v) if v not in ('', None) else None)
    return out

def emit_summary(stem, label, caption, exh_name, gain_rows, runs, note, exclude_cost=(), exh_jobh=None):
    src = parse_summary(stem)
    heads = [exh_name.replace('Exhaustive Search, ', 'Exhaustive\\\\Search, ') + '$^{a}$', 'Grid\\\\Search', 'Learned\\\\Bandit', 'Fixed Worker\\\\Order$^{b}$',
             'Rule-Based\\\\Orchestration', 'LLM\\\\Orchestration$^{c}$', 'Direct LLM\\\\Control$^{c}$']
    L = ['%% %s: re-laid-out from paper_figures/out/final/%s.tex; values copied by make_tables.py (v2), not retyped.' % (label.split(':')[1], stem),
         '\\begin{table}[t]', '\\centering', '\\caption{%s}' % caption, '\\label{%s}' % label, PM, RH,
         '{\\fontsize{7.5}{8.5}\\selectfont', '\\setlength{\\tabcolsep}{2.4pt}' + RULESEP,
         '\\begin{tabular}{@{}l c%s*{5}{c}%sc@{}}' % (SEP, SEP), '\\toprule',
         ' & ' + ' & '.join('\\rh{%s}' % h for h in heads) + ' \\\\', '\\midrule']
    checks = []
    n_gain = len(gain_rows)
    # gain rows
    for i, (lab, sub) in enumerate(gain_rows):
        vals = [src[i][1][j] for j in NEW]
        fl = bold_flags([v[0] if 'not run' not in v[0] and '---' not in v[0] else None for v in vals[1:]], 'max')
        cs = [cell(vals[0][0], vals[0][1], False)] + [cell(v[0].replace('not run', 'n.r.'), v[1], f) for v, f in zip(vals[1:], fl)]
        L.append('%s {\\fontsize{6}{7}\\selectfont %s} & %s \\\\' % (lab, sub, ' & '.join(cs)))
        checks += [(strip_fmt(src[i][1][j][0]), src[i][1][j][1]) for j in NEW]
    L.append('\\midrule')
    # tool runs row (from the cost csv)
    fl = bold_flags(runs[1:], 'min')
    cs = [runs[0] or 'n/a'] + [('\\textbf{%s}' % v if f else v) for v, f in zip(runs[1:], fl)]
    L.append('Tool runs / task {\\fontsize{6}{7}\\selectfont budget $\\le$6} & %s \\\\' % ' & '.join(cs))
    # cost rows from the agent's table: job-h, tokens
    for i, (lab, sub) in [(n_gain, ('Vivado job-h / task', '')), (n_gain + 1, ('LLM tokens / task', ''))]:
        vals = [src[i][1][j] for j in NEW]
        disp = [v[0] for v in vals]
        rank = [None if ('---' in d or 'n/a' in d or k + 1 in exclude_cost) else d for k, d in enumerate(disp[1:])]
        fl = bold_flags(rank, 'min')
        d0 = exh_jobh if (i == n_gain and exh_jobh) else disp[0]
        cs = [d0.replace('---', '--')] + [('\\textbf{%s}' % strip_fmt(d) if f else d.replace('---', '--')) for d, f in zip(disp[1:], fl)]
        if i == n_gain and exclude_cost:
            cs = [c + ('$^{d}$' if k in exclude_cost else '') for k, c in enumerate(cs)]
        L.append('%s & %s \\\\' % (lab, ' & '.join(cs)))
        checks += [(strip_fmt(src[i][1][j][0]), src[i][1][j][1]) for j in NEW]
    L += ['\\bottomrule', '\\end{tabular}', '}', '\\par\\vspace{2pt}',
          '\\parbox{\\columnwidth}{\\fontsize{6}{7}\\selectfont %s}' % note, '\\end{table}']
    open(OUT + label.replace(':', '_') + '.tex', 'w').write('\n'.join(L) + '\n')
    # check: every copied cell equals the source, as a multiset per row
    mine = open(OUT + label.replace(':', '_') + '.tex').read()
    for main, pm in checks:
        if exh_jobh and strip_fmt(main) == 'n/a':
            continue  # the exhaustive job-h cell is replaced by a labelled estimate
        assert strip_fmt(main).replace('not run', 'n.r.').replace('---', '--') in mine, main
    return src

note2 = ('$^{a}$Best-known result on all 7 tasks; a reference outside the budget (682 runs per task), never bold. '
         '$^{b}$Area, then timing. $^{c}$Gemini 3.1 Flash-Lite, 3 repeats; $\\pm$: sample sd over repeats; other policies are deterministic. '
         'Gain ($\\Delta$): final pick minus start in the goal metric (WNS improvement in ps; LUTs removed); positive is better. '
         'WNS is improved, not necessarily closed. LUTs: LUTs after placement. '
         'Gains are means over the TEST designs where the exhaustive search beats run-to-run variation (3 of 5 timing-first, 4 of 5 area-first); '
         'on the others no method can show a real gain. Over all 5 designs the ranking of the policies in Table~\\ref{tab:breakdown_2worker} is unchanged. '
         'Budget: at most 6 Vivado recipe runs per task, the same for every policy. '
         'Vivado job-h: summed run time of a task\'s Vivado recipe runs (a reused result is charged its original time); '
         'for Exhaustive Search, per design, shared by both briefs. '
         'Bold: best policy per row (highest gain, lowest cost). Every final pick meets every limit.')
note3 = ('As Table~\\ref{tab:summary_2worker}, with three workers and power saved (Vivado vectorless dynamic power estimate). '
         '$^{a}$2-step search over all 14 recipes; best known on 6 of 9 tasks (a six-run policy result beats it on 3); its runs and job-h were not logged as one campaign, so they are estimated ($\\approx$) by scaling the measured 2-step search over the ten two-worker recipes (82 runs, 17.97 job-h per design) by the number of 2-step sequences, 14$\\times$15 against 10$\\times$11. '
         '$^{b}$Timing, area, then power; power, area, then timing on the power-first brief. $^{c}$As Table~\\ref{tab:summary_2worker}. '
         '$^{d}$Timing-first is post-freeze: its prior was fitted on 17 Nov from TRAINING and VALIDATION records only, after the other TEST rows were known; job-h over all 15 tasks; not ranked. '
         'Means over the 3 TEST designs per brief that beat run-to-run variation; over all 5 designs the ranking of the policies in Table~\\ref{tab:breakdown_3worker} is unchanged, except that Fixed Worker Order and Direct LLM Control swap on timing-first (81 and 83 ps).')

r2 = cost_runs('Paper_Tables_Table1_cost_per_task.csv', ['ref3', 'grid', 'bandit', 'fixed', 'v2', 'V9', 'GA-R1'])
r2[0] = '682'
r3 = cost_runs('Paper_Tables_Table2_cost_per_task.csv', ['ref2p', 'grid', 'bandit', 'fixed', 'v2', 'V9', 'GA-R1'])
r3[0] = '$\\approx$157'  # estimate: 82 runs x (14*15)/(10*11), measured 2-step over 10 recipes, TEST
s2 = emit_summary('tab_summary_2worker_TEST', 'tab:summary_2worker',
    'Two workers, TEST designs: gain over the starting checkpoint and cost per task, at an equal budget.',
    'Exhaustive Search, 3 Steps', [('WNS gain (ps)', 'timing-first'), ('LUTs removed', 'area-first')], r2, note2)
s3 = emit_summary('tab_summary_3worker_TEST', 'tab:summary_3worker',
    'Three workers (power added, no policy changed), TEST designs: gain over the starting checkpoint and cost per task.',
    'Exhaustive Search, 2 Steps', [('WNS gain (ps)', 'timing-first'), ('LUTs removed', 'area-first'), ('Power saved (mW)', 'power-first')], r3, note3,
    exclude_cost=(2,), exh_jobh='$\\approx$34.3')  # 17.97 job-h x (14*15)/(10*11)
print('runs 2W', r2); print('runs 3W', r3)

# ---------------------------------------------------------------- per-design tables (IV, VI)
ORDER = [0, 4, 3, 2, 1]   # Exhaustive | Fixed, Rule-Based, LLM | Direct  (from source Exhaustive, Direct, LLM, Rule, Fixed)

def emit_breakdown(stem, label, caption, blocks, note, font, sep):
    items = parse_breakdown(stem)
    nb = len(blocks)
    spec = 'l' + ''.join((BSEP if b else '') + 'r r' + SEP + 'rrr' + SEP + 'r' for b in range(nb))
    heads = ['Start', 'Exhaustive\\\\Search', 'Fixed Worker\\\\Order', 'Rule-Based\\\\Orch.', 'LLM\\\\Orchestration', 'Direct LLM\\\\Control']
    L = ['%% %s: re-laid-out from paper_figures/out/final/%s.tex (one line per design, start value next to its brief); values copied by make_tables.py (v2).' % (label.split(':')[1], stem),
         '\\begin{table*}[t]', '\\centering', '\\caption{%s}' % caption, '\\label{%s}' % label, PM, RH,
         '{%s' % font, '\\setlength{\\tabcolsep}{%s}' % sep + RULESEP,
         '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}%s@{}}' % spec, '\\toprule']
    hdr, rules, c = '', '', 2
    for title, _ in blocks:
        hdr += ' & \\multicolumn{6}{c}{%s}' % title
        rules += '\\cmidrule(lr){%d-%d}' % (c, c + 5); c += 6
    L += [hdr + ' \\\\', rules]
    L += ['& ' + ' & '.join('\\rh{%s}' % h for _ in blocks for h in heads) + ' \\\\', '\\midrule']
    checks = []
    for it in items:
        if it[0] == 'rule': L.append('\\midrule'); continue
        if it[0] == 'total':
            name, vals, g = 'Total (13 designs)', it[2], ''
        else:
            _, g, name, vals = it
        if g:
            L.append('\\multicolumn{%d}{@{}l}{\\fontsize{6}{6}\\selectfont\\textit{%s}} \\\\[-0.5pt]' % (1 + 6 * nb, g))
        out = []
        for b in range(nb):
            start = vals[b]
            pol = vals[nb + 5 * b: nb + 5 * b + 5]
            for v in [start] + [pol[k] for k in ORDER]:
                m = v[0].replace('$^{2}$', '').replace('$^{1}$', '')
                out.append(m + ('\\pmf{%s}' % v[1] if v[1] else ''))
                checks.append((m, v[1]))
        L.append('%s & %s \\\\' % (name, ' & '.join(out)))
    L += ['\\bottomrule', '\\end{tabular*}', '}', '\\par\\vspace{1pt}',
          '\\parbox{\\textwidth}{\\fontsize{6}{7}\\selectfont %s}' % note, '\\end{table*}']
    open(OUT + label.replace(':', '_') + '.tex', 'w').write('\n'.join(L) + '\n')
    # check: the emitted data cells, in order, equal the source cells re-ordered (value and +- identical)
    src = [x for x in items if x[0] != 'rule']
    expect = []
    for x in src:
        vals = x[2] if x[0] == 'total' else x[3]
        for b in range(nb):
            pol = vals[nb + 5 * b: nb + 5 * b + 5]
            for v in [vals[b]] + [pol[k] for k in ORDER]:
                expect.append((v[0].replace('$^{2}$', '').replace('$^{1}$', ''), v[1]))
    assert expect == checks
    mine = open(OUT + label.replace(':', '_') + '.tex').read().split('\\midrule', 1)[1].split('\\bottomrule')[0]
    got = []
    for line in mine.split('\n'):
        if '&' not in line: continue
        for cc in [x.strip() for x in line.rstrip('\\ ').split('&')][1:]:
            m = re.match(r'(.*?)(\\pmf\{(.*)\})?$', cc)
            got.append((m.group(1), m.group(3) or ''))
    print(stem, 'cells', len(expect), 'identical:', got == expect)
    assert got == expect

nb2 = ('Two workers; TRAINING (5 designs), VALIDATION (3) and TEST (5), ordered by start LUTs; budget of 6 Vivado recipe runs per task. '
       'Start: the starting checkpoint (Vivado default implementation); WNS in ps (negative = failing timing). '
       'Gain ($\\Delta$): improvement of the final pick over the start in the brief\'s goal (WNS in ps; LUTs removed), mean over repeats; '
       '$\\pm$: sample sd over 3 repeats (LLM Orchestration, Direct LLM Control; Gemini 3.1 Flash-Lite); other policies deterministic. '
       'Exhaustive Search: the 3-step search (TRAINING: stored depth-3 state; VALIDATION: 2-step only; on mcml area-first a policy run reached 1,625 LUTs, above its 1,595); a reference, never bold. '
       'Fixed Worker Order: area, then timing. '
       'Italic: no headroom, i.e. the exhaustive search\'s best gain is within the design\'s run-to-run variation, so no gain there can be told apart from noise. '
       'Bold: best policy per row and brief (ties all bold; none where no policy gained). Total: sum over all 13 designs including italic ones; $\\pm$ from per-repeat sums.')
nb3 = ('Three workers; as Table~\\ref{tab:breakdown_2worker}, with power in mW (Vivado vectorless dynamic power estimate). '
       'Exhaustive Search: the 2-step search over 14 recipes (TRAINING: the fill run\'s 2-step search). '
       'Fixed Worker Order: timing, area, then power; power, area, then timing on power-first. '
       'Italic values can be large when a design\'s variation is large (digit-recognition power varies by about 300 mW between runs that should not differ).')
emit_breakdown('tab_breakdown_2worker_all', 'tab:breakdown_2worker',
    'Two workers, per design (TRAINING, VALIDATION and TEST): start value and gain of each policy over the starting checkpoint.',
    [('Timing-first: WNS start and gain (ps)', ''), ('Area-first: LUTs at start and removed', '')], nb2, '\\fontsize{7}{8}\\selectfont', '1.5pt')
emit_breakdown('tab_breakdown_3worker_all', 'tab:breakdown_3worker',
    'Three workers, per design (TRAINING, VALIDATION and TEST): start value and gain of each policy over the starting checkpoint.',
    [('Timing-first: WNS start and gain (ps)', ''), ('Area-first: LUTs at start and removed', ''), ('Power-first: power at start and saved (mW)', '')], nb3, '\\fontsize{6.3}{7.3}\\selectfont', '0.9pt')
print('done')
