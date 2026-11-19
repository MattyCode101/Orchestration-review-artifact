"""Table model, LaTeX emitter and matplotlib preview of the final results tables.

One model (Table) feeds both outputs, so the preview shows the content of the .tex file. No colour anywhere: black text, grey only for second lines.
Bold is reserved for the best policy value; headers and labels are never bold. The .tex files are plain ASCII (symbols are written as TeX macros) and need
booktabs, graphicx and multirow. THEY ARE NOT COMPILED HERE (no TeX engine on the compute server or on the laptop); the preview is a matplotlib drawing of the same model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import RendererAgg  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

SERIF = next((n for n in ("Liberation Serif", "Nimbus Roman", "Times New Roman", "DejaVu Serif") if any(f.name == n for f in fm.fontManager.ttflist)), "DejaVu Serif")
FALLBACK = "DejaVu Serif"
GREY = "#555555"
LINE_SPACING = 1.2
TABCOLSEP_PT = 2.6           # per side
ROW_PAD_PT = 1.6             # above and below the text block of a row (booktabs rows: \extrarowheight-like)


@dataclass
class Cell:
    main: str = ""
    sub: str = ""
    bold: bool = False
    italic: bool = False
    sup: str = ""                # superscript marker after the main text ("2", "†", "a")
    span: int = 1
    rowspan: int = 1
    align: str = "c"             # l c r
    rot: bool = False            # rotate by 90 degrees (header text reads bottom to top)
    lines: list[str] | None = None   # multi-line text (headers); replaces main
    two: bool = False            # second line reserved (body cells of a two-line row)
    covered: bool = False        # position covered by a rowspan above
    small: bool = False          # main text at the small size (spanner and note-like labels)
    meta: dict | None = None     # provenance record for the values csv


@dataclass
class Table:
    stem: str
    caption: str
    label: str
    ncols: int
    aligns: list[str]            # per column l/c/r
    items: list                  # ("top",) ("mid",) ("bottom",) ("cmid", [(a, b), ...]) ("row", [Cell, ...])
    note: str
    width_in: float
    font_main: float
    font_sub: float
    font_note: float = 6.5
    title_no: str = ""
    records: list = field(default_factory=list)


# ----------------------------------------------------------------------------------------------------------------- text measurement
_R = RendererAgg(72, 72, 72)       # 1 px = 1 pt


def _prop(size, bold=False, italic=False, family=None):
    return fm.FontProperties(family=family or SERIF, size=size, weight="bold" if bold else "normal", style="italic" if italic else "normal")


def text_w(s: str, size: float, bold=False, italic=False) -> float:
    if not s:
        return 0.0
    w, _, _ = _R.get_text_width_height_descent(s, _prop(size, bold, italic), ismath=False)
    return float(w)


def cell_lines(c: Cell) -> list[tuple[str, float, bool, bool, str]]:
    """(text, size_marker, bold, italic, colour) per line: size_marker 'm' main or 's' sub."""
    out = []
    if c.lines:
        for ln in c.lines:
            out.append((ln, "m", c.bold, c.italic, "black"))
        return out
    out.append((c.main + (c.sup or ""), "m", c.bold, c.italic, "black"))
    if c.two or c.sub:
        out.append((c.sub, "s", False, c.italic, GREY))
    return out


def cell_dims(c: Cell, fm_: float, fs: float) -> tuple[float, float]:
    """(width, height) in pt of the text block of a cell, before rotation."""
    ls = cell_lines(c)
    w = 0.0
    for t, k, b, i, _ in ls:
        sz = fm_ if k == "m" else fs
        if c.small and k == "m":
            sz = fs
        w = max(w, text_w(t, sz, b, i))
    h = sum((fm_ if (k == "m" and not c.small) else fs) * LINE_SPACING for _, k, *_ in ls)
    return w, h


# ----------------------------------------------------------------------------------------------------------------- layout
def layout(T: Table):
    """Column widths (pt, including padding and the share of the extra width), row heights (pt), and the placement of every cell."""
    fm_, fs = T.font_main, T.font_sub
    n = T.ncols
    minw = [0.0] * n
    spans = []
    rows = [it[1] for it in T.items if it[0] == "row"]
    for r in rows:
        j = 0
        for c in r:
            if c.covered:
                j += c.span
                continue
            w, h = cell_dims(c, fm_, fs)
            need = (h if c.rot else w) + 2 * TABCOLSEP_PT
            if c.span == 1:
                minw[j] = max(minw[j], need)
            else:
                spans.append((j, c.span, need))
            j += c.span
    for j, sp, need in sorted(spans, key=lambda x: x[1]):
        have = sum(minw[j:j + sp])
        if have < need:
            for k in range(j, j + sp):
                minw[k] += (need - have) / sp
    target = T.width_in * 72.0
    nat = sum(minw)
    extra = target - nat
    colw = [w + extra / n for w in minw] if extra > 0 else list(minw)
    heights = []
    row_cells = []
    for r in rows:
        h = 0.0
        j = 0
        for c in r:
            if c.covered:
                j += c.span
                continue
            w, hh = cell_dims(c, fm_, fs)
            if not c.rot and c.rowspan == 1:
                h = max(h, hh + 2 * ROW_PAD_PT)
            elif c.rot and c.rowspan == 1:
                h = max(h, w + 2 * ROW_PAD_PT)
            j += c.span
        heights.append(h)
    # rowspan needs
    ri = 0
    for r in rows:
        j = 0
        for c in r:
            if not c.covered and c.rowspan > 1:
                w, hh = cell_dims(c, fm_, fs)
                need = (w if c.rot else hh) + 2 * ROW_PAD_PT
                have = sum(heights[ri:ri + c.rowspan])
                if have < need:
                    heights[ri + c.rowspan - 1] += need - have
            j += c.span
        ri += 1
    return {"colw": colw, "natural": nat, "extra": extra, "rowh": heights, "target": target}


def rule_pt(kind: str) -> float:
    return {"top": 0.8, "bottom": 0.8, "mid": 0.5, "cmid": 0.5}[kind]


# ----------------------------------------------------------------------------------------------------------------- preview
def wrap(text: str, width_pt: float, size: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if text_w(t, size) <= width_pt or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_text(ax, x, y, s, size, bold=False, italic=False, colour="black", ha="center", va="center", rot=0):
    ax.text(x, y, s, fontsize=size, fontweight="bold" if bold else "normal", fontstyle="italic" if italic else "normal", color=colour, ha=ha, va=va,
            rotation=rot, family=[SERIF, FALLBACK], rotation_mode="anchor")


def preview(T: Table, outdir: str) -> dict:
    L = layout(T)
    colw, rowh = L["colw"], L["rowh"]
    cap = f"{T.title_no}  {T.caption}" if T.title_no else T.caption
    cap_lines = wrap(cap, T.width_in * 72.0, 8.0)
    note_lines = wrap(T.note, T.width_in * 72.0, T.font_note)
    cap_h = len(cap_lines) * 8.0 * LINE_SPACING + 6
    rule_h = {"top": 0.8, "mid": 0.5, "bottom": 0.8, "cmid": 0.5}
    body_h = sum(rowh) + sum(rule_h[it[0]] for it in T.items if it[0] in ("top", "mid", "bottom")) + sum(1.5 for it in T.items if it[0] == "cmid") + 4
    note_h = len(note_lines) * T.font_note * LINE_SPACING + 8
    H = cap_h + body_h + note_h + 6
    W = T.width_in * 72.0
    fig = plt.figure(figsize=(W / 72.0, H / 72.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    y = 3.0
    for ln in cap_lines:
        draw_text(ax, 0, y + 4, ln, 8.0, ha="left", va="center")
        y += 8.0 * LINE_SPACING
    y += 3
    xs = [0.0]
    for w in colw:
        xs.append(xs[-1] + w)
    ri = 0
    row_y = {}
    lw = lambda pt: pt * 1.0
    for it in T.items:
        kind = it[0]
        if kind in ("top", "mid", "bottom"):
            ax.plot([0, W], [y + rule_h[kind] / 2, y + rule_h[kind] / 2], color="black", lw=rule_pt(kind), solid_capstyle="butt")
            y += rule_h[kind]
        elif kind == "cmid":
            for a, b in it[1]:
                ax.plot([xs[a - 1] + 3, xs[b] - 3], [y + 0.5, y + 0.5], color="black", lw=0.5, solid_capstyle="butt")
            y += 1.5
        elif kind == "row":
            h = rowh[ri]
            row_y[ri] = y
            j = 0
            for c in it[1]:
                if c.covered:
                    j += c.span
                    continue
                x0, x1 = xs[j], xs[j + c.span]
                hh = sum(rowh[ri:ri + c.rowspan])
                cy = y + hh / 2
                ls = cell_lines(c)
                if c.rot:
                    nlines = len(ls)
                    lh = (T.font_main if not c.small else T.font_sub) * LINE_SPACING
                    top = (x0 + x1) / 2 - nlines * lh / 2
                    for k, (t, kk, b, i, col) in enumerate(ls):
                        sz = T.font_main if kk == "m" else T.font_sub
                        if c.small:
                            sz = T.font_sub
                        draw_text(ax, top + (k + 0.5) * lh, cy, t, sz, b, i, col, ha="center", va="center", rot=90)
                else:
                    block_h = sum((T.font_main if (k == "m" and not c.small) else T.font_sub) * LINE_SPACING for _, k, *_ in ls)
                    ty = cy - block_h / 2
                    for (t, kk, b, i, col) in ls:
                        sz = T.font_main if (kk == "m" and not c.small) else T.font_sub
                        lh = sz * LINE_SPACING
                        if c.align == "l":
                            tx, ha = x0 + TABCOLSEP_PT, "left"
                        elif c.align == "r":
                            tx, ha = x1 - TABCOLSEP_PT, "right"
                        else:
                            tx, ha = (x0 + x1) / 2, "center"
                        if t:
                            if c.sup and kk == "m" and not c.lines:
                                base = t[: -len(c.sup)]
                                wb = text_w(base, sz, b, i)
                                ws = text_w(c.sup, sz * 0.7, b, i)
                                tot = wb + ws
                                sx = tx - tot / 2 if ha == "center" else (tx if ha == "left" else tx - tot)
                                draw_text(ax, sx, ty + lh / 2, base, sz, b, i, col, ha="left", va="center")
                                draw_text(ax, sx + wb, ty + lh / 2 - sz * 0.32, c.sup, sz * 0.7, b, i, col, ha="left", va="center")
                            else:
                                draw_text(ax, tx, ty + lh / 2, t, sz, b, i, col, ha=ha, va="center")
                        ty += lh
                j += c.span
            y += h
            ri += 1
    y += 3
    for ln in note_lines:
        draw_text(ax, 0, y + T.font_note * 0.6, ln, T.font_note, ha="left", va="center", colour="#222222")
        y += T.font_note * LINE_SPACING
    fig.savefig(f"{outdir}/{T.stem}.pdf", metadata={"CreationDate": None, "Creator": "paper_figures/final", "Title": T.stem})
    fig.savefig(f"{outdir}/{T.stem}.png", dpi=300, metadata={"Software": "paper_figures/final"})
    plt.close(fig)
    return {"width_in": W / 72.0, "height_in": H / 72.0, "natural_width_in": L["natural"] / 72.0, "extra_pt": L["extra"], "ok_fit": L["extra"] >= -0.5,
            "min_pt": min([T.font_main, T.font_sub])}


# ----------------------------------------------------------------------------------------------------------------- LaTeX
_TEX = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"),
        ("−", "$-$"), ("±", r"$\pm$"), ("→", r"$\rightarrow$"), ("†", r"$^{\dagger}$"), ("ᵃ", r"$^{a}$"), ("—", "---"), ("–", "--"), ("×", r"$\times$"), ("≥", r"$\geq$"), ("≤", r"$\leq$")]


def tex(s: str) -> str:
    out = []
    for ch in s:
        for k, v in _TEX:
            if ch == k:
                out.append(v)
                break
        else:
            out.append(ch)
    r = "".join(out)
    assert all(ord(c) < 128 for c in r), f"non-ASCII left in {s!r}"
    return r


def fsz(size: float) -> str:
    return rf"\fontsize{{{size:g}}}{{{size * LINE_SPACING:.1f}}}\selectfont"


def tex_sup(sup: str) -> str:
    if not sup:
        return ""
    return r"$^{\dagger}$" if sup == "†" else rf"$^{{{sup}}}$"


def tex_cell(c: Cell, T: Table) -> str:
    if c.covered:
        return ""
    fm_, fs = T.font_main, T.font_sub
    if c.lines:
        body = r"\\".join(tex(x) for x in c.lines)
        inner = rf"\shortstack[c]{{{body}}}"
        if c.small:
            inner = "{" + fsz(fs) + " " + inner + "}"
        if c.rot:
            inner = rf"\rotatebox[origin=c]{{90}}{{{inner}}}"
        return inner
    main = tex(c.main) + tex_sup(c.sup)
    if c.bold:
        main = rf"\textbf{{{main}}}"
    if c.italic:
        main = rf"\textit{{{main}}}"
    if c.small:
        main = "{" + fsz(fs) + " " + main + "}"
    if c.sub or c.two:
        sub = tex(c.sub) if c.sub else r"\phantom{$\pm$0}"
        if c.italic and c.sub:
            sub = rf"\textit{{{sub}}}"
        body = rf"{main}\\{{{fsz(fs)} {sub}}}"
        inner = rf"\shortstack[{'l' if c.align == 'l' else 'c'}]{{{body}}}"
    else:
        inner = main
    if c.rot:
        inner = rf"\rotatebox[origin=c]{{90}}{{{inner}}}"
    if c.rowspan > 1:
        inner = rf"\multirow{{{c.rowspan}}}{{*}}{{{inner}}}"
    if c.span > 1:
        a = {"l": "l", "c": "c", "r": "r"}[c.align]
        inner = rf"\multicolumn{{{c.span}}}{{{a}}}{{{inner}}}"
    return inner


def emit_tex(T: Table, wide: bool = True) -> str:
    env = "table*" if wide else "table"
    wid = r"\textwidth" if wide else r"\columnwidth"
    spec = "".join(T.aligns)
    out = [f"% {T.stem}.tex: generated by paper_figures/final/final_tables.py from the values csv of the same stem. Needs booktabs, graphicx and multirow.",
           "% NOT COMPILED: there is no TeX engine on the compute server or on the laptop; the preview pdf/png is a matplotlib drawing of the same content.",
           rf"\begin{{{env}}}[t]", r"\centering", rf"\caption{{{tex(T.caption)}}}", rf"\label{{{T.label}}}",
           "{" + fsz(T.font_main), rf"\setlength{{\tabcolsep}}{{{TABCOLSEP_PT:g}pt}}", rf"\begin{{tabular*}}{{{wid}}}{{@{{\extracolsep{{\fill}}}}{spec}@{{}}}}"]
    for it in T.items:
        k = it[0]
        if k == "top":
            out.append(r"\toprule")
        elif k == "mid":
            out.append(r"\midrule")
        elif k == "bottom":
            out.append(r"\bottomrule")
        elif k == "cmid":
            out.append("".join(rf"\cmidrule(lr){{{a}-{b}}}" for a, b in it[1]))
        elif k == "row":
            cells = [tex_cell(c, T) for c in it[1]]
            out.append(" & ".join(cells) + r" \\")
    out += [r"\end{tabular*}", r"\par\medskip", rf"\begin{{minipage}}{{{wid}}}", r"{" + fsz(T.font_note), r"\raggedright", r"\textit{Note.} " + tex(T.note), r"\par}", r"\end{minipage}", "}", rf"\end{{{env}}}"]
    return "\n".join(out) + "\n"


def lint_tex(s: str, T: Table) -> list[str]:
    """Structural checks of the generated TeX (it is not compiled): braces, environments, cells per row."""
    bad = []
    depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            i += 2                      # a control symbol or the first letter of a control word: never a brace
            continue
        if ch == "%":                   # comment to the end of the line
            while i < len(s) and s[i] != "\n":
                i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                bad.append(f"closing brace without opening at {i}")
                break
        i += 1
    if depth > 0:
        bad.append(f"unbalanced braces: {depth}")
    for env in ("table*", "table", "tabular*", "minipage"):
        if s.count(rf"\begin{{{env}}}") != s.count(rf"\end{{{env}}}"):
            bad.append(f"environment {env} not closed")
    for ln in s.splitlines():
        if ln.endswith(r" \\") and not ln.startswith("%"):
            parts = split_cells(ln[:-3])
            width = 0
            for p in parts:
                m = re.match(r"\\multicolumn\{(\d+)\}", p.strip())
                width += int(m.group(1)) if m else 1
            if width != T.ncols:
                bad.append(f"row has {width} columns, expected {T.ncols}: {ln[:80]}")
    return bad


def split_cells(line: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            cur += line[i:i + 2]
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    parts.append(cur)
    return parts
