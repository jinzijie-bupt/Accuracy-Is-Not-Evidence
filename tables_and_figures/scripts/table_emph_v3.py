#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""table_emph_v3.py -- bold the table entries the text uses to draw its conclusions (2026-09-24).

Plan and reasons: review/research_20260924/table_bold_plan.md.  Bold marks the anchors between a sentence of the
text and a table cell, not the best value per column.  Every generator calls emph(text, name) just before it writes
its .tex file; name is the output file name without ".tex".

Cells are located by (section context, row label, column name), never by value, so a data update cannot move the
bold onto another cell:
  * context   = text of the last \\multicolumn{..}{..}{\\textit{..}} / {\\emph{..}} row above ("" before the first one;
                None in a spec means any context);
  * row label = first cell without \\addlinespace / \\quad; a row whose first cell is empty or "trained" (S6, S13)
                is named "<parent row label> :: <second cell>"; every row can also be addressed as
                "<label> :: <second cell>" (S9, two rows share a label);
  * column    = header text of the last header row above the first \\midrule (or of a repeated header row inside
                the body, S5), with an occurrence number for duplicated header names (S7, S9).
Labels and contexts are normalised with rename_display_v3.REPL, so specs are written in the printed names whether
the generator still prints the internal "grouped" keys or not.  Cells that are \\pending, "--" or \\NA are never
bolded.  A spec that matches no cell or more than one cell prints "BOLD-MISS" and is skipped; it never stops a rebuild.
Tables with at least one bold cell get the same closing sentence in their caption (NOTE).
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    from rename_display_v3 import REPL  # noqa: E402
except Exception:          # the pass must never break a generator
    REPL = []

NOTE = "Bold: the entries discussed in the text."
PEND = "\\pending"

# table -> [(context or None, row label, column name, occurrence of that column name)]
# round 2: appendix tables carry no bold (review/round2_20260924/figures_B_tables.md, section 4.6), and the
# three main-text tables use RULES below, so no table is bolded cell by cell any more.  The machinery is kept for a
# table that needs it again.
SPECS = {}

# Main-text tables use a rule instead of a cell list, so the bold means the same thing everywhere and can be read
# without the text: a change is bold when its 95% interval excludes zero; Gamma (and its token parts) is bold beyond
# the audit's resolution and grey within it.  Rows of the placebo-controlled objective get a light background.
RULE_NOTE = (r"Bold marks a change in points whose 95\% interval excludes zero or a $\Gamma$ beyond the audit's resolution; "
             r"grey marks a $\Gamma$ within it.")
RULE_NOTES = {"T4_split": r"Bold marks a change whose 95\% interval excludes zero.",
              "T1_audit": r"Bold marks a change in points whose interval excludes zero or a $\Gamma$ beyond the audit's resolution; "
                          r"grey marks a $\Gamma$ within it.",
              "T2_objectives": r"bold follow Table~\ref{tab:audit}."}   # contained in the T2 caption itself, so nothing is appended there
# column header (as printed in the last header row) -> rule; the token parts of Gamma (T4) carry no rule: the
# resolution is defined for the whole Gamma only
RULES = {
    "T1_audit": {r"\Rfive{}": ("ci",), r"$\Gm$": ("res", 0.23), r"$\Gs$": ("res", 0.31),
                 r"\shortstack{evidence\\share}": ("ci",), r"\shortstack{follows\\moved event}": ("ci",)},
    "T2_objectives": {r"$\Delta$\Rfive{}": ("ci",), r"$\Gm$": ("res", 0.23), r"$\Gs$": ("res", 0.31)},
    "T4_split": {"no interval": ("ci",), "wrong interval": ("ci",)},
}
# the Gamma columns of T1/T2 are printed as two-line headers; apply the Gamma rule to them as well
for _t in ("T1_audit", "T2_objectives"):
    RULES[_t][r"\shortstack{$\Gamma$\\mask}"] = RULES[_t][r"$\Gm$"]
    RULES[_t][r"\shortstack{$\Gamma$\\swap}"] = RULES[_t][r"$\Gs$"]
SHADE = {"T1_audit": lambda label, ctx: ctx.startswith("Placebo-controlled"),
         "T4_split": lambda label, ctx: ctx.startswith("Placebo-controlled")}
_NUM = re.compile(r"^\$?([+-]?\d+(?:\.\d+)?)")
_CIV = re.compile(r"\[\s*\$?([+-]?\d+(?:\.\d+)?)\$?\s*,\s*\$?([+-]?\d+(?:\.\d+)?)\$?\s*\]\s*$")


def _grey(cell):
    m = re.match(r"^(\s*)(.*?)(\s*)$", cell, re.S)
    lead, body, trail = m.groups()
    return lead + "\\textcolor{black!45}{" + body + "}" + trail


def _rule_cell(cell, rule):
    body = cell.strip()
    if _blank(cell) or "\\textbf{" in body:
        return cell
    m = _NUM.match(body)
    if not m:
        return cell
    est = float(m.group(1))
    if rule[0] == "ci":
        c = _CIV.search(body)
        if c and (float(c.group(1)) > 0 or float(c.group(2)) < 0):
            return _bold(cell)
        return cell
    return _bold(cell) if abs(est) > rule[1] else _grey(cell)


def _emph_rule(text, name):
    rules, shade = RULES[name], SHADE.get(name)
    lines = text.split("\n")
    header, in_head, in_body, ctx, heads, n = None, False, False, "", [], 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("%"):
            continue
        if "\\toprule" in s:
            in_head, heads = True, []
            continue
        if in_head and "\\midrule" in s:
            in_head, in_body = False, True
            header = [_norm(c) for c in heads[-1]] if heads else []
            continue
        m = _ROWEND.match(ln)
        if not m or not _AMP.search(m.group(1).replace("\\&", "")):
            mc = _CTX.search(s)
            if in_body and mc and not _AMP.search(s.replace("\\&", "")):
                ctx = _norm(mc.group(1))
            continue
        pieces = _AMP.split(m.group(1))
        if in_head:
            heads.append(pieces)
            continue
        if not in_body:
            continue
        for k, h in enumerate(header):
            r = rules.get(h) or rules.get(_norm(h))
            if r and k < len(pieces):
                new = _rule_cell(pieces[k], r)
                n += new != pieces[k]
                pieces[k] = new
        row = "&".join(pieces) + m.group(2)
        if shade and shade(_norm(pieces[0]), ctx):
            row = "\\rowcolor{pcrow}" + row
        lines[i] = row
    note = RULE_NOTES.get(name, RULE_NOTE)
    text = "\n".join(lines).replace(" " + NOTE, "")
    if note not in text:
        text = _add_note(text.replace(NOTE, "")).replace(NOTE, note)
    print(f"   emphasis {name}: rule applied to {n} cells")
    return text


_CTX = re.compile(r"\\multicolumn\{\d+\}\{(?:[^{}]|\{\})*\}\{\\(?:textit|emph)\{(.+?)\}\}\s*\\\\\s*$")
_ROWEND = re.compile(r"^(.*?)(\s*\\\\\s*)$")
_AMP = re.compile(r"(?<!\\)&")


def _norm(s):
    s = re.sub(r"^\\addlinespace\[[^\]]*\]", "", s.strip()).strip()
    s = re.sub(r"^\\quad\b", "", s).strip()
    for a, b in REPL:
        s = s.replace(a, b)
    return s


def _blank(cell):
    c = cell.strip()
    return (not c) or PEND in c or c in ("--", "\\NA", "---", "n/a")


def _bold(cell):
    """wrap the point estimate, keep the surrounding spaces and leave a trailing " [lo, hi]" interval in regular
    weight (narrower, and the estimate is what the text quotes); math content also needs \\boldmath."""
    m = re.match(r"^(\s*)(.*?)(\s*)$", cell, re.S)
    lead, body, trail = m.groups()
    if "\\textbf{" in body:
        return cell
    est, ci = body, ""
    k = body.find(" [")
    if k > 0 and body.rstrip().endswith("]"):
        est, ci = body[:k], body[k:]
    bm = "\\boldmath " if ("$" in est or est.lstrip().startswith("-")) else ""   # a text minus becomes $-$ later
    return lead + "\\textbf{" + bm + est + "}" + ci + trail


def _add_note(text):
    for opener in ("\\caption{", "\\captionof{table}{"):
        i = text.find(opener)
        if i >= 0:
            break
    else:
        return text
    if NOTE in text:
        return text
    j, depth = i + len(opener), 1
    while j < len(text) and depth:
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        depth += (ch == "{") - (ch == "}")
        j += 1
    close = j - 1                     # index of the caption's closing brace
    if depth or text[close] != "}":
        return text
    head = text[:close].rstrip()
    return head + " " + NOTE + text[close:]


def emph(text, name):
    """apply SPECS[name] to one generated table (full file text); returns the new text."""
    try:
        if name in RULES:
            return _emph_rule(text, name)
        specs = SPECS.get(name, [])
        if not specs:
            return text
        lines = text.split("\n")
        header, in_head, in_body, ctx, parent = None, False, False, "", ""
        rows = []                     # (line index, pieces, keys, ctx, header)
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s.startswith("%"):
                continue
            if "\\toprule" in s:
                in_head, heads = True, []
                continue
            if in_head and "\\midrule" in s:
                in_head, in_body = False, True
                header = [_norm(c) for c in heads[-1]] if heads else None
                continue
            m = _ROWEND.match(ln)
            if not m or not _AMP.search(m.group(1).replace("\\&", "")):
                mc = _CTX.search(s)
                if in_body and mc and not _AMP.search(s.replace("\\&", "")):
                    ctx = _norm(mc.group(1))
                continue
            pieces = _AMP.split(m.group(1))
            if in_head:
                heads.append(pieces)
                continue
            if not in_body:
                continue
            first = _norm(pieces[0])
            if header and first and first == header[0]:          # repeated header inside the body (S5)
                header = [_norm(c) for c in pieces]
                continue
            second = _norm(pieces[1]) if len(pieces) > 1 else ""
            if first in ("", "trained"):
                label = f"{parent} :: {second}"
                keys = {label}
            else:
                parent, label = first, first
                keys = {label, f"{label} :: {second}"}
            rows.append((i, pieces, m.group(2), keys, ctx, header))
        n_bold = 0
        for sctx, slab, scol, occ in specs:
            sctx = None if sctx is None else _norm(sctx)
            slab_n = " :: ".join(_norm(x) for x in slab.split(" :: "))
            hits = [r for r in rows if slab_n in r[3] and (sctx is None or r[4] == sctx)]
            if len(hits) != 1:
                print(f"   BOLD-MISS {name}: {len(hits)} rows for ({sctx}, {slab})")
                continue
            i, pieces, end, _, _, hdr = hits[0]
            idx = [k for k, h in enumerate(hdr or []) if h == _norm(scol)]
            if len(idx) < occ:
                print(f"   BOLD-MISS {name}: column {scol!r} #{occ} not in header")
                continue
            k = idx[occ - 1]
            if k >= len(pieces) or _blank(pieces[k]):
                print(f"   bold skipped {name}: ({slab}, {scol}) is empty or pending")
                continue
            pieces[k] = _bold(pieces[k])
            lines[i] = "&".join(pieces) + end
            n_bold += 1
        text = "\n".join(lines)
        if n_bold:
            text = _add_note(text)
        print(f"   bold {name}: {n_bold}/{len(specs)} cells")
        return text
    except Exception as e:          # never break a rebuild over emphasis
        print(f"   BOLD-MISS {name}: emphasis pass failed ({e!r}); table written without bold")
        return text


if __name__ == "__main__":           # dry run on the tables as they are: python table_emph_v3.py
    T = os.path.join(HERE, "..", "tables")
    for n in SPECS:
        p = os.path.join(T, n + ".tex")
        if os.path.exists(p):
            emph(open(p, encoding="utf-8").read(), n)
