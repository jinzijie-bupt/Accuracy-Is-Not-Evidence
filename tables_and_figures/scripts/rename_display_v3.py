#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rename_display_v3.py -- method-name pass over the generated tables (2026-09-24).

The generators and the registry keep their internal objective keys ("grouped", "grouped-nohinge", ...). Only the
printed labels change: the objective is called the placebo-controlled objective everywhere in the paper, and its
ablations are named by what they remove. Runs after the generators and after diff_tables_v3.py (which compares
against the original copy), replaces a fixed list of display strings, and fails if any printed "grouped" remains.
Comment lines (sources, hashes) are left untouched.
"""
import glob, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# longest patterns first
REPL = [
    ("grouped, no hinge, no group wts", "no bound, uniform"),     # the name Table 2 and Figure 3 use (round 2)
    ("grouped, no group wts", "uniform weights"),
    ("grouped, no hinge", "no placebo bound"),
    ("two-stage (SFT then grouped)", "two-stage (SFT, then placebo-controlled)"),
    ("every grouped Qwen2.5-VL-7B and Time-R1 model", "every placebo-controlled Qwen2.5-VL-7B and Time-R1 model"),
    ("every grouped model", "every placebo-controlled model"),
    ("the grouped models", "the placebo-controlled models"),
    ("grouped $", "placebo-controlled $"),
]
KEEP = re.compile(r"Rows are grouped by")
NEGZERO = re.compile(r"(?<![\d.])-(0\.0+)(?!\d)")   # a value that rounds to zero prints without a sign
# a " [lo, hi]" interval after an estimate prints lighter (\ci, main.tex); a text-mode hyphen before a number prints
# as a minus sign (never before a length such as -2pt)
CIV = re.compile(r"\s\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
MINUS = re.compile(r"(?<=[\s&{\[(])-(?=\d+(?:\.\d+)?(?![\d.]*(?:pt|em|ex|cm|mm)))")


def main():
    left = []
    for f in sorted(glob.glob(os.path.join(ROOT, "tables", "*.tex"))):
        if "T3_survival" in f:      # legacy table of the original copy, not \input anywhere (same exclusion as check_tex)
            continue
        src = open(f, encoding="utf-8").read()
        out = []
        for line in src.split("\n"):
            if not line.lstrip().startswith("%"):
                for a, b in REPL:
                    line = line.replace(a, b)
                line = NEGZERO.sub(r"\1", line)
                if "&" in line and line.rstrip().endswith("\\\\"):   # data rows only
                    line = CIV.sub(r" \\ci{\1}{\2}", line)
                    line = MINUS.sub("$-$", line)
                if os.path.basename(f).startswith("S"):   # appendix tables sit next to the text that reads them
                    line = re.sub(r"\\begin\{table\}\[tb?\]", r"\\begin{table}[!htbp]", line)
                probe = KEEP.sub("", line)
                if re.search(r"\bgrouped\b", probe):
                    left.append((os.path.basename(f), line.strip()[:120]))
            out.append(line)
        new = "\n".join(out)
        if new != src:
            tmp = f + ".tmp"
            open(tmp, "w", encoding="utf-8", newline="\n").write(new)
            os.replace(tmp, f)
    if left:
        for x in left:
            print("   leftover:", *x)
        raise SystemExit(f"rename_display_v3: {len(left)} printed 'grouped' remain")
    print("   rename_display_v3: no printed 'grouped' left")


if __name__ == "__main__":
    main()
