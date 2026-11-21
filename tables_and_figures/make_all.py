#!/usr/bin/env python3
"""Regenerate every data table and data figure of the paper from the result files in ../results.

    python make_all.py            # tables + figures
    python make_all.py --tables   # tables only

Steps (each is a separate script in scripts/, run in this order):
  1. build_numbers_v2.py      reader products -> tables/paper_numbers_v2.json (one record per run/pair)
  2. canonical_v3.py          one evaluation stack per model configuration -> tables/CANONICAL_MANIFEST_v3.json
  3. table builders           Tables 1-4 and S1-S9, S11-S17 (LaTeX, tables/*.tex)
  4. rename_display_v3.py     display names used in the paper
  5. plot_figs_v6.py          data figures (figures/**); the qualitative-case figure (fig8_cases) additionally
                              needs two Charades-STA videos (BL7OF.mp4, AK2KG.mp4) in assets/source_videos/ (not included) and is
                              skipped when they are absent.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, "scripts")
PY = sys.executable
TABLES = ["build_numbers_v2.py", "canonical_v3.py", "build_tables_v3.py", "build_appendix_tables_v3.py",
          "build_query_spec_table_v3.py", "build_newly_correct_table_v3.py", "build_order_table_v3.py",
          "build_cd_table_v3.py", "build_scale_newly_tables_v3.py", "build_split_tables_v3.py", "rename_display_v3.py"]
FIGS = ["fig1_map", "fig2_findings", "fig5_mechanism", "fig6_dose", "fig5_readouts", "fig9_ood", "fig10_dynamics"]


def run(args):
    print(">>", " ".join(os.path.basename(a) for a in args[1:]), flush=True)
    subprocess.run(args, check=True, cwd=S)


def main():
    for d in ("tables", "figures/data", "figures/data_redesign_v3", "figures/data_redesign_v4"):
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    for s in TABLES:
        run([PY, os.path.join(S, s)])
    if "--tables" in sys.argv:
        return
    figs = list(FIGS)
    man = json.load(open(os.path.join(HERE, "assets", "source_videos", "REAL_CASES_MANIFEST_20260922.json"), encoding="utf-8"))
    vids = {c["video"] for c in man["cases"]}
    if all(os.path.exists(os.path.join(HERE, "assets", "source_videos", v)) for v in vids):
        figs.append("fig8_cases")
    else:
        print("   fig8_cases skipped: place", sorted(vids), "in assets/source_videos/")
    run([PY, os.path.join(S, "plot_figs_v6.py")] + figs)
    print("done")


if __name__ == "__main__":
    main()
