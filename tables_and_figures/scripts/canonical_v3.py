#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""canonical_v3.py -- canonical environments, revision 3 (author ruling 2026-09-23).

Differences from canonical_v2 (which is kept unchanged):
  R2' stack choice by pairing completeness: for every configuration count, in the tf5 stack (A u B) and in the
      tf4.51 stack (C), the table cells that close against the configuration's own base INSIDE ONE specific
      environment (the v2 'base pairing' count; union over A and B for tf5).  More wins; a tie goes to tf5.
      A configuration with products only in one stack takes that stack.
  R2'' inside tf5, the environment is chosen per (configuration, kind, dataset): the one of A / B that holds the
      product; if both hold it, the environment of the configuration's 14Ln gamma (Charades-STA); without a gamma
      in exactly one of A/B, the environment with more table cells; then A.
  R4' a configuration whose stack is C takes every kind from C and never borrows from tf5 (and vice versa).
  Hard constraint: one value per (configuration, kind, dataset) across all tables -- guaranteed because every
  pick goes through env_for(); paired quantities close inside one specific environment (A-A, B-B, C-C).
  Base rows: every (base, kind, dataset) likewise has one global value.  A table that prints a base value from
  one environment next to a Delta paired in another gets the caption sentence CAPTION_NOTE.
Rulings kept from v2: message order is part of the key; PRIMARY_RUN; TimeZero a=0.5 train=v eval=t stays in A
(it is a tf5 configuration; its kinds are resolved by R2'').
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import canonical_v2 as v2  # noqa: E402
from canonical_v2 import M, key_str, base_of, rec_name, Q7, Q3, TZ, TR, VC, TL, PEND, TF5  # noqa: E402,F401

ROOT = v2.ROOT
MANIFEST3 = os.path.join(ROOT, "tables", "CANONICAL_MANIFEST_v3.json")
OVERRIDES3 = os.path.join(ROOT, "tables", "CANONICAL_OVERRIDES_v3.json")   # optional {"<key>": "tf5"|"C"}
CAPTION_NOTE = "$\\Delta$ is paired against the base run on the same software environment."
# trained on node4 (tf4.51, env C) and otherwise read through a key-converted adapter under tf5; for them every
# (kind, dataset) that C holds is taken from C, and only the kinds C lacks stay in tf5.  The converted adapter lost
# about 3 points of accuracy against its own base (abstentions 160 -> 317) while Gamma moved by 0.1 (review
# round 3, results/judgment_20260925_native/).
TRAIN_STACK_C = {"Qwen2.5-VL-7B|grouped|a=0.5|b=0.1|s=0|train=v|eval=v", "Qwen2.5-VL-7B|sft|b=0|s=0|train=v|eval=v"}


class Canon(v2.Canon):
    def __init__(self, reg_path=v2.REG_V2):
        super().__init__(reg_path=reg_path, overrides_path=None)

    # ------------------------------------------------------------------ decisions
    def _closable(self, k, env):
        """set of (kind, dataset) cells of k in env that close against k's base inside env."""
        cells = self.cov.get(k, {}).get(env, {})
        if k[1] == "base":
            return set()
        bcells = self.cov.get(base_of(k), {}).get(env, {})
        out = set()
        for (kind, ds) in cells:
            if kind in ("gamma", "pair_acc", "behav", "qswap", "newly"):
                out.add((kind, ds))
            elif kind in ("acc", "black", "tneg", "labelswap", "cd") and (kind, ds) in bcells:
                out.add((kind, ds))
        return out

    def _decide(self):
        self.cov = self._cells()
        ov = json.load(open(OVERRIDES3, encoding="utf-8")) if os.path.exists(OVERRIDES3) else {}
        self.decisions = {}
        for k, envs in sorted(self.cov.items(), key=lambda kv: key_str(kv[0])):
            cand = sorted(envs)
            tf5_c = set().union(*[self._closable(k, e) for e in cand if e in TF5]) if any(e in TF5 for e in cand) else set()
            c_c = self._closable(k, "C") if "C" in cand else set()
            has_tf5, has_c = any(e in TF5 for e in cand), "C" in cand
            if has_tf5 and not has_c:
                stack, why = "tf5", "R2': only the tf5 stack has outputs"
            elif has_c and not has_tf5:
                stack, why = "C", "R2': only the C stack has outputs"
            elif len(c_c) > len(tf5_c):
                stack, why = "C", f"R2': closable cells C={len(c_c)} > tf5={len(tf5_c)}"
            else:
                stack, why = "tf5", f"R2': closable cells tf5={len(tf5_c)} >= C={len(c_c)}(tie goes to tf5)"
            ks = key_str(k)
            if ks in ov:
                stack, why = ov[ks], f"OVERRIDE; rule decision: {why}"
            kind_env = {}
            if stack == "C":
                for cell in envs.get("C", {}):
                    kind_env[f"{cell[0]}@{cell[1]}"] = "C"
            else:
                gam = [e for e in ("A", "B") if ("gamma", "charades-sta") in envs.get(e, {})]
                for e in ("A", "B"):
                    for cell in envs.get(e, {}):
                        c = f"{cell[0]}@{cell[1]}"
                        if c in kind_env:
                            continue
                        hold = [x for x in ("A", "B") if cell in envs.get(x, {})]
                        if len(hold) == 1:
                            kind_env[c] = hold[0]
                        elif len(gam) == 1:
                            kind_env[c] = gam[0]
                        else:
                            na, nb = len(envs.get("A", {})), len(envs.get("B", {}))
                            kind_env[c] = "B" if nb > na else "A"
                if ks in TRAIN_STACK_C:
                    for cell in envs.get("C", {}):
                        kind_env[f"{cell[0]}@{cell[1]}"] = "C"
                    why += "; R5: kinds available on training stack C take C"
            self.decisions[k] = {"stack": stack, "env": ("C" if stack == "C" else "tf5"), "candidates": cand,
                                 "closable": {"tf5": sorted(f"{a}@{b}" for a, b in tf5_c), "C": sorted(f"{a}@{b}" for a, b in c_c)},
                                 "kind_env": dict(sorted(kind_env.items())), "reason": why,
                                 "coverage": {e: len(envs[e]) for e in cand},
                                 "cells": {e: {f"{c[0]}@{c[1]}": v for c, v in sorted(envs[e].items())} for e in cand}}

    def env(self, k):
        d = self.decisions.get(k)
        return None if d is None else ("C" if d["stack"] == "C" else "tf5")

    def default_env(self, k):
        d = self.decisions.get(k)
        if d is None:
            return None
        if d["stack"] == "C":
            return "C"
        g = d["kind_env"].get("gamma@charades-sta")
        if g:
            return g
        cov = d["coverage"]
        a, b = cov.get("A", 0), cov.get("B", 0)
        return "B" if b > a else "A"

    def env_for(self, k, kind, dataset="charades-sta"):
        d = self.decisions.get(k)
        if d is None:
            return None
        return d["kind_env"].get(f"{kind}@{dataset}") or self.default_env(k)

    # ------------------------------------------------------------------ selection (v2 semantics, per-kind env)
    def pick(self, kind, k, dataset="charades-sta", column=None, env=None, pending=True, **extra):
        rule = "canonical" if env is None else "paired-stack"
        env = env or self.env_for(k, kind, dataset)
        if env is None:
            if pending:
                self.log_pending(column, kind, k, dataset, None, "this config has no table output in any environment", extra)
            return None
        out = [r for r in self.R if self._match(r, kind, k, env, dataset, extra)]
        out = self._resolve(out, kind, k, env, dataset)
        if out:
            self.note(out[0], column, rule)
            return out[0]
        if pending:
            self.log_pending(column, kind, k, dataset, env, None, extra)
        return None

    def log_pending(self, column, kind, k, dataset, env, why=None, extra=None):
        extra = extra or {}
        stack = self.env(k) if k else None
        allowed = ("C",) if stack == "C" else ("A", "B")
        other = sorted({r["_env"] for r in self.R if self._match(r, kind, k, None, dataset, extra) and r["_env"] != env})
        in_stack = [e for e in other if e in allowed]
        cls = "other_stack" if (other and not in_stack) else ("other_env_same_stack" if in_stack else "absent")
        self.pending.append({"table": self._ctx[0], "row": self._ctx[1], "column": column, "kind": kind,
                             "key": key_str(k) if k else None, "dataset": dataset, "canonical_env": env,
                             "stack": stack, "class": cls, "other_envs": other, "why": why})

    def delta3(self, k, dataset="charades-sta"):
        """(text-ready tuple) -> (value, ci, dagger, paired_env); pair reader inside env_for(pair_acc), else the
        difference of marginals inside env_for(acc) with the base run of THAT environment; else None."""
        env = self.env_for(k, "pair_acc", dataset)
        out = [r for r in self.R if r.get("kind") == "pair_acc" and r["_key"] == k and r["_env"] == env
               and r["_ref_env"] == env and r.get("dataset") == dataset and r.get("_ref_key") == base_of(k)]
        if len(out) > 1:
            raise SystemExit(f"canonical_v3: pair selector {key_str(k)} matched {len(out)}")
        if out:
            self.note(out[0], "Delta", "canonical")
            return out[0]["delta_R50"], out[0]["delta_R50_ci"], False, env
        ea = self.env_for(k, "acc", dataset)
        acc = self.pick("acc", k, dataset, column="Delta:arm", pending=False)
        bacc = self.pick("acc", base_of(k), dataset, column="Delta:base (same env)", env=ea, pending=False)
        if acc and bacc:
            return acc["R50"] - bacc["R50"], None, True, ea
        if acc is None:
            self.log_pending("Delta", "acc", k, dataset, ea, "trained run has no acc")
        else:
            self.log_pending("Delta", "acc", base_of(k), dataset, ea,
                             f"trained run acc is on {ea}, no base model acc and no paired readout in that specific environment; base model not borrowed across environments")
        return None, None, False, None

    def rows_for(self, kind, dataset=None, **extra):
        keys = []
        for r in self.R:
            if r.get("kind") == kind and r.get("_key") and (dataset is None or r.get("dataset") == dataset) \
                    and all((extra[kk] in rec_name(r)) if kk == "name_has" else r.get(kk) == vv for kk, vv in extra.items()):
                t = (r["_key"], r.get("dataset"))
                if t not in keys:
                    keys.append(t)
        out = []
        for k, ds in keys:
            env = self.env_for(k, kind, ds)
            recs = [r for r in self.R if self._match(r, kind, k, env, ds, extra)]
            recs = self._resolve(recs, kind, k, env, ds)
            envs = sorted({r["_env"] for r in self.R if self._match(r, kind, k, None, ds, extra)})
            out.append((k, ds, recs[0] if recs else None, envs))
        return out

    def ext_pick(self, kind, k, dataset=None, column=None, **extra):
        env = self.env_for(k, kind, dataset) if dataset else None
        if env is None:
            hits_any = [x for x in self.ext if x[0] == kind and x[2] == k]
            env = self.env_for(k, kind, hits_any[0][1]) if hits_any else self.default_env(k)
        hits = [x for x in self.ext if x[0] == kind and x[2] == k and x[3] == env and (dataset is None or x[1] == dataset)
                and all(x[5].get(kk) == vv for kk, vv in extra.items())]
        if len(hits) > 1:
            raise SystemExit(f"canonical_v3: external {kind} {key_str(k)} env={env} matched {len(hits)}")
        if hits:
            x = hits[0]
            self.sources.append({"table": self._ctx[0], "row": self._ctx[1], "column": column, "kind": kind,
                                 "key": key_str(k), "env": env, "env_rule": "canonical", "name": os.path.basename(x[4]),
                                 "dataset": x[1], "src_file": os.path.basename(x[4]), "sha16": v2.sha16(x[4]), "extra": x[5]})
            return x
        other = sorted({x[3] for x in self.ext if x[0] == kind and x[2] == k and (dataset is None or x[1] == dataset)
                        and all(x[5].get(kk) == vv for kk, vv in extra.items())} - {env})
        stack = self.env(k)
        allowed = ("C",) if stack == "C" else ("A", "B")
        in_stack = [e for e in other if e in allowed]
        self.pending.append({"table": self._ctx[0], "row": self._ctx[1], "column": column, "kind": kind,
                             "key": key_str(k), "dataset": dataset, "canonical_env": env, "stack": stack,
                             "class": "other_stack" if (other and not in_stack) else ("other_env_same_stack" if in_stack else "absent"),
                             "other_envs": other, "why": None})
        return None

    def manifest(self):
        out = {"generated_by": "canonical_v3.py", "registry": os.path.basename(self.reg_path), "registry_sha16": self.reg_sha16,
               "rules": __doc__.strip(), "primary_run": {"|".join(map(str, k)): {"must_contain": v[0], "reason": v[1]}
                                                         for k, v in v2.PRIMARY_RUN.items()},
               "configurations": {key_str(k): d for k, d in self.decisions.items()}}
        return out


if __name__ == "__main__":
    C = Canon()
    man = C.manifest()
    json.dump(man, open(MANIFEST3, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    for ks, d in man["configurations"].items():
        print(f"{d['stack']:4s} {ks:62s} closable tf5={len(d['closable']['tf5'])} C={len(d['closable']['C'])}  {d['kind_env'] if d['stack'] == 'tf5' else ''}")
    print("->", MANIFEST3)
