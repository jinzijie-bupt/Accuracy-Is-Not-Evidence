# -*- coding: utf-8 -*-
"""_common.py -- shared helpers for the zero-GPU rebuttal analyses (2026-09-23).

Estimator conventions are copied from the registered reader readers/read_14Ln_v1.py:
  hit set H(f) = {qid : hit==1 and every op has *_tokmean_drop_evid and *_tokmean_drop_rand}
  psi = {op}_tokmean_drop_evid - {op}_tokmean_drop_rand          (nats/token)
  Gamma = mean_{H(base) & H(arm)} (psi_arm - psi_base)
  CI = video-cluster bootstrap, B=2000, percentile; for Gamma the rng stream is default_rng(0) consumed
       mask first then swap, exactly as the reader does (so numbers replicate the registry bit-for-bit).
Nothing in <local>/paper or <local>/results is modified; all outputs go to this directory.
"""
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..", "..")).replace("\\", "/")
import os, json, hashlib, datetime, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = _SUPP + "/results"
AUD = RES + "/native_audit_20260917"
OPS = ("mask", "swap")
ENV_BY_MACHINE = {"node2": "A", "node1": "B", "node3": "B", "node5": "B", "node4": "C"}
TF_BY_ENV = {"A": "5.13", "B": "5.12", "C": "4.51"}
B_DEFAULT = 2000

_cache, _sha = {}, {}


def sha16(p):
    p = os.path.abspath(p)
    if p not in _sha:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        _sha[p] = h.hexdigest()[:16]
    return _sha[p]


def load(p):
    p = os.path.abspath(p)
    if p not in _cache:
        _cache[p] = json.load(open(p, encoding="utf-8"))
    return _cache[p]


def audit_path(stem):
    """stem like '14Ln_base_timer1_default_node2' -> full path of '<stem>_nat_o0n200R3.json' (unique)"""
    fn = stem + "_nat_o0n200R3.json"
    hits = [f for f in glob.glob(RES + "/**/" + fn, recursive=True)]
    hits = sorted(set(os.path.abspath(h) for h in hits))
    if len(hits) > 1 and len({hashlib.sha256(open(h, "rb").read()).hexdigest() for h in hits}) == 1:
        hits = hits[:1]                            # byte-identical copies of one audit (09-25)
    if len(hits) != 1:
        raise SystemExit(f"audit_path: {fn} -> {len(hits)} matches {hits}")
    return hits[0].replace("\\", "/")


def machine_of(path):
    b = "_" + os.path.basename(path)
    for m in ("node2", "node1", "node3", "node4", "node5"):
        if f"_{m}_" in b:
            return m
    return None


def env_of(path):
    return ENV_BY_MACHINE.get(machine_of(path))


def hitset(rows, ops=OPS):
    out = {}
    for r in rows:
        if r.get("hit") == 1 and all(f"{op}_tokmean_drop_evid" in r and f"{op}_tokmean_drop_rand" in r for op in ops):
            out[int(r["qid"])] = r
    return out


def psi(r, op):
    return float(r[f"{op}_tokmean_drop_evid"]) - float(r[f"{op}_tokmean_drop_rand"])


def cidx(r, op, eps=0.05):
    e, o = float(r[f"{op}_tokmean_drop_evid"]), float(r[f"{op}_tokmean_drop_rand"])
    return (e - o) / (abs(e) + abs(o) + eps)


def nll(r):
    return -float(r["lp_clean_mean"])


def cluster_boot(vals, vids, rng, B=B_DEFAULT):
    """verbatim copy of read_14Ln_v1.cluster_boot (same rng consumption)"""
    vals = np.asarray(vals, float)
    uv = sorted(set(vids))
    gi = {v: [i for i, x in enumerate(vids) if x == v] for v in uv}
    bs = []
    for _ in range(B):
        pick = rng.integers(0, len(uv), len(uv))
        idx = np.concatenate([gi[uv[p]] for p in pick])
        bs.append(vals[idx].mean())
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], len(uv)


class Boot:
    """generic video-cluster bootstrap over a fixed item universe (items = qids with a video each).
    stat(idx) receives an index array into the universe (with repeats) and returns a float or np.nan."""

    def __init__(self, vids, B=B_DEFAULT, seed=0):
        self.vids = list(vids)
        uv = sorted(set(self.vids))
        gi = {v: [] for v in uv}
        for i, v in enumerate(self.vids):
            gi[v].append(i)
        self.groups = [np.array(gi[v]) for v in uv]
        rng = np.random.default_rng(seed)
        G = len(uv)
        self.draws = [np.concatenate([self.groups[k] for k in rng.integers(0, G, G)]) for _ in range(B)]
        self.nv = G

    def ci(self, stat):
        bs = np.array([stat(ix) for ix in self.draws], float)
        bs = bs[~np.isnan(bs)]
        if bs.size == 0:
            return [None, None]
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def masked_mean(vals, mask):
    def f(ix):
        m = mask[ix]
        return float(vals[ix][m].mean()) if m.any() else np.nan
    return f


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def gamma_pair(base_path, arm_path, B=B_DEFAULT, extra=True):
    """Gamma exactly as read_14Ln_v1 + scale-free variants on the same cohort."""
    Br, Ar = load(base_path), load(arm_path)
    hb, ha = hitset(Br), hitset(Ar)
    inter = sorted(set(hb) & set(ha))
    # structural check (same as reader) + target string equality
    mism = [q for q in inter for k in ("n_estar", "nframes", "tgt_len", "video") if hb[q].get(k) != ha[q].get(k)]
    tmis = [q for q in inter if hb[q].get("gt_target_text") != ha[q].get("gt_target_text")]
    q2v = {int(r["qid"]): r.get("video") for r in Ar}
    vids = [q2v[q] for q in inter]
    rng = np.random.default_rng(0)
    out = {"n_H_int": len(inter), "n_hits_base": len(hb), "n_hits_arm": len(ha),
           "n_structural_mismatch": len(set(mism)), "n_target_text_mismatch": len(tmis),
           "R50_audit_base": float(np.mean([r.get("hit", 0) == 1 for r in Br])),
           "R50_audit_arm": float(np.mean([r.get("hit", 0) == 1 for r in Ar]))}
    for op in OPS:
        g = np.array([psi(ha[q], op) - psi(hb[q], op) for q in inter])
        ci, nv = cluster_boot(g, vids, rng, B)
        d = {"gamma": float(g.mean()), "gamma_ci95": ci, "n_videos": nv,
             "pos_frac": float((g > 0).mean()), "n_pos": int((g > 0).sum()), "n_zero": int((g == 0).sum()),
             "dE_arm": float(np.mean([float(ha[q][f"{op}_tokmean_drop_evid"]) for q in inter])),
             "dO_arm": float(np.mean([float(ha[q][f"{op}_tokmean_drop_rand"]) for q in inter])),
             "dE_base": float(np.mean([float(hb[q][f"{op}_tokmean_drop_evid"]) for q in inter])),
             "dO_base": float(np.mean([float(hb[q][f"{op}_tokmean_drop_rand"]) for q in inter]))}
        out[op] = d
    if extra:
        bt = Boot(vids, B=B, seed=1)
        nb = np.array([nll(hb[q]) for q in inter]); na = np.array([nll(ha[q]) for q in inter])
        out["nll_base"] = float(nb.mean()); out["nll_arm"] = float(na.mean())
        out["d_nll"] = float((na - nb).mean())
        out["d_nll_ci95"] = bt.ci(lambda ix: float((na[ix] - nb[ix]).mean()))
        for op in OPS:
            gc = np.array([cidx(ha[q], op) - cidx(hb[q], op) for q in inter])
            k = int((np.array([psi(ha[q], op) - psi(hb[q], op) for q in inter]) > 0).sum())
            out[op]["sign_frac"] = k / len(inter)
            out[op]["sign_frac_wilson95"] = wilson(k, len(inter))
            out[op]["sign_frac_cluster_ci95"] = bt.ci(lambda ix, gg=np.array([psi(ha[q], op) - psi(hb[q], op) for q in inter]): float((gg[ix] > 0).mean()))
            out[op]["gamma_c"] = float(gc.mean())
            out[op]["gamma_c_ci95"] = bt.ci(lambda ix, gg=gc: float(gg[ix].mean()))
    return out


def write_out(name, script_file, payload, inputs):
    """inputs: iterable of paths. Adds provenance and writes <name>.json next to this module."""
    ins = sorted(set(os.path.abspath(p).replace("\\", "/") for p in inputs))
    prov = {"script": os.path.basename(script_file), "script_sha16": sha16(script_file),
            "common_sha16": sha16(os.path.join(HERE, "_common.py")),
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs": [{"path": p, "sha16": sha16(p)} for p in ins]}
    out = {"provenance": prov, **payload}
    op = os.path.join(HERE, name + ".json")
    json.dump(out, open(op, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return op


def f3(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def fci(ci, d=3):
    if not ci or ci[0] is None:
        return "[n/a]"
    return f"[{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]"
