# read_newly_correct_v1.py
# Finding-3 support: of the questions an arm newly localises correctly (arm clean hit, base clean miss),
# how many are (a) solvable by the constant (0,10)s window with no model, (b) still answered correctly
# on an all-black video, and -- for the GRPO arm, where regenerated perturbation products exist --
# (c) still correct under evidence mask vs matched placebo mask.
# Zero GPU: reads per-question generation products already on disk.
#
# Acceptance (positive controls, all must pass before any number is trusted):
#   PC1  qid<->caption match across paired files: 0 mismatches allowed.
#   PC2  recomputed clean R@0.5 within 0.002 of the registry value for that cell.
#   PC3  own IoU function agrees with the p5n product's none_iou on every row (<1e-6).
import os as _os_supp
_SUPP = _os_supp.path.abspath(_os_supp.path.join(_os_supp.path.dirname(_os_supp.path.abspath(__file__)), "..")).replace("\\", "/")
import io, json, os, sys, hashlib
import numpy as np

ROOT = _SUPP + "/results/native_full_20260918"
OUT_DIR = _SUPP + "/results/judgment_20260923"
os.makedirs(OUT_DIR, exist_ok=True)
B = 2000
RNG = np.random.default_rng(20260923)

def iou(pred, gs, ge):
    if pred is None or not isinstance(pred, (list, tuple)) or len(pred) != 2: return 0.0
    a, b = float(pred[0]), float(pred[1])
    if b < a: a, b = b, a
    inter = max(0.0, min(b, ge) - max(a, gs))
    union = max(b, ge) - min(a, gs)
    return inter / union if union > 0 else 0.0

def load_rows(path):
    d = json.load(io.open(path, encoding="utf-8"))
    if isinstance(d, dict): d = d.get("rows") or d.get("items")
    return d

def load_jsonl(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]

def hit_map(rows, pred_key="pred"):
    m = {}
    for r in rows:
        m[r["qid"]] = dict(video=r["video"], caption=r["caption"],
                           hit=iou(r.get(pred_key), r["gsA"], r["geA"]) >= 0.5,
                           gs=r["gsA"], ge=r["geA"])
    return m

def cw_solvable(gs, ge):  # constant (0,10) s window
    return iou([0.0, 10.0], gs, ge) >= 0.5

def boot_rate(flags, vids):
    flags = np.asarray(flags, float); vids = np.asarray(vids)
    uv = np.unique(vids); idx = {v: np.where(vids == v)[0] for v in uv}
    stats = []
    for _ in range(B):
        pick = RNG.choice(uv, size=len(uv), replace=True)
        sel = np.concatenate([idx[v] for v in pick])
        stats.append(flags[sel].mean())
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

def check_captions(a, b, scope):
    common = sorted(set(a) & set(b))
    bad = [q for q in common if a[q]["caption"] != b[q]["caption"]]
    assert not bad, f"PC1 FAIL {scope}: {len(bad)} caption mismatches, e.g. qid {bad[:3]}"
    return common

def analyse(tag, base_clean, arm_clean, arm_black=None, base_black=None, registry_R50=None):
    qs = check_captions(base_clean, arm_clean, tag)
    armR = np.mean([arm_clean[q]["hit"] for q in qs])
    if registry_R50 is not None:
        assert abs(armR - registry_R50) < 0.002, f"PC2 FAIL {tag}: recomputed {armR:.4f} vs registry {registry_R50:.4f}"
    N = [q for q in qs if arm_clean[q]["hit"] and not base_clean[q]["hit"]]
    lost = [q for q in qs if not arm_clean[q]["hit"] and base_clean[q]["hit"]]
    out = dict(tag=tag, n_common=len(qs), arm_R50_recomputed=round(float(armR), 4),
               n_newly_correct=len(N), n_lost=len(lost))
    vN = [arm_clean[q]["video"] for q in N]
    vA = [arm_clean[q]["video"] for q in qs]
    cwN = [cw_solvable(arm_clean[q]["gs"], arm_clean[q]["ge"]) for q in N]
    cwA = [cw_solvable(arm_clean[q]["gs"], arm_clean[q]["ge"]) for q in qs]
    out["cw_solvable_on_newly"] = dict(rate=round(float(np.mean(cwN)), 4), ci=boot_rate(cwN, vN), n=len(N))
    out["cw_solvable_on_all"] = dict(rate=round(float(np.mean(cwA)), 4), ci=boot_rate(cwA, vA), n=len(qs))
    if arm_black is not None:
        qsb = check_captions(arm_clean, arm_black, tag + "/black")
        Nb = [q for q in N if q in arm_black]
        blN = [arm_black[q]["hit"] for q in Nb]
        blA = [arm_black[q]["hit"] for q in qsb]
        out["black_retained_on_newly"] = dict(rate=round(float(np.mean(blN)), 4),
                                              ci=boot_rate(blN, [arm_black[q]["video"] for q in Nb]), n=len(Nb))
        out["black_R50_all"] = dict(rate=round(float(np.mean(blA)), 4), n=len(qsb))
        if base_black is not None:
            qbb = [q for q in qsb if q in base_black]
            out["base_black_R50_all"] = round(float(np.mean([base_black[q]["hit"] for q in qbb])), 4)
    return out, set(N)

results = {"reader": "read_newly_correct_v1", "B": B, "inputs": {}}

def reg(path):
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    results["inputs"][os.path.basename(path)] = h
    return path

# ---------------- env B : IoU-GRPO ----------------
baseB = hit_map(load_rows(reg(f"{ROOT}/p4nat_base_timer1_default_tf5meta_node1_full.json")))
g1 = load_rows(reg(f"{ROOT}/p5n_x0s0iou700_timer1_default_node1_n1000_o0.json"))
# PC3 on the p5n file
for r in g1:
    mine = iou(r.get("none_pred"), r["gsA"], r["geA"])
    assert abs(mine - r["none_iou"]) < 1e-6, f"PC3 FAIL qid {r['qid']}"
g2 = load_rows(reg(f"{ROOT}/p4nat_x0s0iou700_timer1_default_tf5meta_node1_n2720_o1000.json"))
grpo_clean = hit_map(g1, "none_pred"); grpo_clean.update(hit_map(g2))
grpo_black = hit_map(load_rows(reg(f"{ROOT}/p4nat_x0s0iou700_timer1_default_black_node1_full.json")))
base_blackB = hit_map(load_rows(reg(f"{ROOT}/p4nat_qwenbase_timer1_default_black_node3_full.json")))
res, N_grpo = analyse("iou-grpo envB", baseB, grpo_clean, grpo_black, base_blackB)
# behavioural channel on the first 1000 (evidence mask vs placebo mask, regenerated)
p5b = {r["qid"]: r for r in load_rows(reg(f"{ROOT}/p5n_qwenbase_timer1_default_node1_n1000_o0.json"))}
p5g = {r["qid"]: r for r in g1}
N1k = [q for q in N_grpo if q in p5g and q in p5b]
ev = [iou(p5g[q].get("mask_evid_pred"), p5g[q]["gsA"], p5g[q]["geA"]) >= 0.5 for q in N1k]
pl = [iou(p5g[q].get("mask_rand_pred"), p5g[q]["gsA"], p5g[q]["geA"]) >= 0.5 for q in N1k]
vids1k = [p5g[q]["video"] for q in N1k]
res["evidence_mask_retained_on_newly_n1000"] = dict(rate=round(float(np.mean(ev)), 4), ci=boot_rate(ev, vids1k), n=len(N1k))
res["placebo_mask_retained_on_newly_n1000"] = dict(rate=round(float(np.mean(pl)), 4), ci=boot_rate(pl, vids1k), n=len(N1k))
diff = np.asarray(pl, float) - np.asarray(ev, float)
res["placebo_minus_evidence_diff"] = dict(mean=round(float(diff.mean()), 4), ci=boot_rate(diff, vids1k))
results["iou_grpo_envB"] = res

# ---------------- env A : grouped arms ----------------
baseA = hit_map(load_rows(reg(f"{ROOT}/p4nat_qwenbase_timer1_default_tf5meta_node2_full.json")))
base_blackA = hit_map(load_rows(reg(f"{ROOT}/p4nat_qwenbase_timer1_default_black_node2_full.json")))
for tag, cf, bf in [
    ("grouped a05 envA", "p4nat_natbase_a05_g1_s0_tf5_timer1_default_tf5meta_node2_full.json",
     "p4nat_natbase_a05_g1_s0_tf5_timer1_default_black_node2_full.json"),
    ("grouped a10 envA", "p4nat_natbase_a10_g1_s0_timer1_default_tf5meta_node2_full.json",
     "p4nat_natbase_a10_g1_s0_timer1_default_black_node2_full.json"),
]:
    arm_c = hit_map(load_rows(reg(f"{ROOT}/{cf}")))
    arm_b = hit_map(load_rows(reg(f"{ROOT}/{bf}")))
    r, _ = analyse(tag, baseA, arm_c, arm_b, base_blackA)
    results[tag.replace(" ", "_")] = r

# ---------------- env C : pure SFT (994-question covered subset of the stopped base run) ----------------
baseC = hit_map(load_jsonl(reg(f"{ROOT}/p4nat_qwenbase_timer1_default_tf451_node4_partial994.jsonl")))
sft_c = hit_map(load_rows(reg(f"{ROOT}/p4nat_natsft_b0_s0_timer1_default_tf451_node4_full.json")))
sft_b = hit_map(load_rows(reg(f"{ROOT}/p4nat_natsft_b0_s0_timer1_default_black_node4_full.json")))
sft_c994 = {q: v for q, v in sft_c.items() if q in baseC}
r, _ = analyse("sft envC subset994", baseC, sft_c994, sft_b)
results["sft_envC_subset994"] = r

out_path = f"{OUT_DIR}/newly_correct_v1.json"
with io.open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
print("WROTE", out_path)
for k, v in results.items():
    if isinstance(v, dict) and "n_newly_correct" in v:
        print(f"== {v['tag']}: common {v['n_common']}, newly {v['n_newly_correct']} (lost {v['n_lost']}), armR {v['arm_R50_recomputed']}")
        print(f"   const-window solvable: newly {v['cw_solvable_on_newly']['rate']} {v['cw_solvable_on_newly']['ci']} vs all {v['cw_solvable_on_all']['rate']}")
        if "black_retained_on_newly" in v:
            print(f"   black retained on newly: {v['black_retained_on_newly']['rate']} {v['black_retained_on_newly']['ci']} (arm black all {v['black_R50_all']['rate']}, base black {v.get('base_black_R50_all')})")
        for kk in ("evidence_mask_retained_on_newly_n1000", "placebo_mask_retained_on_newly_n1000", "placebo_minus_evidence_diff"):
            if kk in v: print(f"   {kk}: {v[kk]}")
