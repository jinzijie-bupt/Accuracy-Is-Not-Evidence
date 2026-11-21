"""merge_shards.py OUT.json SHARD1.json SHARD2.json ... -- concatenate row-list shards (14Ln / 12p4n / 12p5n products) into
one row list for the registered readers. Asserts: every shard is a row list, qids unique across shards, all shard metas share
the same args except --n/--offset/--out. Writes OUT.json and OUT.json.meta.json (= first shard's meta + merged_from
[{file, sha16, n_rows}]); text_first is carried over from the shards' meta so read_14Ln_v1 infers it from the merged meta."""
import hashlib, json, os, sys

out, shards = sys.argv[1], sys.argv[2:]
rows, src, seen, metas = [], [], set(), []
for s in shards:
    b = open(s, "rb").read()
    d = json.loads(b)
    assert isinstance(d, list), f"not a row list: {s}"
    for r in d:
        q = int(r["qid"])
        assert q not in seen, f"duplicate qid {q} ({s})"
        seen.add(q)
    rows += d
    src.append({"file": os.path.basename(s), "sha16": hashlib.sha256(b).hexdigest()[:16], "n_rows": len(d)})
    mp = s + ".meta.json"
    metas.append(json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else None)
IGN = {"n", "offset", "out"}
import re
# per-shard paths (14Ln tokc --base-pred-file points at the base's product for the SAME shard): compare them with the
# _o<off>n<n> shard token normalised, so a genuinely different base file still fails (11-07)
SHARD_PATH_ARGS = {"base_pred_file"}
_norm = lambda k, v: re.sub(r"_o\d+n\d+", "_o*n*", v) if k in SHARD_PATH_ARGS and isinstance(v, str) else v
a0 = {k: _norm(k, v) for k, v in (metas[0] or {}).get("args", {}).items() if k not in IGN}
for s, m in zip(shards, metas):
    if m is None:
        continue
    a = {k: _norm(k, v) for k, v in m.get("args", {}).items() if k not in IGN}
    assert a == a0, f"args differ in {s}: {sorted(k for k in set(a) | set(a0) if a.get(k) != a0.get(k))}"
rows.sort(key=lambda r: int(r["qid"]))
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
json.dump(rows, open(out, "w", encoding="utf-8"))
meta = dict(metas[0] or {})
meta["merged_from"] = src
json.dump(meta, open(out + ".meta.json", "w", encoding="utf-8"), indent=1)
print(f"{out}: {len(rows)} rows from {len(shards)} shards, sha16 {hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16]}")
