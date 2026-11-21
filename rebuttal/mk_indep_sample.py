"""Independent audit sample for the rebuttal (R1 Q2 / R3 Q4).

Population: Charades-STA test questions (charades_sta_test.txt, 3,720 lines, qid = line index) whose video is
NOT among the videos of the first 200 questions (the main audit sample). Draw 600 questions uniformly at random
without replacement, seed 20261106, then sort by qid. Output: indep600_qids.json + a summary.
"""
import hashlib, json, os, random

ANNO = os.environ.get("CHARADES_TEST_ANNO", "data/charades_sta/anno/charades_sta_test.txt")  # Charades-STA test annotation (sha256 checked below)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indep600_qids.json")
raw = open(ANNO, "rb").read()
assert hashlib.sha256(raw).hexdigest()[:16] == "fab341679c20b091", "test anno drifted"
items = []
for i, line in enumerate(raw.decode("utf-8").splitlines()):
    if not line.strip():
        continue
    left, cap = line.split("##", 1)
    vid, s, e = left.split()
    items.append((i, vid, float(s), float(e)))
first200_videos = {v for q, v, _, _ in items[:200]}
pool = [q for q, v, _, _ in items if v not in first200_videos]
rng = random.Random(20261106)
sel = sorted(rng.sample(pool, 600))
json.dump(sel, open(OUT, "w"))
vids = {items[q][1] for q in sel}
lens = sorted(items[q][3] - items[q][2] for q in sel)
print("anno sha16", hashlib.sha256(raw).hexdigest()[:16], "n_items", len(items))
print("first200 videos", len(first200_videos), "pool", len(pool), "selected", len(sel), "videos", len(vids),
      "median len %.1fs" % lens[len(lens) // 2], "overlap with first-200 videos", len(vids & first200_videos))
print("wrote", OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:16])
