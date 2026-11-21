"""wait_launch.py FILE TIMEOUT_SEC SCRIPT [args...] -- wait until FILE exists (and its size is stable for 10 s), then run
SCRIPT as __main__ with args (same runpy mechanism as attn_launch.py, so SCRIPT may itself be attn_launch.py). Exits 3 if
FILE does not appear within TIMEOUT_SEC (q.py then marks the job FAIL and it can be requeued)."""
import os, runpy, sys, time

f, tmo, script, argv = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4:]
t0 = time.time(); last = -1
while True:
    if os.path.exists(f):
        s = os.path.getsize(f)
        if s == last and s > 0:
            break
        last = s
    if time.time() - t0 > tmo:
        print(f"[wait_launch] timeout waiting for {f}", flush=True); sys.exit(3)
    time.sleep(10)
print(f"[wait_launch] {f} ready after {time.time() - t0:.0f}s", flush=True)
sys.argv = [script] + argv
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
