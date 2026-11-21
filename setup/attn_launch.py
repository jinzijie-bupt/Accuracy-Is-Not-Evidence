"""attn_launch.py IMPL SCRIPT [args...] -- run SCRIPT as __main__ with every *.from_pretrained model load forced to
attn_implementation=IMPL (e.g. flash_attention_2). The script file itself is untouched (its sha self-check still
sees the original file). Writes IMPL into <--out>.attn.json next to the product so the product is traceable."""
import json, os, runpy, sys

impl, script, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
import transformers
from transformers import modeling_utils

_orig = modeling_utils.PreTrainedModel.from_pretrained.__func__


def _fp(cls, *a, **kw):
    kw["attn_implementation"] = impl
    return _orig(cls, *a, **kw)


modeling_utils.PreTrainedModel.from_pretrained = classmethod(_fp)
if "--out" in argv:
    o = argv[argv.index("--out") + 1]
    os.makedirs(os.path.dirname(os.path.abspath(o)), exist_ok=True)
    json.dump({"attn_implementation": impl, "transformers": transformers.__version__, "script": script},
              open(o + ".attn.json", "w"))
print(f"[attn_launch] attn_implementation={impl} transformers={transformers.__version__}", flush=True)
sys.argv = [script] + argv
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
