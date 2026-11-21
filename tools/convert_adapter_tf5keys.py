# -*- coding: utf-8 -*-
"""convert_adapter_tf5keys.py (2026-09-20) — rewrite a PEFT LoRA adapter saved under transformers 4.x Qwen2.5-VL module names
into the transformers 5.x module tree (silent no-op otherwise: peft finds 0 matching keys, lora_B norm 0).
  tf4:  base_model.model.visual.<...>            -> tf5: base_model.model.model.visual.<...>
  tf4:  base_model.model.model.layers.<...>      -> tf5: base_model.model.model.language_model.layers.<...>
Idempotent: keys already in tf5 layout are left as is. Refuses to write if the mapping is not a bijection.
usage: python convert_adapter_tf5keys.py <in_dir> <out_dir>   (copies adapter_config.json, writes CONVERTED_FROM.json with source sha16)
"""
import sys, os, json, hashlib, shutil
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1], sys.argv[2]
sd = load_file(os.path.join(src, "adapter_model.safetensors"))
def remap(k):
    if k.startswith("base_model.model.visual."):
        return "base_model.model.model.visual." + k[len("base_model.model.visual."):]
    if k.startswith("base_model.model.model.layers."):
        return "base_model.model.model.language_model.layers." + k[len("base_model.model.model.layers."):]
    return k
new = {remap(k): v for k, v in sd.items()}
n_changed = sum(1 for k in sd if remap(k) != k)
assert len(new) == len(sd), "remap is not injective"
if n_changed == 0:
    print(f"[convert] {src}: already tf5 layout ({len(sd)} keys); nothing to do"); sys.exit(0)
os.makedirs(dst, exist_ok=True)
save_file(new, os.path.join(dst, "adapter_model.safetensors"), metadata={"format": "pt"})
shutil.copy2(os.path.join(src, "adapter_config.json"), os.path.join(dst, "adapter_config.json"))
h = hashlib.sha256(open(os.path.join(src, "adapter_model.safetensors"), "rb").read()).hexdigest()[:16]
h2 = hashlib.sha256(open(os.path.join(dst, "adapter_model.safetensors"), "rb").read()).hexdigest()[:16]
json.dump({"source_dir": os.path.abspath(src), "source_sha16": h, "converted_sha16": h2, "keys": len(sd), "keys_remapped": n_changed,
           "rule": "tf4 Qwen2.5-VL module names -> tf5 (model.visual / model.language_model.layers)"}, open(os.path.join(dst, "CONVERTED_FROM.json"), "w"), indent=1)
print(f"[convert] {src} -> {dst}: keys={len(sd)} remapped={n_changed} src_sha16={h} dst_sha16={h2}")
