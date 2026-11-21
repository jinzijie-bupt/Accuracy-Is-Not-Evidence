"""check_lora_keys.py -- make sure a LoRA adapter really loads under the installed transformers version.

Adapters saved under transformers 4.51 name their modules base_model.model.model.layers.* / base_model.model.visual.*;
transformers 5 names them base_model.model.model.language_model.layers.* / base_model.model.model.visual.*.
PEFT loads an adapter with the other naming without an error and leaves every LoRA weight at its initial value
(lora_B = 0), i.e. the adapter is a no-op. This script catches that.

  python tools/check_lora_keys.py --adapter <dir or hub repo> [--subfolder transformers-v5]
      static check: key tree of the file against the installed transformers major version (no GPU, no base weights)
  python tools/check_lora_keys.py --adapter <...> --base Qwen/Qwen2.5-VL-7B-Instruct --load
      full check: load base + adapter and assert that every tensor of the file landed on a module with the file's values

Also reports how many vision-tower lora_B matrices are all zero (transformers 4.51 runs with gradient checkpointing
left the vision LoRA untrained).

Use tools/convert_adapter_tf5keys.py to rename a transformers 4.51 adapter for transformers 5.
"""
import argparse
import os
import sys


def adapter_file(adapter, subfolder):
    local = os.path.join(adapter, subfolder or "", "adapter_model.safetensors")
    if os.path.isfile(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(adapter, "adapter_model.safetensors", subfolder=subfolder or None)


def key_tree(keys):
    if any(k.startswith("base_model.model.model.language_model.") for k in keys):
        return "tf5"
    if any(k.startswith("base_model.model.model.layers.") for k in keys):
        return "tf4"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="local adapter folder or Hugging Face repo id")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--base", default=None, help="base model (hub id or folder); needed with --load")
    ap.add_argument("--load", action="store_true", help="load base + adapter and compare every tensor")
    args = ap.parse_args()

    import transformers
    from safetensors.torch import load_file
    f = adapter_file(args.adapter, args.subfolder)
    ref = load_file(f)
    tree = key_tree(ref)
    major = int(transformers.__version__.split(".")[0])
    want = "tf5" if major >= 5 else "tf4"
    vis_b = [k for k in ref if ".visual." in k and "lora_B" in k]
    vis_zero = sum(1 for k in vis_b if not ref[k].any())
    print(f"file: {f}\ntensors: {len(ref)}  key tree: {tree}  installed transformers: {transformers.__version__}")
    print(f"vision lora_B matrices: {len(vis_b)}, all-zero: {vis_zero}")
    if tree != "unknown" and tree != want:
        sys.exit(f"MISMATCH: a {tree} adapter under transformers {transformers.__version__} loads as a no-op. "
                 f"Use the other transformers version or the converted copy (tools/convert_adapter_tf5keys.py).")
    print("static check OK")
    if not args.load:
        return
    if not args.base:
        sys.exit("--load needs --base")
    import torch
    from peft import PeftModel
    cfg = transformers.AutoConfig.from_pretrained(args.base)
    cls = {"qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration", "qwen3_vl": "Qwen3VLForConditionalGeneration"}.get(cfg.model_type)
    if cls is None:
        sys.exit(f"unsupported model_type {cfg.model_type}")
    model = getattr(transformers, cls).from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="auto")
    kw = {"subfolder": args.subfolder} if args.subfolder else {}
    model = PeftModel.from_pretrained(model, args.adapter, **kw)
    sd = {k.replace(".default.", "."): v for k, v in model.state_dict().items() if "lora_" in k}
    missing = [k for k in ref if k not in sd]
    if missing:
        sys.exit(f"FAIL: {len(missing)} of {len(ref)} adapter tensors matched no module, e.g. {missing[0]}")
    bad = [k for k, v in ref.items() if not torch.equal(sd[k].float().cpu(), v.to(sd[k].dtype).float())]
    if bad:
        sys.exit(f"FAIL: {len(bad)} tensors differ from the file, e.g. {bad[0]}")
    print(f"load check OK: {len(ref)} LoRA tensors loaded with the file's values")


if __name__ == "__main__":
    main()
