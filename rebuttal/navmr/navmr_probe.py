"""probe: environment, imports, ckpt keys, feature coverage (no results)."""
import sys, os, json, importlib, hashlib
WORK_ROOT = os.environ.get("WORK_ROOT", "work")  # working directory on the GPU machine (models/, results/, the scripts)
ROOT = (WORK_ROOT + '/navmr')
print('py', sys.version)
for m in ['scipy', 'h5py', 'nncore', 'tqdm', 'einops', 'timm']:
    try:
        importlib.import_module(m); print('have', m)
    except Exception as e:
        print('MISSING', m, type(e).__name__)
import torch
for name, d in [('base', 'univtg_charades'), ('na', 'univtg_na_charades'), ('pre', 'pretrained_only')]:
    ck = torch.load(f'{ROOT}/dl/ckpt/{d}/model_best.ckpt', map_location='cpu', weights_only=False)
    print(name, 'top keys', list(ck.keys()), 'epoch', ck.get('epoch'))
    sd = ck['model']
    ks = list(sd.keys())
    print(name, 'n', len(ks), 'first', ks[:3], 'nonstd', [k for k in ks if k.startswith('module.')][:2])
    print(name, 'lstm/class_layer', [k for k in ks if 'lstm' in k or 'class_layer' in k])
    print(name, 'prefixes', sorted(set(k.split('.')[0] for k in ks)))
for p in sys.argv[1:]:
    print(p, os.path.exists(p), len(os.listdir(p)) if os.path.isdir(p) else '')
