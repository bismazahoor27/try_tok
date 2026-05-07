#!/usr/bin/env python3
import sys
import json
import argparse
import os
import glob
import pandas as pd 

# --- APPLY LEGACY PATCHES (From your original script) ---
import six
import types
shim = types.ModuleType("rdkit.six")
shim.__dict__.update(six.__dict__)
sys.modules["rdkit.six"] = shim

import pandas as pd
if not hasattr(pd.DataFrame, 'append'):
    def _append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _append

_paths = glob.glob('/usr/local/lib/python3*/dist-packages/moses/metrics/utils.py')
if not _paths:
    _paths = glob.glob(os.path.expanduser('~/.local/lib/python3*/site-packages/moses/metrics/utils.py'))
if _paths:
    UTILS_PATH = _paths[0]
    with open(UTILS_PATH, 'r') as f:
        content = f.read()
    old = "_mcf.append(_pains, sort=True)['smarts'].values"
    new = "pd.concat([_mcf, _pains], sort=True)['smarts'].values"
    if old in content:
        content = content.replace(old, new)
        with open(UTILS_PATH, 'w') as f:
            f.write(content)

# --- IMPORT MOSES AFTER PATCHING ---
import moses
from moses.metrics import get_all_metrics

# --- PARSE ARGUMENTS FROM MAIN SCRIPT ---
parser = argparse.ArgumentParser()
parser.add_argument("--gen_path", required=True, help="Path to generated SMILES txt")
parser.add_argument("--n_jobs", type=int, default=8)
args = parser.parse_args()

# --- LOAD DATA ---
with open(args.gen_path, 'r') as f:
    valid_smiles = [line.strip() for line in f if line.strip()]

train_smiles = list(moses.get_dataset('train'))
test_smiles  = list(moses.get_dataset('test'))
test_sf      = list(moses.get_dataset('test_scaffolds'))

# --- COMPUTE METRICS ---
metrics = get_all_metrics(
    gen=valid_smiles,
    test=test_smiles,
    test_scaffolds=test_sf,
    train=train_smiles,
    n_jobs=args.n_jobs,
    device='cpu' # Safest to keep legacy eval on CPU to avoid CUDA conflicts
)

# --- PRINT JSON TO STDOUT ---
# This allows the main script to capture the dictionary perfectly
print(json.dumps(metrics))