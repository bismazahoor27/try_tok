#!/usr/bin/env python3
"""
python one_fold_moses_bpe_diff.py --bpe-variant bpe_right_linker  # default
python one_fold_moses_bpe_diff.py --bpe-variant bpe

FragSMILES RNN — 1-Fold
Run with: python run.py
Outputs saved to ~/working/
  - fold_K/  : per-fold model checkpoints, sampled SMILES, metrics JSON
  - cv_results.csv : per-fold metrics
  - cv_summary.csv : mean ± std across all folds
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys, shutil, os, types, gc, time, json, glob
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
WORK_DIR = os.path.expanduser('~/working')
REPO_DIR = os.path.expanduser('~/try_tok')
os.makedirs(WORK_DIR, exist_ok=True)

# Shared across all folds (BPE is fit once on the full training corpus)
CHECKPOINT_BASE = os.path.join(WORK_DIR, 'checkpoint_base_tokens.npy')
BPE_MERGES_PATH = os.path.join(WORK_DIR, 'bpe_merges.json')
VOCAB_PATH      = os.path.join(WORK_DIR, 'vocab.json')
TOKENS_OUT      = os.path.join(WORK_DIR, 'train_tokens.npy')

print(f'Working directory: {WORK_DIR}')
print(f'Repo directory:    {REPO_DIR}')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Clone repo if not present
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(REPO_DIR):
    subprocess.run(
        ['git', 'clone', 'https://github.com/bismazahoor27/try_tok.git', REPO_DIR],
        check=True
    )
else:
    print(f'Repo already exists at {REPO_DIR}, skipping clone.')

# Dependencies must be pre-installed in the conda environment.
# pip install "numpy<2" "pomegranate>=1.0.4" molsets --no-deps
# pip install scipy==1.13.0 fcd-torch tqdm matplotlib seaborn six rdkit
# pip install -e ~/try_tok --no-build-isolation

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Patch: rdkit.six shim
# ─────────────────────────────────────────────────────────────────────────────
import six
shim = types.ModuleType("rdkit.six")
shim.__dict__.update(six.__dict__)
sys.modules["rdkit.six"] = shim
print("rdkit.six shim applied ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Patch: moses/metrics/utils.py DataFrame.append removal
# ─────────────────────────────────────────────────────────────────────────────
_paths = glob.glob('/usr/local/lib/python3*/dist-packages/moses/metrics/utils.py')
if not _paths:
    _paths = glob.glob(os.path.expanduser('~/.local/lib/python3*/site-packages/moses/metrics/utils.py'))
UTILS_PATH = _paths[0] if _paths else None
print(f'moses utils path: {UTILS_PATH}')
if UTILS_PATH is None:
    raise FileNotFoundError('Could not find moses/metrics/utils.py')

with open(UTILS_PATH, 'r') as f:
    content = f.read()
old = "_mcf.append(_pains, sort=True)['smarts'].values"
new = "pd.concat([_mcf, _pains], sort=True)['smarts'].values"
if old in content:
    content = content.replace(old, new)
    with open(UTILS_PATH, 'w') as f:
        f.write(content)
    print("moses utils.py patched ✓")
else:
    print("moses utils.py already patched ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Patch: pandas.append compatibility shim
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
if not hasattr(pd.DataFrame, 'append'):
    def _append(self, other, ignore_index=False, **kwargs):
        other_df = other if isinstance(other, pd.DataFrame) else pd.DataFrame([other])
        return pd.concat([self, other_df], ignore_index=ignore_index)
    pd.DataFrame.append = _append
    print("pandas.append shim applied ✓")
else:
    print("pandas.append already present ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Import moses
# ─────────────────────────────────────────────────────────────────────────────
for key in list(sys.modules.keys()):
    if 'moses' in key:
        del sys.modules[key]

import moses
print(f"moses {moses.__version__} imported ✓  ({os.path.dirname(moses.__file__)})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Install r-fragSMILES from try_tok repo
# ─────────────────────────────────────────────────────────────────────────────
# result = subprocess.run(
#     [sys.executable, '-m', 'pip', 'install', REPO_DIR, '--no-build-isolation', '--quiet'],
#     capture_output=True, text=True
# )
# if result.returncode != 0:
#     print("pip install failed — falling back to sys.path insertion")
#     sys.path.insert(0, REPO_DIR)
# else:
#     print("r-fragSMILES (try_tok) installed via pip ✓")
sys.path.insert(0, REPO_DIR)
print("r-fragSMILES (try_tok) added to path ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Core imports + sanity check
# ─────────────────────────────────────────────────────────────────────────────
for key in list(sys.modules.keys()):
    if 'chemicalgof' in key:
        del sys.modules[key]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from chemicalgof import encode, decode, split
from chemicalgof.bpe import BPETrainer, BPETokenizer

RDLogger.DisableLog('rdApp.*')

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CORES = os.cpu_count()
# Cap N_JOBS at 8 — beyond that, joblib process overhead outweighs gains
N_JOBS    = min(8, max(4, NUM_CORES or 4))

print(f"Device:    {device}")
print(f"CPU cores: {NUM_CORES}")
print(f"N_JOBS:    {N_JOBS}")
if device.type == 'cuda':
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}:    {p.name}  {p.total_memory/1e9:.1f} GB")
else:
    print("WARNING: No GPU detected — training on CPU will be very slow.")
    print("         Consider checking nvidia-smi or your CUDA installation.")

_frag = encode('c1ccccc1')
_back = decode(_frag)
assert Chem.CanonSmiles('c1ccccc1') == Chem.CanonSmiles(_back), "Round-trip failed!"
print(f"Benzene r-fragSMILES: {_frag}")
print(f"Tokens:               {split(_frag)}")
print(f"Round-trip OK ✓\n=== Environment ready ===")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MOSES DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("Loading MOSES datasets...")
train_smiles = list(moses.get_dataset('train'))
test_smiles  = list(moses.get_dataset('test'))
test_sf      = list(moses.get_dataset('test_scaffolds'))

print(f"Train:          {len(train_smiles):,}")
print(f"Test:           {len(test_smiles):,}")
print(f"Test scaffolds: {len(test_sf):,}")

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
MAX_LEN        = 100
BATCH_SIZE     = 2048       # larger batches = fewer steps per epoch = faster wall time
N_EPOCHS       = 30
N_FOLDS        = 1
N_SAMPLES      = 30000     # sequences sampled per fold
CHUNK          = 200_000
BPE_MAX_MERGES = 2048
BPE_MIN_FREQ   = 500

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Encode full training corpus to base fragSMILES tokens (done once)
# ─────────────────────────────────────────────────────────────────────────────
def encode_to_base_tokens(smi: str):
    try:
        from chemicalgof import encode, split
        return split(encode(smi))
    except Exception:
        return None

if os.path.exists(CHECKPOINT_BASE):
    base_tokens = list(np.load(CHECKPOINT_BASE, allow_pickle=True))
    remaining   = train_smiles[len(base_tokens):]
    print(f"Resuming base encoding from {len(base_tokens):,} / {len(train_smiles):,}")
else:
    base_tokens = []
    remaining   = train_smiles
    print(f"Starting base encoding — {len(remaining):,} molecules")

t0      = time.time()
total_c = (len(remaining) + CHUNK - 1) // CHUNK

for i in range(0, len(remaining), CHUNK):
    chunk     = remaining[i : i + CHUNK]
    chunk_num = i // CHUNK + 1

    results = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(encode_to_base_tokens)(smi)
        for smi in tqdm(chunk, desc=f'Encoding chunk {chunk_num}/{total_c}', leave=False, dynamic_ncols=True)
    )

    base_tokens.extend([r for r in results if r is not None])
    np.save(CHECKPOINT_BASE, np.array(base_tokens, dtype=object))

    done = len(base_tokens)
    eta  = ((time.time() - t0) / done) * (len(remaining) - done + 1) if done > 0 else 0
    print(f"Chunk {chunk_num}/{total_c} | encoded: {done:,} | ETA: {eta/3600:.2f} hrs")

print(f"\nBase encoding done: {len(base_tokens):,}/{len(train_smiles):,} molecules ✓")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Train BPE on the full corpus (done once, shared across all folds)
# The BPE vocabulary is a tokenization scheme — it should be fixed before CV
# so that all folds use the same token space.
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists(BPE_MERGES_PATH) and os.path.exists(VOCAB_PATH) and os.path.exists(TOKENS_OUT):
    print("\nLoading existing BPE vocab ...")
    with open(VOCAB_PATH, 'r') as f:
        token2idx = json.load(f)
    vocab     = list(token2idx.keys())
    idx2token = {i: t for t, i in token2idx.items()}
    tokenizer = BPETokenizer.load(BPE_MERGES_PATH)
    print(f"Vocab size: {len(vocab):,} ✓")
else:
    print(f"\nTraining BPE on full corpus (max_merges={BPE_MAX_MERGES}, min_freq={BPE_MIN_FREQ}) ...")
    t_bpe   = time.time()
    trainer = BPETrainer()
    trainer.fit(base_tokens, max_merges=BPE_MAX_MERGES, min_freq=BPE_MIN_FREQ)
    trainer.save(BPE_MERGES_PATH)
    print(f"BPE training time: {time.time()-t_bpe:.1f}s")

    tokenizer = BPETokenizer(trainer)

    print("Applying BPE merges to full corpus ...")
    bpe_tokens   = [tokenizer.encode(seq) for seq in tqdm(base_tokens, desc='BPE encode')]
    all_toks     = [tok for seq in bpe_tokens for tok in seq]
    token_counts = Counter(all_toks)

    SPECIAL   = ['<pad>', '<sos>', '<eos>', '<unk>']
    vocab     = SPECIAL + [t for t, _ in token_counts.most_common()]
    token2idx = {t: i for i, t in enumerate(vocab)}
    idx2token = {i: t for t, i in token2idx.items()}

    with open(VOCAB_PATH, 'w') as f:
        json.dump(token2idx, f)

    avg_len_base = sum(len(s) for s in base_tokens) / len(base_tokens)
    avg_len_bpe  = sum(len(s) for s in bpe_tokens)  / len(bpe_tokens)
    print(f"Vocab size:      {len(vocab):,} ✓")
    print(f"Avg seq length:  {avg_len_base:.1f} base → {avg_len_bpe:.1f} after BPE")
    print(f"Token reduction: {(1 - avg_len_bpe/avg_len_base)*100:.1f}%")

    # Build and save the full indexed token array
    PAD = token2idx['<pad>']
    SOS = token2idx['<sos>']
    EOS = token2idx['<eos>']
    UNK = token2idx['<unk>']

    print("Indexing tokens into int32 array ...")
    rows = np.full((len(bpe_tokens), MAX_LEN), PAD, dtype=np.int32)
    for idx, toks in enumerate(tqdm(bpe_tokens, desc='Indexing')):
        ids = [SOS] + [token2idx.get(t, UNK) for t in toks] + [EOS]
        ids = ids[:MAX_LEN]
        rows[idx, :len(ids)] = ids
    np.save(TOKENS_OUT, rows)
    print(f"Saved {rows.shape} int32 array → {TOKENS_OUT} ✓")

PAD = token2idx['<pad>']
SOS = token2idx['<sos>']
EOS = token2idx['<eos>']
UNK = token2idx['<unk>']

# ─────────────────────────────────────────────────────────────────────────────
# DATASET (full — splits happen via KFold indices)
# ─────────────────────────────────────────────────────────────────────────────
class FragSMILESDataset(Dataset):
    def __init__(self, tokens_path=TOKENS_OUT):
        self.data = torch.from_numpy(np.load(tokens_path))
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

full_dataset = FragSMILESDataset()
print(f"\nFull dataset: {len(full_dataset):,} molecules, vocab: {len(vocab):,}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
class FragSMILES_RNN(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512,
                 num_layers=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn       = nn.LSTM(
            embed_size, hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout
        )
        self.fc      = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        emb         = self.dropout(self.embedding(x))
        out, hidden = self.rnn(emb, hidden)
        logits      = self.fc(self.dropout(out))
        return logits, hidden

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: decode a list of BPE token sequences → valid SMILES
# ─────────────────────────────────────────────────────────────────────────────
def decode_sequences(all_seqs, bpe_merges_path, n_jobs):
    def decode_worker(token_list):
        try:
            from chemicalgof import decode
            from chemicalgof.bpe import BPETokenizer
            from rdkit import Chem
            global _BPE_TOK
            try:
                _BPE_TOK
            except NameError:
                _BPE_TOK = BPETokenizer.load(bpe_merges_path)
            base_toks = _BPE_TOK.decode(token_list)
            smi = decode(base_toks, strict_chirality=False)
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol) if mol else None
        except Exception:
            return None

    decoded = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(decode_worker)(seq) for seq in tqdm(all_seqs, desc='  Decoding', leave=False)
    )
    return [s for s in decoded if s is not None]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: sample N_SAMPLES sequences from a trained model
# ─────────────────────────────────────────────────────────────────────────────
def sample_sequences(model, n_samples, sos_idx, idx2token, max_len, device):
    model.eval()
    all_seqs = []
    with torch.no_grad():
        while len(all_seqs) < n_samples:
            cur_bs    = min(512, n_samples - len(all_seqs))
            x         = torch.full((cur_bs, 1), sos_idx, dtype=torch.long, device=device)
            hidden    = None
            sequences = [[] for _ in range(cur_bs)]
            done      = [False] * cur_bs

            for _ in range(max_len):
                logits, hidden = model(x, hidden)
                probs  = F.softmax(logits[:, -1, :], dim=-1)
                next_t = torch.multinomial(probs, 1)
                for i in range(cur_bs):
                    if not done[i]:
                        tok = idx2token[next_t[i].item()]
                        if tok == '<eos>':
                            done[i] = True
                        elif tok not in ('<pad>', '<sos>'):
                            sequences[i].append(tok)
                x = next_t
                if all(done):
                    break
            all_seqs.extend(sequences)
    return all_seqs

# ─────────────────────────────────────────────────────────────────────────────
# 1-FOLD 
# ─────────────────────────────────────────────────────────────────────────────
all_indices = np.arange(len(full_dataset))
kf = [(all_indices, all_indices)]
fold_metrics   = []   # list of dicts, one per fold

print(f"\n{'='*60}")
print(f"  1 Fold")
print(f"  {len(full_dataset):,} molecules | {N_EPOCHS} epochs | {N_SAMPLES:,} samples/fold")
print(f"{'='*60}\n")

for fold, (train_idx, val_idx) in enumerate(kf, start=1):

    print(f"\n{'─'*60}")
    print(f"  FOLD {fold}/{N_FOLDS}  |  train: {len(train_idx):,} ")
    print(f"{'─'*60}")

    fold_dir = os.path.join(WORK_DIR, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    save_best   = os.path.join(fold_dir, 'rnn_best.pt')
    save_second = os.path.join(fold_dir, 'rnn_second_best.pt')
    smiles_out  = os.path.join(fold_dir, 'sampled_smiles.txt')
    metrics_out = os.path.join(fold_dir, 'metrics.json')

    # ── DataLoaders for this fold ────────────────────────────────────────────
    _pin  = device.type == 'cuda'   # pin_memory only helps with GPU
    _nw   = min(4, NUM_CORES // 2)  # parallel prefetch workers

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size       = BATCH_SIZE,
        shuffle          = True,
        num_workers      = _nw,
        pin_memory       = _pin,
        persistent_workers = False,
        prefetch_factor  = 2 if _nw > 0 else None,
        drop_last        = True
    )
    # val_loader = DataLoader(
    #     Subset(full_dataset, val_idx),
    #     batch_size       = BATCH_SIZE,
    #     shuffle          = False,
    #     num_workers      = _nw,
    #     pin_memory       = _pin,
    #     persistent_workers = False,
    #     prefetch_factor  = 2 if _nw > 0 else None,
    #     drop_last        = False
    # )

    # ── Fresh model for each fold ────────────────────────────────────────────
    model = FragSMILES_RNN(vocab_size=len(vocab)).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    scaler    = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_loss        = float('inf')
    second_best_loss = float('inf')
    fold_history     = []

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0.0
        t_start    = time.time()

        for i, batch in enumerate(train_loader):
            batch = batch.to(device, dtype=torch.long, non_blocking=True)
            x, y  = batch[:, :-1], batch[:, 1:]
            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(x)
                    loss = criterion(logits.reshape(-1, len(vocab)), y.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, len(vocab)), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

            if i % 200 == 0:
                print(f"  Fold {fold} | Epoch {epoch+1} | Batch {i}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f} | Time: {time.time()-t_start:.0f}s", flush=True)

        # ── Validation loss ──────────────────────────────────────────────────
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for batch in val_loader:
        #         batch = batch.to(device, dtype=torch.long, non_blocking=True)
        #         x, y  = batch[:, :-1], batch[:, 1:]
        #         if scaler:
        #             with torch.amp.autocast('cuda'):
        #                 logits, _ = model(x)
        #                 loss = criterion(logits.reshape(-1, len(vocab)), y.reshape(-1))
        #         else:
        #             logits, _ = model(x)
        #             loss = criterion(logits.reshape(-1, len(vocab)), y.reshape(-1))
        #         val_loss += loss.item()
        # val_loss /= len(val_loader)

        train_loss = total_loss / len(train_loader)
        elapsed    = time.time() - t_start
        fold_history.append({'fold': fold, 'epoch': epoch+1,
                              'train_loss': train_loss})
        scheduler.step(train_loss)
        torch.cuda.empty_cache()

        print(f"\n  {'='*50}", flush=True)
        print(f"  Fold {fold} | Epoch {epoch+1:02d}/{N_EPOCHS} | "
              f"Train: {train_loss:.4f} | Time: {elapsed:.1f}s", flush=True)
        print(f"  {'='*50}\n", flush=True)

        # ── Save best checkpoints (tracked by val_loss) ──────────────────────
        m = model.module if hasattr(model, 'module') else model

        if train_loss < best_loss:
            if best_loss < float('inf') and os.path.exists(save_best):
                shutil.copy(save_best, save_second)
                print(f"  Promoted previous best (val={best_loss:.4f}) to second best ✓", flush=True)
            second_best_loss = best_loss
            best_loss = train_loss
            torch.save({'model_state': {k: v.cpu() for k, v in m.state_dict().items()}, 'vocab': vocab,
                        'token2idx': token2idx,
                        'fold': fold, 'epoch': epoch+1}, save_best)
            print(f"  Saved new best (val={best_loss:.4f}) at epoch {epoch+1} ✓", flush=True)

        elif train_loss < second_best_loss:
            second_best_loss = train_loss
            torch.save({'model_state': {k: v.cpu() for k, v in m.state_dict().items()}, 'vocab': vocab,
                        'token2idx': token2idx,
                        'fold': fold, 'epoch': epoch+1}, save_second)
            print(f"  Saved new second best (val={second_best_loss:.4f}) at epoch {epoch+1} ✓", flush=True)

    # Save per-fold training history
    pd.DataFrame(fold_history).to_csv(os.path.join(fold_dir, 'loss_history.csv'), index=False)
    print(f"\n  Fold {fold} training complete. Best train loss: {best_loss:.4f} ✓")

    # ── Sample 6000 molecules from best model ────────────────────────────────
    print(f"  Sampling {N_SAMPLES:,} molecules from fold {fold} best model ...")
    ckpt = torch.load(save_best, map_location=device)
    sample_model = FragSMILES_RNN(vocab_size=len(vocab)).to(device)
    sample_model.load_state_dict(ckpt['model_state'])

    all_seqs = sample_sequences(
        sample_model, N_SAMPLES, SOS, idx2token, MAX_LEN, device
    )
    print(f"  Sampled {len(all_seqs):,} raw sequences ✓")

    # ── Decode to valid SMILES ────────────────────────────────────────────────
    print(f"  Decoding sequences → SMILES ...")
    valid_smiles = decode_sequences(all_seqs, BPE_MERGES_PATH, N_JOBS)
    validity_rate = len(valid_smiles) / len(all_seqs) * 100
    print(f"  Valid: {len(valid_smiles):,}/{len(all_seqs):,} ({validity_rate:.1f}%)")

    with open(smiles_out, 'w') as f:
        for smi in valid_smiles:
            f.write(smi + '\n')
    print(f"  Saved {smiles_out} ✓")

    # ── MOSES metrics ─────────────────────────────────────────────────────────
    print(f"  Computing MOSES metrics for fold {fold} ...")
    from moses.metrics import get_all_metrics
    metrics = get_all_metrics(
        gen            = valid_smiles,
        test           = test_smiles,
        test_scaffolds = test_sf,
        train          = train_smiles,
        n_jobs         = N_JOBS,
        device         = str(device)
    )
    metrics['fold']       = fold
    metrics['best_train_loss'] = best_loss
    metrics['validity']   = validity_rate / 100.0

    with open(metrics_out, 'w') as f:
        json.dump({k: float(v) if isinstance(v, float) else v
                   for k, v in metrics.items()}, f, indent=2)

    print(f"\n  Fold {fold} MOSES metrics:")
    for k, v in metrics.items():
        if k != 'fold':
            print(f"    {k:25s}: {round(v,4) if isinstance(v,float) else v}")

    fold_metrics.append(metrics)

    # Free GPU memory before next fold
    del model, sample_model, optimizer, scheduler
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE RESULTS: mean ± std across all 5 folds
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f" RESULTS  ({N_FOLDS} folds)")
print(f"{'='*60}")

results_df = pd.DataFrame(fold_metrics)
results_df.to_csv(os.path.join(WORK_DIR, 'cv_results.csv'), index=False)

# Compute mean and std for every numeric metric
numeric_cols = [c for c in results_df.columns if c != 'fold' and pd.api.types.is_numeric_dtype(results_df[c])]
summary_rows = []
for col in numeric_cols:
    mean = results_df[col].mean()
    std  = results_df[col].std()
    summary_rows.append({'metric': col, 'mean': mean, 'std': std,
                          'mean_pm_std': f"{mean:.4f} ± {std:.4f}"})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(WORK_DIR, 'cv_summary.csv'), index=False)

print(f"\n{'Metric':<25} {'Mean':>10} {'Std':>10}  {'95% CI (approx)'}")
print('─' * 65)
for _, row in summary_df.iterrows():
    ci = row['std'] * 1.96 / (N_FOLDS ** 0.5)
    print(f"  {row['metric']:<23} {row['mean']:>10.4f} {row['std']:>10.4f}   ± {ci:.4f}")

print(f"\nPer-fold results  → {os.path.join(WORK_DIR, 'cv_results.csv')}")
print(f"Summary (mean±std) → {os.path.join(WORK_DIR, 'cv_summary.csv')}")
print(f"\nDone ✓")
