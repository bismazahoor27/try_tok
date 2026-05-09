"""Train and apply r-fragSMILES BPE — supports both variants.

Usage examples
--------------
Train on a CSV column and save merges:

    python scripts/run_bpe.py train \\
        --input data/test.csv --column smiles \\
        --variant bpe_right_linker \\
        --max-merges 2048 --min-freq 50 \\
        --out bpe_merges.json

Encode / decode a single SMILES with an existing merges file:

    python scripts/run_bpe.py encode \\
        --smiles "CC(=O)Oc1ccccc1C(=O)O" \\
        --variant bpe_right_linker \\
        --merges bpe_merges.json

    python scripts/run_bpe.py roundtrip \\
        --smiles "CC(=O)Oc1ccccc1C(=O)O" \\
        --variant bpe \\
        --merges bpe_merges.json

The ``--variant`` flag selects which BPE module to import:
  * ``bpe``               — original (linker runs ARE merged)
  * ``bpe_right_linker``  — strict (linker runs are NOT merged)
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path
from typing import Iterable

# Ensure the project root (parent of this file's directory) is on sys.path so
# that `chemicalgof` can always be imported regardless of how the script is
# invoked (e.g. `python scripts/run_bpe.py` from the project root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from chemicalgof import encode as fragsmiles_encode
from chemicalgof import decode as fragsmiles_decode
from chemicalgof.parse import split as fragsmiles_split


VARIANTS = {
    "bpe": "chemicalgof.bpe",
    "bpe_right_linker": "chemicalgof.bpe_right_linker",
}


def _load_variant(name: str):
    if name not in VARIANTS:
        raise SystemExit(
            f"Unknown variant '{name}'. Choose one of: {', '.join(VARIANTS)}"
        )
    mod = importlib.import_module(VARIANTS[name])
    return mod.BPETrainer, mod.BPETokenizer


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _read_smiles(path: Path, column: str | None) -> list[str]:
    smiles: list[str] = []
    with path.open() as fh:
        if path.suffix.lower() == ".csv":
            reader = csv.reader(fh)
            first = next(reader, None)
            if first is None:
                return []
            # Detect a header row: header cells contain no SMILES-only chars.
            has_header = column is not None and column in first
            if has_header:
                col_idx = first.index(column)  # type: ignore[union-attr]
            else:
                col_idx = 0
                # First row is data — keep it.
                smiles.append(first[col_idx].strip())
            for row in reader:
                if row:
                    smiles.append(row[col_idx].strip())
        else:
            for line in fh:
                line = line.strip()
                if line:
                    smiles.append(line)
    return [s for s in smiles if s]


def _smiles_to_tokens(smiles_list: Iterable[str]) -> list[list[str]]:
    corpus: list[list[str]] = []
    skipped = 0
    for smi in smiles_list:
        try:
            fs = fragsmiles_encode(smi)
            corpus.append(fragsmiles_split(fs))
        except Exception:
            skipped += 1
    if skipped:
        print(f"[warn] skipped {skipped} SMILES that failed to encode",
              file=sys.stderr)
    return corpus


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    BPETrainer, _ = _load_variant(args.variant)

    print(f"[info] variant     : {args.variant}")
    print(f"[info] max_merges  : {args.max_merges}")
    print(f"[info] min_freq    : {args.min_freq}")
    print(f"[info] reading     : {args.input}")

    smiles = _read_smiles(Path(args.input), args.column)
    print(f"[info] {len(smiles)} SMILES loaded")

    corpus = _smiles_to_tokens(smiles)
    print(f"[info] {len(corpus)} sequences tokenized")

    trainer = BPETrainer().fit(
        corpus, max_merges=args.max_merges, min_freq=args.min_freq
    )
    print(f"[info] learned {len(trainer.merges)} merges "
          f"(vocab size {len(trainer.vocab)})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(out))
    print(f"[info] saved merges → {out}")


def cmd_encode(args: argparse.Namespace) -> None:
    _, BPETokenizer = _load_variant(args.variant)
    tokenizer = BPETokenizer.load(args.merges)

    fs = fragsmiles_encode(args.smiles)
    base_tokens = fragsmiles_split(fs)
    merged = tokenizer.encode(base_tokens)

    print("fragSMILES   :", fs)
    print("base tokens  :", base_tokens)
    print("merged tokens:", merged)


def cmd_decode(args: argparse.Namespace) -> None:
    _, BPETokenizer = _load_variant(args.variant)
    tokenizer = BPETokenizer.load(args.merges)

    merged = args.tokens
    base = tokenizer.decode(merged)
    smiles = fragsmiles_decode(base, strict_chirality=args.strict_chirality)

    print("merged tokens:", merged)
    print("base tokens  :", base)
    print("smiles       :", smiles)


def cmd_roundtrip(args: argparse.Namespace) -> None:
    _, BPETokenizer = _load_variant(args.variant)
    tokenizer = BPETokenizer.load(args.merges)

    fs = fragsmiles_encode(args.smiles)
    base = fragsmiles_split(fs)
    merged = tokenizer.encode(base)
    base_back = tokenizer.decode(merged)
    smi_back = fragsmiles_decode(base_back, strict_chirality=args.strict_chirality)

    print("input smiles :", args.smiles)
    print("fragSMILES   :", fs)
    print("base tokens  :", base)
    print("merged tokens:", merged)
    print("decoded base :", base_back)
    print("decoded smi  :", smi_back)
    print("bijective    :", base == base_back)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train / apply r-fragSMILES BPE (both variants).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    common_variant = dict(
        choices=list(VARIANTS),
        default="bpe_right_linker",
        help="Which BPE module to use (default: bpe_right_linker).",
    )

    # train
    pt = sub.add_parser("train", help="Train BPE on a SMILES corpus.")
    pt.add_argument("--input", required=True,
                    help="Path to SMILES file (.csv or .smi/.txt).")
    pt.add_argument("--column", default="smiles",
                    help="CSV column name with SMILES (default: smiles).")
    pt.add_argument("--variant", **common_variant)
    pt.add_argument("--max-merges", type=int, default=2048,
                    help="Vocabulary budget (default: 2048).")
    pt.add_argument("--min-freq", type=int, default=50,
                    help="Stop when top pair count < this (default: 50).")
    pt.add_argument("--out", default="bpe_merges.json",
                    help="Output merges file (default: bpe_merges.json).")
    pt.set_defaults(func=cmd_train)

    # encode
    pe = sub.add_parser("encode", help="Encode one SMILES into merged tokens.")
    pe.add_argument("--smiles", required=True)
    pe.add_argument("--variant", **common_variant)
    pe.add_argument("--merges", required=True,
                    help="Path to merges JSON saved by `train`.")
    pe.set_defaults(func=cmd_encode)

    # decode
    pd = sub.add_parser("decode", help="Decode a merged-token list to SMILES.")
    pd.add_argument("--tokens", nargs="+", required=True,
                    help="Merged token sequence (whitespace-separated).")
    pd.add_argument("--variant", **common_variant)
    pd.add_argument("--merges", required=True)
    pd.add_argument("--strict-chirality", action="store_true")
    pd.set_defaults(func=cmd_decode)

    # roundtrip
    pr = sub.add_parser(
        "roundtrip",
        help="Encode then decode one SMILES; verify token-level bijection.",
    )
    pr.add_argument("--smiles", required=True)
    pr.add_argument("--variant", **common_variant)
    pr.add_argument("--merges", required=True)
    pr.add_argument("--strict-chirality", action="store_true")
    pr.set_defaults(func=cmd_roundtrip)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
