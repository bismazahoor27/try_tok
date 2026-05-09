"""Byte-Pair Encoding (BPE) for r-fragSMILES — STRICT linker-protection variant.

Difference vs. ``bpe.py``
-------------------------
The original :mod:`chemicalgof.bpe` only marks **bare single-atom** tokens
(``C``, ``N``, ``O``, …) as non-mergeable.  As a result, multi-atom *linker-run*
tokens produced by ``_collapse_linker_chains`` (e.g. ``CCN``, ``CC(=O)C``,
``COC``) and side-chain tokens (e.g. ``C(=O)``, ``C#N``) are still eligible
for BPE merging.  This module enforces the original r-fragSMILES design:

* **Only ring-fragment tokens and connector tokens are mergeable.**
* **Every other token type — bare atoms, multi-atom linker runs, side-chains,
  brackets — is excluded from BPE.**

How is a "ring fragment" identified without changing ``gof.py`` or
``parse.py``?
A ring SMILES is the *only* fragSMILES token class that contains a
ring-closure digit (``1``..``9`` or ``%10``..).  Linker runs are pure
open-chain aliphatic concatenations and never contain digits.
Connectors are matched by their own regex first, so the digit inside
``<0>`` is not confused with a ring closure.  This single rule cleanly
separates the two classes.

Public API is identical to :mod:`chemicalgof.bpe`:

    from chemicalgof.bpe_right_linker import BPETrainer, BPETokenizer
"""

from __future__ import annotations

import heapq
import json
import re
from collections import defaultdict
from typing import Iterable

# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------
_RE_CONNECTOR = re.compile(r'^<[0-9]+[RSrs]?>$')
_RE_BRANCH    = re.compile(r'^\(|\)$')
# A ring SMILES contains at least one ring-closure digit.
# We accept either a plain digit (1..9) or a multi-digit closure (%10, %11, …).
_RE_RING_DIGIT = re.compile(r'[0-9]')

_MERGE_SEP = '\x00'  # internal separator used to join merged token pairs


def _is_mergeable(token: str) -> bool:
    """Return True if *token* may participate in a BPE merge.

    Mergeable:
        * Connector tokens (``<i>``, ``<iR>``, ``<iS>``).
        * Ring-fragment tokens (any token containing a ring-closure digit).
        * Already-merged tokens (they always include a ring fragment, since
          merges can only originate from ring/connector pairs).

    Non-mergeable (left untouched by BPE):
        * Branch brackets ``(`` / ``)``.
        * Bare single-atom tokens (``C``, ``N``, ``O``, …).
        * Multi-atom linker-run tokens (``CCN``, ``CC(=O)C``, ``COC``, …).
        * Side-chain tokens (``C(=O)``, ``C#N``, …).
    """
    # Brackets
    if _RE_BRANCH.match(token):
        return False
    # Connectors (must be checked before the digit rule because they contain
    # a digit inside the angle brackets).
    if _RE_CONNECTOR.match(token):
        return True
    # Already-merged tokens: they always contain at least one ring fragment
    # (because merges only ever originate from a ring–connector–ring chain).
    if _MERGE_SEP in token:
        return True
    # Ring fragments contain at least one ring-closure digit.
    if _RE_RING_DIGIT.search(token):
        return True
    # Anything else (linker atoms, linker runs, side-chains) is non-mergeable.
    return False


# ---------------------------------------------------------------------------
# Helper functions (used by the simple, non-incremental fallback path)
# ---------------------------------------------------------------------------

def _count_pairs(corpus: list[list[str]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for seq in corpus:
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if _is_mergeable(a) and _is_mergeable(b):
                counts[(a, b)] += 1
    return counts


def _apply_merge(corpus: list[list[str]], pair: tuple[str, str]) -> list[list[str]]:
    merged_token = _MERGE_SEP.join(pair)
    a, b = pair
    new_corpus = []
    for seq in corpus:
        new_seq: list[str] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                new_seq.append(merged_token)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        new_corpus.append(new_seq)
    return new_corpus


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BPETrainer:
    """Learn BPE merge rules from a corpus of fragSMILES token sequences.

    Strict linker-protection variant: only ring-fragment and connector
    tokens are merged; every other token type is left untouched.

    Parameters
    ----------
    max_merges : int
        Hard cap on the number of merge rules to learn (vocabulary budget).
    min_freq : int
        Stop early when the most-frequent pair appears fewer than this many
        times in the corpus.
    """

    def __init__(self) -> None:
        self.merges: list[tuple[str, str]] = []
        self.vocab: set[str] = set()

    def fit(
        self,
        corpus: Iterable[list[str]],
        max_merges: int = 2048,
        min_freq: int = 50,
    ) -> 'BPETrainer':
        work = [list(seq) for seq in corpus]

        for seq in work:
            self.vocab.update(seq)

        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        pair_seqs: dict[tuple[str, str], set] = defaultdict(set)

        for si, seq in enumerate(work):
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                if _is_mergeable(a) and _is_mergeable(b):
                    pair_counts[(a, b)] += 1
                    pair_seqs[(a, b)].add(si)

        heap: list[tuple[int, tuple[str, str]]] = [
            (-cnt, pair) for pair, cnt in pair_counts.items()
        ]
        heapq.heapify(heap)

        for _ in range(max_merges):
            best_pair: tuple[str, str] | None = None
            best_count = 0
            while heap:
                neg_cnt, pair = heapq.heappop(heap)
                actual = pair_counts.get(pair, 0)
                if actual == -neg_cnt and actual > 0:
                    best_pair = pair
                    best_count = actual
                    break

            if best_pair is None or best_count < min_freq:
                break

            a, b = best_pair
            merged = _MERGE_SEP.join(best_pair)
            self.merges.append(best_pair)
            self.vocab.add(merged)

            affected = pair_seqs.pop(best_pair, set())
            del pair_counts[best_pair]

            for si in affected:
                seq = work[si]
                n = len(seq)
                new_seq: list[str] = []
                i = 0
                while i < n:
                    if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                        prev_tok = new_seq[-1] if new_seq else None
                        rn = seq[i + 2] if i + 2 < n else None

                        if prev_tok is not None and _is_mergeable(prev_tok):
                            lp = (prev_tok, a)
                            pair_counts[lp] -= 1
                            if pair_counts[lp] <= 0:
                                pair_counts.pop(lp, None)

                        if rn is not None and _is_mergeable(rn):
                            rp = (b, rn)
                            pair_counts[rp] -= 1
                            if pair_counts[rp] <= 0:
                                pair_counts.pop(rp, None)

                        new_seq.append(merged)

                        if prev_tok is not None and _is_mergeable(prev_tok):
                            nlp = (prev_tok, merged)
                            pair_counts[nlp] += 1
                            pair_seqs[nlp].add(si)
                            heapq.heappush(heap, (-pair_counts[nlp], nlp))

                        if rn is not None and _is_mergeable(rn):
                            nrp = (merged, rn)
                            pair_counts[nrp] += 1
                            pair_seqs[nrp].add(si)
                            heapq.heappush(heap, (-pair_counts[nrp], nrp))

                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1

                work[si] = new_seq

        return self

    def save(self, path: str) -> None:
        with open(path, 'w') as fh:
            json.dump(self.merges, fh)

    @classmethod
    def load(cls, path: str) -> 'BPETrainer':
        obj = cls()
        with open(path) as fh:
            obj.merges = [tuple(pair) for pair in json.load(fh)]
        return obj


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """Apply / reverse BPE merge rules — strict linker-protection variant."""

    def __init__(self, trainer: BPETrainer) -> None:
        self.merges: list[tuple[str, str]] = list(trainer.merges)
        self._merge_rank: dict[tuple[str, str], tuple[int, str]] = {
            pair: (rank, _MERGE_SEP.join(pair))
            for rank, pair in enumerate(self.merges)
        }
        self._split_map: dict[str, tuple[str, str]] = {
            _MERGE_SEP.join(pair): pair for pair in self.merges
        }

    def encode(self, tokens: list[str]) -> list[str]:
        seq = list(tokens)
        while True:
            best_rank = len(self.merges)
            best_i = -1
            best_merged = ""
            for i in range(len(seq) - 1):
                entry = self._merge_rank.get((seq[i], seq[i + 1]))
                if entry is not None:
                    rank, merged = entry
                    if rank < best_rank:
                        best_rank = rank
                        best_i = i
                        best_merged = merged
            if best_i == -1:
                break
            seq = seq[:best_i] + [best_merged] + seq[best_i + 2:]
        return seq

    def decode(self, tokens: list[str]) -> list[str]:
        def _expand(token: str) -> list[str]:
            if token in self._split_map:
                a, b = self._split_map[token]
                return _expand(a) + _expand(b)
            return [token]

        result: list[str] = []
        for token in tokens:
            result.extend(_expand(token))
        return result

    def save(self, path: str) -> None:
        with open(path, 'w') as fh:
            json.dump(self.merges, fh)

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        trainer = BPETrainer.load(path)
        return cls(trainer)
