"""Byte-Pair Encoding (BPE) for r-fragSMILES token streams.

Only ring-fragment and connector tokens participate in merges; linker-run
tokens are left untouched so their chemical meaning is preserved.

Typical usage
-------------
Train::

    from chemicalgof import encode
    from chemicalgof.parse import split
    from chemicalgof.bpe import BPETrainer, BPETokenizer

    corpus = [split(encode(smi)) for smi in smiles_list]
    trainer = BPETrainer()
    trainer.fit(corpus, max_merges=2048, min_freq=50)
    trainer.save("bpe_merges.json")

Encode / decode::

    tokenizer = BPETokenizer.load("bpe_merges.json")
    merged = tokenizer.encode(split(encode(smiles)))
    original_tokens = tokenizer.decode(merged)
"""

from __future__ import annotations

import heapq
import json
import re
from collections import defaultdict
from typing import Iterable

# Patterns that identify tokens eligible for BPE merging.
# Ring fragments: any token that is NOT a bare heavy atom, NOT a connector,
# NOT a bracket-open/close, and NOT a linker-run.
_RE_CONNECTOR = re.compile(r'^<[0-9]+[RSrs]?>$')
_RE_BARE_ATOM = re.compile(r'^[BCNOPSFIcnops][l|r]?[\+\-]?[0-9]?$')
_RE_BRANCH = re.compile(r'^\(|\)$')

_MERGE_SEP = '\x00'  # internal separator used to join merged token pairs


def _is_mergeable(token: str) -> bool:
    """Return True if *token* may participate in a BPE merge."""
    if _RE_BRANCH.match(token):
        return False
    # connectors are mergeable (they glue ring pairs together)
    if _RE_CONNECTOR.match(token):
        return True
    # bare single atoms are NOT mergeable — they are linker atoms or
    # simple side-chains and should stay as cheap single tokens.
    if _RE_BARE_ATOM.match(token):
        return False
    # everything else (ring fragments, merged tokens) is mergeable
    return True


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


class BPETrainer:
    """Learn BPE merge rules from a corpus of fragSMILES token sequences.

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
        min_freq: int = 500,
    ) -> 'BPETrainer':
        """Run BPE on *corpus* and populate ``self.merges``.

        Parameters
        ----------
        corpus:
            Iterable of token-lists (output of ``chemicalgof.parse.split``).
        max_merges:
            Maximum number of merge operations (caps vocabulary growth).
        min_freq:
            Minimum pair frequency to keep merging.

        Returns
        -------
        self
        """
        work = [list(seq) for seq in corpus]

        # Collect initial vocabulary
        for seq in work:
            self.vocab.update(seq)

        # ------------------------------------------------------------------
        # Fast incremental BPE training
        # ------------------------------------------------------------------
        # pair_counts[pair]  – corpus-wide frequency (kept accurate)
        # pair_seqs[pair]    – set of sequence indices that contain the pair
        #                      (may have stale entries; treated as a superset)
        # heap               – max-heap of (-count, pair) with lazy deletion
        # ------------------------------------------------------------------
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        pair_seqs: dict[tuple[str, str], set] = defaultdict(set)

        for si, seq in enumerate(work):
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                if _is_mergeable(a) and _is_mergeable(b):
                    pair_counts[(a, b)] += 1
                    pair_seqs[(a, b)].add(si)

        # Build initial max-heap (negate counts for min-heap machinery)
        heap: list[tuple[int, tuple[str, str]]] = [
            (-cnt, pair) for pair, cnt in pair_counts.items()
        ]
        heapq.heapify(heap)

        for _ in range(max_merges):
            # Find the best valid pair (lazy-delete stale heap entries)
            best_pair: tuple[str, str] | None = None
            best_count = 0
            while heap:
                neg_cnt, pair = heapq.heappop(heap)
                actual = pair_counts.get(pair, 0)
                if actual == -neg_cnt and actual > 0:
                    best_pair = pair
                    best_count = actual
                    break
                # stale entry – discard and try next

            if best_pair is None or best_count < min_freq:
                break

            a, b = best_pair
            merged = _MERGE_SEP.join(best_pair)
            self.merges.append(best_pair)
            self.vocab.add(merged)

            # Remove the pair we just consumed
            affected = pair_seqs.pop(best_pair, set())
            del pair_counts[best_pair]

            # Apply merge to every affected sequence and update counts
            for si in affected:
                seq = work[si]
                n = len(seq)
                new_seq: list[str] = []
                i = 0
                while i < n:
                    if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                        prev_tok = new_seq[-1] if new_seq else None
                        rn = seq[i + 2] if i + 2 < n else None

                        # ── remove pair (prev_tok, a) ──────────────────────
                        if prev_tok is not None and _is_mergeable(prev_tok):
                            lp = (prev_tok, a)
                            pair_counts[lp] -= 1
                            if pair_counts[lp] <= 0:
                                pair_counts.pop(lp, None)

                        # ── remove pair (b, rn) ────────────────────────────
                        if rn is not None and _is_mergeable(rn):
                            rp = (b, rn)
                            pair_counts[rp] -= 1
                            if pair_counts[rp] <= 0:
                                pair_counts.pop(rp, None)

                        new_seq.append(merged)

                        # ── add pair (prev_tok, merged) ────────────────────
                        if prev_tok is not None and _is_mergeable(prev_tok):
                            nlp = (prev_tok, merged)
                            pair_counts[nlp] += 1
                            pair_seqs[nlp].add(si)
                            heapq.heappush(heap, (-pair_counts[nlp], nlp))

                        # ── add pair (merged, rn) ──────────────────────────
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
        """Serialise merge rules to a JSON file."""
        with open(path, 'w') as fh:
            json.dump(self.merges, fh)

    @classmethod
    def load(cls, path: str) -> 'BPETrainer':
        """Deserialise merge rules from a JSON file."""
        obj = cls()
        with open(path) as fh:
            obj.merges = [tuple(pair) for pair in json.load(fh)]
        return obj


class BPETokenizer:
    """Apply / reverse BPE merge rules on fragSMILES token sequences.

    Parameters
    ----------
    trainer:
        A fitted :class:`BPETrainer` instance (or one loaded from disk).
    """

    def __init__(self, trainer: BPETrainer) -> None:
        # Ordered list of (a, b) → merged_token rules
        self.merges: list[tuple[str, str]] = list(trainer.merges)
        # pair → (rank, merged_token): used for O(1) lookup during encode
        self._merge_rank: dict[tuple[str, str], tuple[int, str]] = {
            pair: (rank, _MERGE_SEP.join(pair))
            for rank, pair in enumerate(self.merges)
        }
        # Fast lookup: merged_token → (a, b)
        self._split_map: dict[str, tuple[str, str]] = {
            _MERGE_SEP.join(pair): pair for pair in self.merges
        }

    def encode(self, tokens: list[str]) -> list[str]:
        """Apply BPE merges to a token sequence in a single pass.

        Non-mergeable tokens (linker-run SMILES, branch brackets) pass through
        unchanged.
        """
        seq = list(tokens)
        # Single-pass: repeatedly find the lowest-rank (earliest) eligible pair
        # and merge it, until no more mergeable pairs exist.
        while True:
            best_rank = len(self.merges)  # sentinel: worse than any real rank
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
        """Reverse BPE merges to recover the original token sequence."""

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
        """Serialise tokenizer merge rules to a JSON file."""
        BPETrainer().__class__.save(
            type('_T', (), {'merges': self.merges})(), path  # type: ignore[arg-type]
        )

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load a tokenizer from a JSON file saved by :meth:`save`."""
        trainer = BPETrainer.load(path)
        return cls(trainer)
