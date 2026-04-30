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
        min_freq: int = 50,
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

        for _ in range(max_merges):
            pairs = _count_pairs(work)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.__getitem__)
            if pairs[best_pair] < min_freq:
                break
            work = _apply_merge(work, best_pair)
            self.merges.append(best_pair)
            self.vocab.add(_MERGE_SEP.join(best_pair))

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
        # Fast lookup: merged_token → (a, b)
        self._split_map: dict[str, tuple[str, str]] = {
            _MERGE_SEP.join(pair): pair for pair in self.merges
        }

    def encode(self, tokens: list[str]) -> list[str]:
        """Apply BPE merges to a token sequence.

        Non-mergeable tokens (linker-run SMILES, branch brackets) pass through
        unchanged.
        """
        seq = list(tokens)
        for pair in self.merges:
            a, b = pair
            merged = _MERGE_SEP.join(pair)
            new_seq: list[str] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
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
