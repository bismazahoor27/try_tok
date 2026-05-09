from chemicalgof import encode, decode_r
from chemicalgof.parse import split
from chemicalgof.bpe import BPETokenizer

tok = BPETokenizer.load("bpe_merges.json")

smiles = "CC(=O)Oc1ccccc1C(=O)O"          # ← change this

base   = split(encode(smiles))             # fragSMILES tokens (before BPE)
merged = tok.encode(base)                  # BPE-merged tokens
back   = decode_r(merged, tok, strict_chirality=False)  # back to SMILES

print("Base tokens :", base)
print("BPE tokens  :", merged)
print("Decoded     :", back)