import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from chemicalgof.bpe import BPETokenizer
import chemicalgof as cg
from rdkit import Chem

smiles = "COC(=O)C(CNC(=O)c1ccc(F)c(C)c1)Oc1ccc(F)cc1"

encoded = cg.encode(smiles)
tokens = cg.split(encoded)
tokenizer = BPETokenizer.load(os.path.join(os.path.dirname(__file__), '..', 'bpe_merges.json'))
bpe_tokens = tokenizer.encode(tokens)        # input must be pre-tokenized
frag_tokens_back = tokenizer.decode(bpe_tokens)
decoded = cg.decode(frag_tokens_back)

canonical_input   = Chem.CanonSmiles(smiles)
canonical_decoded = Chem.CanonSmiles(decoded)

print("SMILES:           ", smiles)
print("Encoded:          ", encoded)
print("BPE tokens:       ", bpe_tokens)
print("Decoded:          ", decoded)
print("Canonical input:  ", canonical_input)
print("Canonical decoded:", canonical_decoded)
print("Match:", canonical_input == canonical_decoded)