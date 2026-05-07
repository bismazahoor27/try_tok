import chemicalgof as cg
from rdkit import Chem

smiles = "Cc1ccc2c(sc(-c3ccc(N(C)C)c(S(=O)(=O)O)c3)[n+]2C)c1S(=O)(=O)O"

encoded = cg.encode(smiles)
decoded = cg.decode(encoded)

canonical_input   = Chem.CanonSmiles(smiles)
canonical_decoded = Chem.CanonSmiles(decoded)

print("SMILES:           ", smiles)
print("Encoded:          ", encoded)
print("Decoded:          ", decoded)
print("Canonical input:  ", canonical_input)
print("Canonical decoded:", canonical_decoded)
print("Match:", canonical_input == canonical_decoded)
