import chemicalgof as cg
from rdkit import Chem

smiles = "Nc1c(C(=O)N2CCOCC2)sc2nc3c(cc12)CCCCC3"

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
