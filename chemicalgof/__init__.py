from .reduce import Reduce2GoF
from .write import GoF2fragSMILES
from .explode import GoF2Mol
from .parse import fragSMILES2GoF, Sequence2GoF, split
from .bpe import BPETrainer, BPETokenizer

def encode(
    smiles:str,
    canonical:bool=True,
    random:bool=False,
    capitalize_chirality=True,
) -> str :
    """Convert SMILES string into fragSMILES representation. Reduced graph is considered as an intermediate of this function but it is not shown.

    Args:
        smiles (str): _description_
        canonical (bool, optional): Traversing intermediate reduced graph by canonical way. Defaults to True.
        random (bool, optional): Reduced intermediate graph is traversed randomly. Defaults to False.
        capitalize_chirality (bool, optional): If consider pseudo-chirality (r or s labels) as a actual chirality (R or S labels). Defaults to True.

    Returns:
        str: fragSMILES representation. Then string can be splitted by function provided by this package.
    """
    DiG = Reduce2GoF(smiles=smiles, capitalize_legacy=capitalize_chirality)
    fragsmiles = GoF2fragSMILES(DiG, canonize=canonical, random=random)

    return fragsmiles
    
def decode(
    fragsmiles:str | list[str],
    strict_chirality:bool = True,
) -> str :
    """Convert fragSMILES string (if string) or fragSMILES tokenized sequence (if list of strings) to relative SMILES.
    
    Args:
        fragsmiles (str | list[str]): string of fragSMILES representation or fragSMILES tokenized sequence (usefull if coversion involves generated sequence by models).
        strict_chirality (bool, optional): If take in account invalid assigned chirality label. Raise error when invalid labels are provided. Defaults to True.

    Returns:
        str: SMILES string. Tip: Canonize it for correct representation and compare invalid chiral atoms.
    """
    from rdkit import Chem
    if type(fragsmiles) is str:
        DiG = fragSMILES2GoF(fragsmiles)
    else:
        DiG = Sequence2GoF(fragsmiles)

    mol = GoF2Mol(DiG, strict_chirality=strict_chirality)
    smiles = Chem.MolToSmiles(mol)
    # smiles = Chem.CanonSmiles(smiles) # [x] Canonization is not preferred because of bug about chirality: it's still expected for aromatic and sp2 carbon atoms. If you canonize returned SMILES, sanification can be done on it!
    return smiles


def encode_r(
    smiles: str,
    tokenizer: BPETokenizer,
    canonical: bool = True,
    random: bool = False,
    capitalize_chirality: bool = True,
) -> list[str]:
    """r-fragSMILES encode: SMILES → BPE-merged token list.

    Runs the full r-fragSMILES pipeline:
      1. ``encode()``  — SMILES → fragSMILES string (with linker-run collapsing)
      2. ``split()``   — fragSMILES string → token list
      3. BPE encode   — merge frequent ring+connector pairs

    Args:
        smiles: Input SMILES string.
        tokenizer: Fitted :class:`BPETokenizer`.
        canonical: Use canonical graph traversal. Defaults to True.
        random: Use random graph traversal. Defaults to False.
        capitalize_chirality: Treat r/s pseudo-chirality as R/S. Defaults to True.

    Returns:
        List of r-fragSMILES tokens (after BPE merging).
    """
    fragsmiles = encode(smiles, canonical=canonical, random=random,
                        capitalize_chirality=capitalize_chirality)
    tokens = split(fragsmiles)
    return tokenizer.encode(tokens)


def decode_r(
    merged_tokens: list[str],
    tokenizer: BPETokenizer,
    strict_chirality: bool = False,
) -> str:
    """r-fragSMILES decode: BPE-merged token list → SMILES.

    Runs the full r-fragSMILES reverse pipeline:
      1. BPE decode   — expand merged tokens back to base fragSMILES tokens
      2. ``decode()`` — token list → SMILES

    Args:
        merged_tokens: BPE-encoded token list (output of :func:`encode_r`).
        tokenizer: Fitted :class:`BPETokenizer` (must match the one used in encoding).
        strict_chirality: If True, raise on invalid chirality after assembly.
            Defaults to False (spurious stereo is silently dropped).

    Returns:
        SMILES string.
    """
    tokens = tokenizer.decode(merged_tokens)
    return decode(tokens, strict_chirality=strict_chirality)