from rdkit import Chem
from .utils import CanonizeFragWithDummies, ClearSmiles, GetPotAtomLinkers
from .gof import DiGraphFrags, FragNode, classify_fragment_type
from rdkit import RDLogger
# from rdkit.Chem import rdCIPLabeler
import numpy as np
RDLogger.DisableLog('rdApp.*')

class Decompositer:
    # default cleavage pattern. exocyclic single bonds but not beetween charged atoms 
    SINGLEXOCYCLICPATT = '[!$([+1,-1]~[-1,+1])]-&!@[*]'

    def __init__(self, cleavage_smarts:str = SINGLEXOCYCLICPATT):
        self.cleavage_smarts = cleavage_smarts
        self.cleavage_pattern = Chem.MolFromSmarts(cleavage_smarts)

        ## initialized lists (pointed mutables) to be filled ##

        self.fragsMap = []
        self.fragsIdxs = []

        # dicts for mapping atom idxs 
        self.mapsFrag2Mol=[]
        self.mapsMol2Frag=[]

    def fragment(self, mol, bondMatches):
        dumLabels=[(0,0) for _ in bondMatches]
        bonds=[mol.GetBondBetweenAtoms(*atoms).GetIdx() for atoms in bondMatches]

        frags:tuple[str] = Chem.FragmentOnBonds(
            mol,
            addDummies=True,
            bondIndices=bonds, 
            dummyLabels=dumLabels,
        )

        fragsMol:tuple[Chem.Mol] = Chem.GetMolFrags(
            frags,
            asMols=True,
            frags=self.fragsIdxs,
            fragsMolAtomMapping=self.fragsMap
        )

        return fragsMol
    
    def frag_bonds(self, fragsMol, bondMatches):
        allBondNeighsFrags:list[list[int,int]] = [[] for _ in range(len(fragsMol))]
        for a,b in bondMatches:
            allBondNeighsFrags[self.fragsIdxs[a]].append((a,b))
            allBondNeighsFrags[self.fragsIdxs[b]].append((b,a))

        return allBondNeighsFrags
    
    def ultimate_smiles(self, fragsMol):
        # initialize list of cleared (no dummy atoms *) canonized smiles for each fragment
        pureSmis:list[str] = []

        for fMol, fMap in zip(fragsMol, self.fragsMap) :

            Chem.RemoveStereochemistry(fMol)
            # clear dummy atoms, canonize mol fragment to map old idxs with new idxs
            fMol, order = CanonizeFragWithDummies(fMol)

            # mapping idxs
            mapFrag2Mol={v:fMap[k] for k,v in order.items()}
            self.mapsFrag2Mol.append(mapFrag2Mol)

            mapMol2Frag=dict( zip(mapFrag2Mol.values(), mapFrag2Mol.keys()) )
            self.mapsMol2Frag.append(mapMol2Frag)

            # assert to have cleared smiles from mol
            s=ClearSmiles(Chem.MolToSmiles(fMol))

            pureSmis.append(s)

        return pureSmis
    
    def setup_nodes_attributes(self,frag_smiles, allChiralAtoms:dict[int,str], allAtomsInter:list[int]):

        nodes_attributes:list[dict[int,str]] = [] # only chirality attributes. str is R or S linked to atom idx

        for s,mapMol2Frag, mapFrag2Mol in zip(frag_smiles, self.mapsMol2Frag, self.mapsFrag2Mol) :
            n_atom_linkers = len(GetPotAtomLinkers(s)) # TODO I still dont like to put this here but it's forcing.

            # initialize chirality dictionary
            node_attributes = {}
            for _,atom_idx in sorted(mapFrag2Mol.items(), key=lambda x: x[0] ):
                # if frag has only one atom for binding, suffix not include atom idx FIXME in gof traverser when we need to write fragSMILES
                if atom_idx in allChiralAtoms and (atom_idx not in allAtomsInter or n_atom_linkers == 1):
                    node_attributes[ mapMol2Frag[atom_idx] ] = allChiralAtoms[atom_idx]

            nodes_attributes.append(node_attributes)

        return nodes_attributes

def Reduce2GoF(
        smiles:str = None, mol:Chem.Mol = None, # smiles or mol
        capitalize_legacy:bool = False,
        cleavage_smarts:str = Decompositer.SINGLEXOCYCLICPATT
    ) -> DiGraphFrags :
    """Reduce atom-based molecular graph (from smiles or mol object rdkit) into reduced graph fragment-based.

    Args:
        smiles (str, optional): input as smiles if mol=None. Defaults to None.
        mol (Chem.Mol, optional): input as mol object RDKit if smiles=None. Defaults to None.
        capitalize_legacy (bool, optional): if True, pseudo chirality labels (r or s) are forced to be capitalized (as actual chirality labels R or S) and stored as data graph.
        cleavage_smarts (str, optional): SMARTS pattern to employ for reduction (fragmentation) rule. Defaults to exocyclic single bonds fragmentation.

    Raises:
        ValueError: if input is invalid.

    Returns:
        DiGraphFrags: Reduced fragment-based molecular graph.
    """
        
    ## Providing canonical smiles and then canonical molecule representation
    if mol is not None and smiles is None:
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    elif mol is None and smiles is not None:
        mol = Chem.MolFromSmiles(Chem.CanonSmiles(smiles))
    else:
        raise ValueError('Input Error : SMILES string or Chem.Mol object is required.')
    
    obj = Decompositer(cleavage_smarts)

    bondMatches:tuple[tuple[int,int]] | tuple = mol.GetSubstructMatches( obj.cleavage_pattern )

    # optical stereochemical data
    allChiralAtoms = Chem.FindMolChiralCenters(mol, useLegacyImplementation=False ) # NOTE False -> r,s CipLabel also included
    if not capitalize_legacy:
        allChiralAtoms=dict(allChiralAtoms)
    else:
        allChiralAtoms = {idx: cip.upper() for idx,cip in allChiralAtoms}

    if not bondMatches: # 1 single fragment -> no edges within graph.
        fragsMol = [mol]
        frag_smiles = [Chem.MolToSmiles(mol)]
        frag_bonds = []
        nodes_attributes = [ {a:label for a,label in sorted(allChiralAtoms.items(), key=lambda x: x[0] )} ]
    else:
        fragsMol = obj.fragment(mol, bondMatches)

        # outer list follows frag idxs
        # inner lists : [ idxAtom_1frag, idxAtom_2frag  ], [ idxAtom_2frag, idxAtom_1frag  ]
        frag_bonds = obj.frag_bonds(fragsMol, bondMatches)

        frag_smiles = obj.ultimate_smiles(fragsMol)

        # store each atoms involved in bonds. It needs to label only chiral atom within fragment, not for edges
        from itertools import chain
        allAtomsInter = set(chain(*bondMatches))

        nodes_attributes = obj.setup_nodes_attributes(frag_smiles, allChiralAtoms, allAtomsInter)

    # initialize directed reduced graph
    diG=DiGraphFrags()

    #initialize nodes
    nodes = [FragNode(smiles=node_smiles, chirality=chirality,
                      node_type=classify_fragment_type(node_smiles))
             for node_smiles,chirality in zip(frag_smiles, nodes_attributes)]

    # add nodes and their attributes (loaded from node class attributes)
    diG.add_nodes_from(nodes)

    # add edges, if there are ...
    for    mapMol2Frag,  node,        fNeig,        in \
    zip(   obj.mapsMol2Frag, nodes, frag_bonds):

        # setup edges in directed graph !
        for a,nn in fNeig:
            neigh=nodes[obj.fragsIdxs[nn]]
            aB=mapMol2Frag[a]
            # Always attach stereo to the edge (CIP-rebuilding strategy).
            stereo = allChiralAtoms.get(a)
            diG.add_edge(node, neigh, aB=aB, stereo=stereo )

    _collapse_linker_chains(diG)

    return diG


def _collapse_linker_chains(diG: DiGraphFrags) -> None:
    """Merge consecutive linker nodes (degree ≤ 2, node_type == 'linker') into
    a single 'linker_run' FragNode whose SMILES is the concatenated chain.

    The operation is performed in-place on *diG*.
    """
    import networkx as nx

    changed = True
    while changed:
        changed = False
        G_undir = diG.to_undirected()
        for node in list(diG.nodes):
            if node not in diG.nodes:
                continue
            if node.node_type != 'linker':
                continue
            if G_undir.degree[node] != 2:
                continue

            neighbours = list(G_undir.neighbors(node))
            # Both neighbours must exist and must NOT be the same node (cycle guard)
            if len(neighbours) != 2 or neighbours[0] is neighbours[1]:
                continue

            left, right = neighbours[0], neighbours[1]

            # Build merged SMILES: left_smiles + node_smiles + right_smiles
            # Only merge when left or right is also a linker (to avoid
            # collapsing a linker directly bonded to two rings in one step;
            # iterating will handle multi-step chains).
            if left.node_type not in ('linker', 'linker_run') and \
               right.node_type not in ('linker', 'linker_run'):
                continue

            # Combine the smaller side (node) into the linker_run neighbour.
            # Prefer merging 'node' into whichever neighbour is already a run.
            run_node = left if left.node_type == 'linker_run' else right
            other_node = right if run_node is left else left

            # Build new SMILES by appending node's SMILES to the run's SMILES.
            # Simple concatenation works for open-chain aliphatic fragments.
            new_smiles = run_node.smiles + node.smiles

            # Validate merged SMILES
            if Chem.MolFromSmiles(new_smiles) is None:
                continue

            # Create replacement node
            merged = FragNode(smiles=new_smiles, chirality={},
                              node_type='linker_run')

            # Collect edges that connect run_node and other_node to the rest of
            # the graph (i.e. edges not involving 'node').
            edges_run  = [(u, v, diG.get_edge_data(u, v))
                          for u, v in list(diG.edges)
                          if (u is run_node or v is run_node)
                          and u is not node and v is not node]
            edges_other = [(u, v, diG.get_edge_data(u, v))
                           for u, v in list(diG.edges)
                           if (u is other_node or v is other_node)
                           and u is not node and v is not node]

            diG.add_node(merged)
            for u, v, data in edges_run + edges_other:
                new_u = merged if (u is run_node or u is other_node) else u
                new_v = merged if (v is run_node or v is other_node) else v
                if new_u is not new_v and not diG.has_edge(new_u, new_v):
                    diG.add_edge(new_u, new_v, **data)

            diG.remove_node(run_node)
            diG.remove_node(node)
            if other_node in diG.nodes:
                diG.remove_node(other_node)

            changed = True
            break  # restart scan after graph mutation

