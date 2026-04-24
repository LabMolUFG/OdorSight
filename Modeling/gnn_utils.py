import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdmolops, Descriptors, AllChem
import numpy as np

def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(x == possible) for possible in permitted_list]
    return binary_encoding

def get_atom_features(atom):
    # Atomic number
    permitted_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Unknown']
    atom_type = atom.GetSymbol() if atom.GetSymbol() in permitted_atoms else 'Unknown'
    
    features = []
    features += one_hot_encoding(atom_type, permitted_atoms)
    
    # Degree
    features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    
    # Formal Charge
    features += one_hot_encoding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
    
    # Hybridization
    features += one_hot_encoding(str(atom.GetHybridization()), ["SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    # Aromaticity
    features += [int(atom.GetIsAromatic())]
    
    # Chirality
    permitted_chirality = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER]
    chirality = atom.GetChiralTag() if atom.GetChiralTag() in permitted_chirality else permitted_chirality[0]
    features += one_hot_encoding(chirality, permitted_chirality)
    
    # Total Num Hs
    features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    
    # Is in Ring
    features += [int(atom.IsInRing())]
    
    # Atomic Mass (Normalized roughly by 100)
    features += [atom.GetMass() * 0.01]
    
    return np.array(features, dtype=np.float32)

def get_bond_features(bond):
    # Bond Type
    permitted_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type = bond.GetBondType() if bond.GetBondType() in permitted_bond_types else permitted_bond_types[-1]
    
    features = []
    features += one_hot_encoding(bond_type, permitted_bond_types)
    
    # Stereochemistry
    permitted_stereo = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY, Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS]
    stereo = bond.GetStereo() if bond.GetStereo() in permitted_stereo else permitted_stereo[0]
    features += one_hot_encoding(stereo, permitted_stereo)

    # Conjugation
    features += [int(bond.GetIsConjugated())]

    return np.array(features, dtype=np.float32)

def smiles_to_data(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features
    # Compute global features
    try:
        mol_wt = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        labute_asa = Descriptors.LabuteASA(mol)  # Surface area
        mol_mr = Descriptors.MolMR(mol)  # Molar refractivity
        AllChem.ComputeGasteigerCharges(mol)
    except Exception as e:
        print(f"Error computing descriptors for {smiles}: {e}")
        return None

    # Node features
    node_features = []
    for atom in mol.GetAtoms():
        feats = list(get_atom_features(atom))
        
        # Partial charge
        try:
            charge = float(atom.GetProp('_GasteigerCharge'))
            if np.isnan(charge) or np.isinf(charge):
                charge = 0.0
        except:
            charge = 0.0
        feats.append(charge)
        
        # Global features (normalized roughly)
        feats.append(mol_wt * 0.01)
        feats.append(logp)
        feats.append(labute_asa * 0.01)  # Normalized surface area
        feats.append(mol_mr * 0.01)  # Normalized molar refractivity
        
        node_features.append(feats)
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    
    # Edges
    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bond_feat = get_bond_features(bond)
        
        edge_indices.append([i, j])
        edge_features.append(bond_feat)
        
        edge_indices.append([j, i])
        edge_features.append(bond_feat)
        
    if not edge_indices:
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_attr = torch.tensor([], dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)
        
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)
        
    return data
