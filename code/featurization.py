import numpy as np 
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from typing import List, Union, Tuple

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),    # 原子序号
    'degree': [0, 1, 2, 3, 4, 5],                 # 原子总连接数
    'formal_charge': [-1, -2, 1, 2, 0],           # 原子形式电荷
    'chiral_tag': [0, 1, 2, 3],                   # 
    'num_Hs': [0, 1, 2, 3, 4],                    # 
    'hybridization': [                            # 原子杂化方式
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def get_atom_fdim():
    """ Gets the dimensionalitu of atom features.
    """
    return ATOM_FDIM


def get_bond_fdim(atom_messages: bool = False) -> int:
    """ 
    Gets the dimensionality of bond features.

    :param atom_messages: Whether atom messages are being used. 
    :return: The dimensionality of bond features.

    If atom_messages is true, only contains bond features. Otherwise contains both atom and bond features.
    """
    return BOND_FDIM + (not atom_messages) * get_atom_fdim()


def one_hot_encoding(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: one-hot encoding of the value in a list of length len(choices) + 1.

    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    """ 
    Builds a feature vector for an atom.

    :param atom: An rdkit atom.
    :return: a list containing the atom features.
    """
    features = one_hot_encoding(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               one_hot_encoding(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               one_hot_encoding(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               one_hot_encoding(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]    # scaled to about the same range as other features
    return features


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += one_hot_encoding(int(bond.GetStereo()), list(range(6)))
    return fbond


def morgan_features(smiles: str, radius: int = 2, num_bits: int = 512) -> np.ndarray:
    ''' 
    Generates a binary Morgan fingerprint for a molecule.

    :param mol: A SMILES string for a molecule.
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1-D numpy array containing the binary Morgan fingerprint.
    '''
    mol = Chem.MolFromSmiles(smiles)
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features

