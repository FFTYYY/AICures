import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
from rdkit.Chem import AllChem

fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

a = "CN1CC[C@@]23C=C[C@@H](C[C@@H]2OC4=C(C=CC(=C34)C1)OC)O.Br"

mol = Chem.MolFromSmiles(a)

feats = factory.GetFeaturesForMol(mol)
print (len(feats))
for f in feats:
	print ( list(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)) )