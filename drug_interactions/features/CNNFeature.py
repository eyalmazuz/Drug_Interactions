import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.info')
from tqdm import tqdm

class CNNFeature():

    def __init__(self, **kwargs):

        self.atom_size = kwargs["atom_size"]
        self.atom_info = kwargs["atom_info"]
        self.struct_info = kwargs["struct_info"]

    def __repr__(self,):
        return "CNNFeature"

    def islower(self, s):
        lowerReg = re.compile(r'^[a-z]+$')
        return lowerReg.match(s) is not None

    def isupper(self, s):
        upperReg = re.compile(r'^[A-Z]+$')
        return upperReg.match(s) is not None

    def calc_atom_feature(self, atom):
        
        Chiral = {"CHI_UNSPECIFIED":0,  "CHI_TETRAHEDRAL_CW":1, "CHI_TETRAHEDRAL_CCW":2, "CHI_OTHER":3}
        Hybridization = {"UNSPECIFIED":0, "S":1, "SP":2, "SP2":3, "SP3":4, "SP3D":5, "SP3D2":6, "OTHER":7}
        
        if atom.GetSymbol() == 'H':   feature = [1,0,0,0,0]
        elif atom.GetSymbol() == 'C': feature = [0,1,0,0,0]
        elif atom.GetSymbol() == 'O': feature = [0,0,1,0,0]
        elif atom.GetSymbol() == 'N': feature = [0,0,0,1,0]
        else: feature = [0,0,0,0,1]
            
        feature.append(atom.GetTotalNumHs()/8)
        feature.append(atom.GetTotalDegree()/4)
        feature.append(atom.GetFormalCharge()/8)
        feature.append(atom.GetTotalValence()/8)
        feature.append(atom.IsInRing()*1)
        feature.append(atom.GetIsAromatic()*1)

        f =  [0]*(len(Chiral)-1)
        if Chiral.get(str(atom.GetChiralTag()), 0) != 0:
            f[Chiral.get(str(atom.GetChiralTag()), 0)] = 1
        feature.extend(f)

        f =  [0]*(len(Hybridization)-1)
        if Hybridization.get(str(atom.GetHybridization()), 0) != 0:
            f[Hybridization.get(str(atom.GetHybridization()), 0)] = 1
        feature.extend(f)
        
        return(feature)


    def calc_structure_feature(self, c, flag, label, struct_info):
        feature = [0] * struct_info

        if c== '(' :
            feature[0] = 1
            flag = 0
        elif c== ')' :
            feature[1] = 1
            flag = 0
        elif c== '[' :
            feature[2] = 1
            flag = 0
        elif c== ']' :
            feature[3] = 1
            flag = 0
        elif c== '.' :
            feature[4] = 1
            flag = 0
        elif c== ':' :
            feature[5] = 1
            flag = 0
        elif c== '=' :
            feature[6] = 1
            flag = 0
        elif c== '#' :
            feature[7] = 1
            flag = 0
        elif c== '\\':
            feature[8] = 1
            flag = 0
        elif c== '/' :
            feature[9] = 1
            flag = 0  
        elif c== '@' :
            feature[10] = 1
            flag = 0
        elif c== '+' :
            feature[11] = 1
            flag = 1
        elif c== '-' :
            feature[12] = 1
            flag = 1
        elif c.isdigit() == True:
            if flag == 0:
                if c in label:
                    feature[20] = 1
                else:
                    label.append(c)
                    feature[19] = 1
            else:
                feature[int(c)-1+12] = 1
                flag = 0
        return(feature,flag,label)


    def calc_featurevector(self, mol, smiles):
        flag = 0
        label = []
        molfeature = []
        idx = 0
        j = 0
        H_Vector = [0] * self.atom_info
        H_Vector[0] = 1
        lensize = self.atom_info + self.struct_info

                
        for c in smiles:
            if self.islower(c) == True: continue
            elif self.isupper(c) == True:
                if c == 'H':
                    molfeature.extend(H_Vector)
                else:
                    molfeature.extend(self.calc_atom_feature(rdchem.Mol.GetAtomWithIdx(mol, idx)))
                    idx = idx + 1
                molfeature.extend([0]*self.struct_info)
                j = j +1
                
            else:   
                molfeature.extend([0] * self.atom_info)
                f, flag, label = self.calc_structure_feature(c, flag, label, self.struct_info)
                molfeature.extend(f)
                j = j +1

        #0-Padding
        molfeature.extend([0]*(self.atom_size-j)*lensize)        
        return(molfeature)


    def mol_to_feature(self, mol, n):
        try: defaultSMILES = Chem.MolToSmiles(mol, kekuleSmiles=False, isomericSmiles=True, rootedAtAtom=int(n)) # pylint: disable=maybe-no-member
        except: defaultSMILES = Chem.MolToSmiles(mol, kekuleSmiles=False, isomericSmiles=True) # pylint: disable=maybe-no-member
        try: isomerSMILES = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True, rootedAtAtom=int(n)) # pylint: disable=maybe-no-member
        except: isomerSMILES = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True) # pylint: disable=maybe-no-member
        return self.calc_featurevector(Chem.MolFromSmiles(defaultSMILES), isomerSMILES) # pylint: disable=maybe-no-member


    def __call__(self, old_drug_bank, new_drug_bank):
        train_drug_ids = set(old_drug_bank.id_to_drug.keys())
        test_drug_ids = set(new_drug_bank.id_to_drug.keys())
        new_drug_ids = test_drug_ids - (train_drug_ids & test_drug_ids)

        drug_to_smiles = {}
        for drug_id in train_drug_ids:
            drug_to_smiles[drug_id] = old_drug_bank.id_to_drug[drug_id].smiles

        test_drug_to_smiles = {}
        for drug_id in new_drug_ids:
            test_drug_to_smiles[drug_id] = new_drug_bank.id_to_drug[drug_id].smiles

        cnn_features = {}
        lensize = self.atom_info + self.struct_info
        
        for drug_id, smile in tqdm({**drug_to_smiles, **test_drug_to_smiles}.items(), desc='cnn features'):
            mol = Chem.MolFromSmiles(smile) # pylint: disable=maybe-no-member
            cnn_features[drug_id] = np.array(self.mol_to_feature(mol, -1)).reshape(self.atom_size, lensize, 1) # pylint: disable=too-many-function-args

        return cnn_features