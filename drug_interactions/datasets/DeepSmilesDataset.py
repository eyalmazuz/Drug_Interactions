from itertools import product
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import re
from typing import List, Dict, Tuple, Optional, Any
import random

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.info')
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm

from drug_interactions.reader.dal import DrugBank
from drug_interactions.datasets.DatasetUtils import Data
from drug_interactions.datasets.AbstractDataset import DrugDataset


class DeepSmilesDrugDataset(DrugDataset):

    def __init__(self, old_drug_bank: DrugBank, new_drug_bank: DrugBank, neg_pos_ratio: float=1.0, **kwargs):
        super().__init__(old_drug_bank, new_drug_bank, neg_pos_ratio)
        self.atom_size = kwargs['atom_size']
        self.atom_info = kwargs['atom_info']
        self.struct_info = kwargs['struct_info']
        self.validation_size = kwargs['validation_size']

    def split_positive_negative(self, data):

        data, labels = data
        print('getiing positive samples')
        idxs = np.where(np.array(labels) == 1)[0]
        pos_data = [data[i] for i in tqdm(idxs, 'positive')]
        pos_labels = [1] * len(pos_data)

        print('getting negative samples')
        idxs = np.where(np.array(labels) == 0)[0]
        neg_data = [data[i] for i in tqdm(idxs, 'negative')]
        neg_labels = [0] * len(neg_data)

        return (pos_data, pos_labels), (neg_data, neg_labels)

    def sample_data(self, negative_instances: List[Tuple[int, int]], len_positive: int) -> Tuple[List[Tuple[int, int]], List[int]]:
        
        print(f'Number of negative samples: {len(negative_instances)=}')
        print(f'Number of positive samples: {len_positive=}')
        if len_positive < len(negative_instances) and self.neg_pos_ratio is not None:
            print('There are less positive cells so sampling from the negative cells')
            negative_indexes = random.sample(range(len(negative_instances)), k=int(self.neg_pos_ratio * len_positive))

            negative_instances = [negative_instances[i] for i in negative_indexes]
            print('done sampling')
        negative_labels = [0] * len(negative_instances)
        
        return negative_instances, negative_labels

    def create_data(self):
        
        train_drug_ids, new_drug_ids = self.get_train_test_ids()

        drug_to_smiles = {}
        for drug_id in train_drug_ids:
            drug_to_smiles[drug_id] = self.old_drug_bank.id_to_drug[drug_id].smiles

        test_drug_to_smiles = {}
        for drug_id in new_drug_ids:
            test_drug_to_smiles[drug_id] = self.new_drug_bank.id_to_drug[drug_id].smiles

        drug_to_smiles_features = self.get_smiles_features(drug_to_smiles, test_drug_to_smiles)
        cnn_features = self.get_cnn_smiles_features(drug_to_smiles, test_drug_to_smiles)

        self.drug_to_smiles_features = drug_to_smiles_features
        self.cnn_features = cnn_features

    def build_dataset(self):
        
        metadata = {}
        metadata['atom_size'] = self.atom_size
        metadata['atom_info'] = self.atom_info
        metadata['struct_info'] = self.struct_info

        self.old_drug_bank = self.get_smiles_drugs(self.old_drug_bank)
        self.new_drug_bank = self.get_smiles_drugs(self.new_drug_bank)

        train_data, test_data = self.get_train_test_pairs()
        print(f'{len(test_data[0])=}')
        self.create_data()

        (pos_instances, pos_labels), (neg_instances, neg_labels) = self.split_positive_negative(train_data)

        neg_instances, neg_labels = self.sample_data(neg_instances, len(pos_instances))

        x = pos_instances + neg_instances
        y = pos_labels + neg_labels
        metadata['data_size'] =len(y)
        print(f'All data len: {len(y)=}')

        print('Creating validation set.')

        if self.validation_size is not None:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.validation_size,
                                                            random_state=42, shuffle=True, stratify=y)
        print('Creating test data')
        x_test, y_test = test_data

        print('shuffeling the data')

        train = list(zip(x_train, y_train))
        random.shuffle(train)
        x_train, y_train = zip(*train)

        val = list(zip(x_val, y_val))
        random.shuffle(val)
        x_val, y_val = zip(*val)

        test = list(zip(x_test, y_test))
        random.shuffle(train)
        x_test, y_test = zip(*test)

        print('Generating dataset objects')
 
        train_dataset = tf.data.Dataset.from_generator(self.data_generator,
                                                        args=[x_train, y_train],
                                                        output_types=(((np.float32, np.float32),
                                                                        (np.float32, np.float32)),
                                                                        np.float32))
        print('finished building train dataset')
        if self.validation_size is not None:
            validation_dataset = tf.data.Dataset.from_generator(self.data_generator,
                                                            args=[x_val, y_val],
                                                            output_types=(((np.float32, np.float32),
                                                                            (np.float32, np.float32)),
                                                                            np.float32))
            print('finished building validation dataset')
        else: 
            validation_dataset = None 

        test_dataset = tf.data.Dataset.from_generator(self.test_data_generator,
                                                        args=[x_test, y_test],
                                                        output_types=((tf.string, tf.string),
                                                                    (((np.float32, np.float32), 
                                                                    (np.float32, np.float32)), np.float32)))
        print('finished building test dataset')

        return train_dataset, validation_dataset, test_dataset, metadata

    def data_generator(self, x, y):
        for (a, b), label in zip(x, y):
            f_a, f_b = self.drug_to_smiles_features[a.decode()], self.drug_to_smiles_features[b.decode()]
            c_a, c_b = self.cnn_features[a.decode()], self.cnn_features[b.decode()]
            yield ((c_a, f_a), (c_b, f_b)), label

    def test_data_generator(self, x, y):
        for (a, b), label in zip(x, y):
            f_a, f_b = self.drug_to_smiles_features[a.decode()], self.drug_to_smiles_features[b.decode()]
            c_a, c_b = self.cnn_features[a.decode()], self.cnn_features[b.decode()]
            yield ((a.decode(), b.decode()), (((c_a, f_a), (c_b, f_b)), label))

    def get_smiles_drugs(self, drug_bank: DrugBank):
        """
        Removes all the drugs that don't have smiles representation from the data.
        as well as the interactions of drugs without smiles.

        Args:
            drug_bank: Drug bank object containing drug data.

        Returns:
            A new drug bank which has only drugs with smiles and interaction between drugs with smiles.
        """
        valid_drug_ids = []
        for drug in drug_bank.drugs:
            try:
                if drug.smiles and len(Chem.MolToSmiles(Chem.MolFromSmiles(drug.smiles), kekuleSmiles=True, isomericSmiles=True)) < self.atom_size: # pylint: disable=maybe-no-member
                    valid_drug_ids.append(drug.id_)
            except:
                pass
        drugs_with_smiles = [drug for drug in drug_bank.drugs if drug.id_ in valid_drug_ids]
        for drug in tqdm(drugs_with_smiles, desc='filtering interactions'):
            new_interactions = [(drug_id, interaction) for drug_id, interaction in drug.interactions if drug_id in valid_drug_ids]
            drug.interactions = set(new_interactions)

        print(f'Num of drugs only with smiles: {len(drugs_with_smiles)=}')
        drugs_with_smiles = [drug for drug in drugs_with_smiles if len(drug.interactions) > 0]
        print(f'Num of drugs only with smiles after filtering 0 interactions: {len(drugs_with_smiles)=}')
        new_bank = DrugBank(drug_bank.version, drugs_with_smiles)
        return new_bank

    def get_smiles_features(self, drug_to_smiles: Dict[str, str], test_drug_to_smiles: Dict[str, str]) -> Dict[str, np.array]:
        charset = sorted(set("".join(list(drug_to_smiles.values()))+"!E?"))
        embed = max([len(smile) for smile in {**drug_to_smiles, **test_drug_to_smiles}.values()]) + 2
        
        drug_to_smiles_features = {}
        
        char_to_int = dict((c, i) for i, c in enumerate(charset))
        for (drug_id, smiles) in tqdm({**drug_to_smiles, **test_drug_to_smiles}.items(), desc='one-hot'):
            one_hot =  np.zeros((embed , len(charset) + 1), dtype=np.float32)
            #encode the startchar
            one_hot[0,char_to_int["!"]] = 1
            #encode the rest of the chars
            for j, c in enumerate(smiles):
                c = c if c in char_to_int else '?'
                try:
                    one_hot[j+1,char_to_int[c]] = 1
                except IndexError:
                    print(f'{j+1=}, {c=}, {char_to_int[c]=}, {smiles=}, {len(smiles)=}, {embed=}, {len(charset)=}')
                    raise IndexError
            #Encode endchar
            one_hot[len(smiles)+1:,char_to_int["E"]] = 1
            drug_to_smiles_features[drug_id] = one_hot

        return drug_to_smiles_features

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


    def get_cnn_smiles_features(self, drug_to_smiles, test_drug_to_smiles):
        cnn_features = {}
        lensize = self.atom_info + self.struct_info
        
        for drug_id, smile in tqdm({**drug_to_smiles, **test_drug_to_smiles}.items(), desc='cnn features'):
            mol = Chem.MolFromSmiles(smile) # pylint: disable=maybe-no-member
            cnn_features[drug_id] = np.array(self.mol_to_feature(mol, -1)).reshape(self.atom_size, lensize, 1) # pylint: disable=too-many-function-args

        return cnn_features