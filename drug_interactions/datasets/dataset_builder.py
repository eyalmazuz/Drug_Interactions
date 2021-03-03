from abc import ABC, abstractmethod
from enum import Enum
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
import tensorflow as tf
from tqdm import tqdm

from drug_interactions.reader.dal import DrugBank

Data = Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, Any]]

class DatasetTypes(Enum):
    COLD_START = 1
    SMILES = 2
    INTERSECTION = 3


class DrugDataset(ABC):

    def __init__(self, old_drug_bank: DrugBank, new_drug_bank: DrugBank, neg_pos_ratio):
        self.old_drug_bank = old_drug_bank
        self.new_drug_bank = new_drug_bank
        self.neg_pos_ratio = neg_pos_ratio
        super().__init__()

    @abstractmethod
    def get_positive_instances(self, data: Any, new_drug_idxs: Any) -> Tuple[Any, Any]:
        """
        Create a list of all the positive instances in the train data.
        A positive instances is a pair (i, j) where train_data[i, j] == 1 and is not a new drug.
        Because the data is symmetric we only take the upper triangle.
        
        Args:
            data: The data object.
            new_drug_idxs: An object containing which drugs are new.

        Returns:
            A tuple of negative samples and their labels.
        """
        raise NotImplementedError

    @abstractmethod
    def get_negative_instances(self, data: Any, new_drug_idxs: Any) -> Tuple[Any, Any]:
        """
        Create a list of all the negative instances in the train data.
        A negative instances is a pair (i, j) where train_data[i, j] == 1 and is not a new drug.
        Because the data is symmetric we only take the upper triangle.
        
        Args:
            data: The data object.
            new_drug_idxs: An object containing which drugs are new.

        Returns:
            A tuple of negative samples and their labels.
        """
        raise NotImplementedError

    @abstractmethod
    def smaple_data(self, data: Any, new_drug_idxs: Any, neg_pos_ratio: float) -> Tuple[Any, List[int]]:
        """
        Generate a sample from the data with equal number of positive and negative instances.

        Args:
            data: A binary matrix containing interaction data between drugs.
            new_drug_idxs: A list of pair (i, j) which indicates the cells in the train_data that are new drugs interactions.
            neg_pos_ratio: The ratio sampling between the negative and positive instances.
        
        Returns:
            x_train: A list of pairs of drug interactions from train_data.
            y_train: the value of train_data[i, j] for the i-th and j-th drug.
        """
        raise NotImplementedError

    @abstractmethod
    def create_data(self) -> Tuple[Any, Any, Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def build_dataset(self) -> Data:
        """
        Generate a sample from the data with equal number of positive and negative instances.

        Returns:
            train_dataset: Tensorflow's dataset object containing train data.
            validation_dataset: Tensorflow's dataset object containing train data.
            test_dataset: Tensorflow's dataset object containing test data (new drugs).
            metadata: A python dict containing metadata about the datasets.
        """
        raise NotImplementedError

    @abstractmethod
    def build_test_dataset(self, test_data: Any, new_drug_idxs: Any) -> tf.data.Dataset:
        pass


class ColdStartDrugDataset(DrugDataset):
    
    def __init__(self, old_drug_bank: DrugBank, new_drug_bank: DrugBank, neg_pos_ratio: float=1.0, **kwargs):
        super().__init__(old_drug_bank, new_drug_bank, neg_pos_ratio)

    def get_positive_instances(self, data: np.ndarray, new_drug_idxs: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Create a list of all the positive samples in the train data.
        A positive sample is a pair (i, j) where train_data[i, j] == 1 and is not a new drug.
        Because the matrix is symmetric we only take the upper triangle.

        Args:
            data: A binary matrix containing interaction data between drugs.
            new_drug_idxs: A list of pair (i, j) which indicates the cells in the train_data that are new drugs interactions.

        Returns:
            positive_samples: A list of pairs of positive drug interactions from train_data.
            positive_labels: the value of train_data[i, j] for the i-th and j-th drug.
        """
        print('Sampling positive cells')
        idxs = np.where(data == 1)
        positive_samples = list(zip(idxs[0], idxs[1]))
        print('filtering duplicates')
        positive_samples = [pair for pair in positive_samples if pair[0] > pair[1]]
        print('filtering new drugs')
        positive_samples = list(set(positive_samples) - set(new_drug_idxs))
        positive_labels = [1] * len(positive_samples)

        return positive_samples, positive_labels

    def get_negative_instances(self, data: np.ndarray, new_drug_idxs: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Create a list of all the negative samples in the train data.
        A negative sample is a pair (i, j) where train_data[i, j] == 0 and is not a new drug.
        Because the matrix is symmetric we only take the upper triangle.

        Args:
            data: A binary matrix containing interaction data between drugs.
            new_drug_idxs: A list of pair (i, j) which indicates the cells in the train_data that are new drugs interactions.

        Returns:
            negative_samples: A list of pairs of negative drug interactions from train_data.
            negative_labels: the value of train_data[i, j] for the i-th and j-th drug.
        """
        print('Sampling negative cells')
        idxs = np.where(data == 0)
        negative_samples = list(zip(idxs[0], idxs[1]))
        print('filtering duplicates')
        negative_samples = [pair for pair in negative_samples if pair[0] > pair[1]]
        print('filtering new drugs')
        negative_samples = list(set(negative_samples) - set(new_drug_idxs))
        
        negative_labels = [0] * len(negative_samples)


        return negative_samples, negative_labels

    def smaple_data(self, negative_instances: List[Tuple[int, int]],
                   len_positive: int,  neg_pos_ratio: float=1.0,) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Generate a sample from the data with equal number of positive and negative instances.

        Args:
            negative_instances: List of drug pairs which have no interaction in the data
            len_positive: the number of positive interaction in the dataset.
            neg_pos_ratio: the sample ratio between negative and positive instnaces.

        Returns:
            x_train: A list of pairs of drug interactions from train_data.
            y_train: the value of train_data[i, j] for the i-th and j-th drug.
        """
        print(f'{len(negative_instances)=}')
        print(f'{len_positive=}')
        if len_positive < len(negative_instances) and neg_pos_ratio is not None:
            print('There are less positive cells so sampling from the negative cells')
            negative_indexes = random.sample(range(len(negative_instances)), k=int(neg_pos_ratio * len_positive))

            negative_instances = [negative_instances[i] for i in negative_indexes]
            print('done sampling')
        negative_labels = [0] * len(negative_instances)
        
        return negative_instances, negative_labels

    def create_data(self):
        """
        Creates training data for the model with cold start.
        The cold start mean that we don't remove drugs that are in the test_data and not in the train data.

        Returns:
            A binary matrix for train data that contains 1 in the (i,j) cell iff the i-th drug
            and the j-th drug have an interaction.

            A binary matrix for test data that contains 1 in the (i,j) cell iff the i-th drug
            and the j-th drug have an interaction.

            A list of the indexes i,j in the matrix that are belong to new drugs only.

            A dict mapping from index in the matrix to list of indexes in the matrix,
            a graph of the drug interactions using a adjacency list format using only old drugs. 
        """
        train_drug_ids, test_drug_ids = set(self.old_drug_bank.id_to_drug.keys()), set(self.new_drug_bank.id_to_drug.keys())
        sorted_drug_ids = sorted(list(train_drug_ids | test_drug_ids))
        new_drug_ids = test_drug_ids - (train_drug_ids & test_drug_ids)
        old_drug_ids = train_drug_ids - (train_drug_ids & test_drug_ids)
        print(f'{len(new_drug_ids)=}')
        print(f'{len(old_drug_ids)=}')
        print(f'{len(sorted_drug_ids)=}')

        # Constructing a binary interaction matrix size of the union of both drug bank for the train set.
        # Has 1 in the (i,j) cell if the i-th and j-th drugs has interaction in the old drug bank. 
        train_interaction_matrix = np.zeros(shape=(len(sorted_drug_ids), len(sorted_drug_ids)))
        for drug in tqdm(self.old_drug_bank.drugs, desc='train'):
            drug_idx = sorted_drug_ids.index(drug.id_)
            for drug_id, _ in drug.interactions:
                interaction_idx = sorted_drug_ids.index(drug_id)
                train_interaction_matrix[drug_idx, interaction_idx] = train_interaction_matrix[interaction_idx, drug_idx] = 1

        # Constructing a binary interaction matrix size of the union of both drug bank for the test set.
        # Has 1 in the (i,j) cell if the i-th and j-th drugs has interaction in the new drug bank. 
        test_interaction_matrix = np.zeros(shape=(len(sorted_drug_ids), len(sorted_drug_ids)))
        for drug in tqdm(self.new_drug_bank.drugs, desc='test'):
            drug_idx = sorted_drug_ids.index(drug.id_)
            for drug_id, _ in drug.interactions:
                interaction_idx = sorted_drug_ids.index(drug_id)
                test_interaction_matrix[drug_idx, interaction_idx] = test_interaction_matrix[interaction_idx, drug_idx] = 1

        print(f'{train_interaction_matrix.sum()=}')
        print(f'{test_interaction_matrix.sum()=}')

        new_drug_idx = [sorted_drug_ids.index(id_) for id_ in new_drug_ids]
        old_drug_idx = [sorted_drug_ids.index(id_) for id_ in train_drug_ids]

        # returning all the indexes of interactions between old drugs and new drugs and new drugs with new drugs
        new_drug_inteactions_list = list(product(new_drug_idx, old_drug_idx)) + list(product(new_drug_idx, new_drug_idx))

        # filtering drug interactions that are interacting with themselves
        new_drug_inteactions_list = list(filter(lambda idxs: idxs[0] != idxs[1] and idxs[0] > idxs[1], new_drug_inteactions_list))

        drug_graph: Dict[int, List[int]] = {}
        for drug in self.old_drug_bank.drugs:
            drug_index = sorted_drug_ids.index(drug.id_)
            drug_graph[drug_index] = [sorted_drug_ids.index(drug_id) for drug_id, _ in drug.interactions]

        return train_interaction_matrix, test_interaction_matrix, new_drug_inteactions_list, drug_graph

    def build_dataset(self, validation_size: int):
        """
        Generate a sample from the data with equal number of positive and negative instances.

        Args:
            validation_size: A float for how much to take from the data set to validation.
        
        Returns:
            train_dataset: Tensorflow's dataset object containing train data.
            val_dataset: Tensorflow's dataset object containing validation data.
            test:dataset: Tensorflow's dataset object containing test data of new drug interactions.
            metadata: Dict mapping from string to values that needed for model initialization.
        """
        metadata = {}

        train_data, test_data, new_drug_idxs, drug_graph = self.create_data()
        metadata['drug_graph'] = drug_graph
        metadata['num_drugs'] = train_data.shape[0]

        positive_samples, positive_labels = self.get_positive_instances(train_data, new_drug_idxs)
        negative_samples, negative_labels = self.get_negative_instances(train_data, new_drug_idxs)

        negative_samples, negative_labels = self.smaple_data(negative_samples, len(positive_labels),
                                                            neg_pos_ratio=self.neg_pos_ratio)

        x = positive_samples + negative_samples
        y = positive_labels + negative_labels
        metadata['data_size'] =len(y)
        print(f'{len(y)=}')

        print('Creating validation set.')
        if validation_size is not None:
            validation_indexes = random.sample(range(len(x)), k=int(validation_size * len(x)))
            train_indexes = list(set(range(len(x))) - set(validation_indexes))

            x_val = [x[i] for i in validation_indexes]
            y_val = [y[i] for i in validation_indexes]

            x_train = [x[i] for i in train_indexes]
            y_train = [y[i] for i in train_indexes]

        print('Generating dataset objects')

        x_train = list(map(np.array, zip(*x_train)))
        x_val = list(map(np.array, zip(*x_val)))

        train_input = tf.data.Dataset.from_tensor_slices((x_train[0], x_train[1]))
        train_labels = tf.data.Dataset.from_tensor_slices(y_train)
        train_dataset = tf.data.Dataset.zip((train_input, train_labels))

        validation_input = tf.data.Dataset.from_tensor_slices((x_val[0], x_val[1]))
        validation_labels = tf.data.Dataset.from_tensor_slices(y_val)
        validation_dataset = tf.data.Dataset.zip((validation_input, validation_labels))

        test_dataset = self.build_test_dataset(test_data, new_drug_idxs)

        return train_dataset, validation_dataset, test_dataset, metadata

    def build_test_dataset(self, test_data, new_drug_idxs: List[Tuple[int, int]]) -> tf.data.Dataset:
        """
        Creating test dataset from the new drug indexes and test matrix.
        """
        print('Creating input data.')
        x_test = list(map(np.array, zip(*new_drug_idxs)))
        
        print('Creating labels.')
        y_test = test_data[tuple(np.array(new_drug_idxs).T)]
        print(f'{len(y_test)=}')

        print('Building dataset object.')
        test_input = tf.data.Dataset.from_tensor_slices((x_test[0], x_test[1]))
        test_labels = tf.data.Dataset.from_tensor_slices(y_test)
        test_dataset = tf.data.Dataset.zip((test_input, test_labels))
        return test_dataset

class SmilesDrugDataset(DrugDataset):
    
    def __init__(self, old_drug_bank: DrugBank, new_drug_bank: DrugBank, neg_pos_ratio: float=1.0, **kwargs):
        super().__init__(old_drug_bank, new_drug_bank, neg_pos_ratio)
        self.atom_size = kwargs['atom_size']
        self.atom_info = kwargs['atom_info']
        self.struct_info = kwargs['struct_info']

    def get_positive_instances(self):
        pass

    def get_negative_instances(self):
        pass

    def smaple_data(self):
        pass

    def create_data(self):
        
        train_drug_ids, test_drug_ids = set(self.old_drug_bank.id_to_drug.keys()), set(self.new_drug_bank.id_to_drug.keys())
        sorted_drug_ids = sorted(list(train_drug_ids | test_drug_ids))
        new_drug_ids = test_drug_ids - (train_drug_ids & test_drug_ids)

        test_drug_pairs = list(product(new_drug_ids, train_drug_ids)) + list(product(new_drug_ids, new_drug_ids))
        train_drug_pairs = list(product(train_drug_ids, train_drug_ids))

        drug_to_smiles = {}
        for drug in self.old_drug_bank.drugs:
            drug_to_smiles[drug.id_] = drug.smiles

        for drug in self.new_drug_bank.drugs:
            drug_to_smiles[drug.id_] = drug.smiles

        drug_to_smiles_features = self.get_smiles_features(drug_to_smiles)
        drug_to_cnn_features = self.get_cnn_features(drug_to_smiles)

        print(f'{len(drug_to_cnn_features.keys())=}')
        train_smiles_features_a = []
        train_cnn_features_a = []
        train_smiles_features_b = []
        train_cnn_features_b = []
        train_labels = []
        for drug_a, drug_b in tqdm(train_drug_pairs, desc='building train pairs'):
            train_cnn_features_a.append(drug_to_cnn_features[drug_a])
            train_smiles_features_a.append(drug_to_smiles_features[drug_a])
 
            train_cnn_features_b.append(drug_to_cnn_features[drug_b])
            train_smiles_features_b.append(drug_to_smiles_features[drug_b])

            train_labels += [1] if self.old_drug_bank.id_to_drug[drug_a].interacts_with(self.old_drug_bank.id_to_drug[drug_b]) else [0]

        test_smiles_features_a = []
        test_cnn_features_a = []
        test_smiles_features_b = []
        test_cnn_features_b = []
        test_labels = []
        for drug_a, drug_b in tqdm(test_drug_pairs, desc='building test pairs'):
            test_cnn_features_a.append(drug_to_cnn_features[drug_a])
            test_smiles_features_a.append(drug_to_smiles_features[drug_a])
            test_cnn_features_b.append(drug_to_cnn_features[drug_b])
            test_smiles_features_b.append(drug_to_smiles_features[drug_b])

            try:
                test_labels += [1] if self.new_drug_bank.id_to_drug[drug_a].interacts_with(self.new_drug_bank.id_to_drug[drug_b]) else [0]
            except:
                test_labels += [1] if self.new_drug_bank.id_to_drug[drug_a].interacts_with(self.old_drug_bank.id_to_drug[drug_b]) else [0]
        
        return (test_smiles_features_a, test_cnn_features_a, test_smiles_features_b, test_cnn_features_b, test_labels),\
                (train_smiles_features_a, train_cnn_features_a, train_smiles_features_b, train_cnn_features_b, train_labels), test_drug_pairs, {}

    def build_dataset(self, validation_size: float=0.2):
        
        self.old_drug_bank = self.get_smiles_drugs(self.old_drug_bank)
        self.new_drug_bank = self.get_smiles_drugs(self.new_drug_bank)

        train_data, test_data, new_drug_idxs, metadata = self.create_data()

        # TODO continue here
        return [], [], [], []
    
    def build_test_dataset(self):
        pass


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
            if drug.smiles is not None:
                try:
                    if len(Chem.MolToSmiles(Chem.MolFromSmiles(drug.smiles), kekuleSmiles=True, isomericSmiles=True)) > self.atom_size: # pylint: disable=maybe-no-member
                        valid_drug_ids.append(drug.id_)
                    else:
                        valid_drug_ids.append(drug.id_)
                except:
                    pass
        drugs_with_smiles = [drug for drug in drug_bank.drugs if drug.id_ in valid_drug_ids]
        for drug in tqdm(drugs_with_smiles, desc='filtering'):
            new_interactions = [(drug_id, interaction) for drug_id, interaction in drug.interactions if drug_id in valid_drug_ids]
            drug.interactions = set(new_interactions)
        print(f'{len(drugs_with_smiles)=}')
        drugs_with_smiles = [drug for drug in drugs_with_smiles if len(drug.interactions) > 0]
        print(f'{len(drugs_with_smiles)=}')
        new_bank = DrugBank(drug_bank.version, drugs_with_smiles)
        return new_bank

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


    def calc_structure_feature(self, c, flag, label):
        feature = [0] * self.struct_info

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
                molfeature.extend([0] * self.struct_info)
                j = j +1
                
            else:   
                molfeature.extend([0] * self.atom_info)
                f, flag, label = self.calc_structure_feature(c, flag, label)
                molfeature.extend(f)
                j = j +1

        #0-Padding
        molfeature.extend([0] * (self.atom_size - j) * lensize)
        # print(f'{len(molfeature)=}')
        return molfeature


    def mol_to_feature(self, mol, n):
        try: 
            defaultSMILES = Chem.MolToSmiles(mol, kekuleSmiles=False, isomericSmiles=True, rootedAtAtom=int(n)) # pylint: disable=maybe-no-member
        except:
            print('failed first') 
            defaultSMILES = Chem.MolToSmiles(mol, kekuleSmiles=False, isomericSmiles=True) # pylint: disable=maybe-no-member
        try: 
            isomerSMILES = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True, rootedAtAtom=int(n)) # pylint: disable=maybe-no-member
        except: 
            print('failed second')
            isomerSMILES = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True) # pylint: disable=maybe-no-member
        return self.calc_featurevector(Chem.MolFromSmiles(defaultSMILES), isomerSMILES) # pylint: disable=maybe-no-member


    def get_cnn_features(self, drug_to_smiles: Dict[str, str]) -> Dict[str, np.array]:
        lensize = self.atom_info + self.struct_info
        drug_to_cnn_features = {}
        for drug_id, smiles in tqdm(drug_to_smiles.items(), desc='cnn'):
            try:
                mol = Chem.MolFromSmiles(smiles) # pylint: disable=maybe-no-member
                cnn_feat = self.mol_to_feature(mol, -1)
                cnn_feat = np.array(cnn_feat)
                # print(cnn_feat.shape)
                cnn_feat = cnn_feat[:self.atom_size * lensize].reshape(self.atom_size, lensize, 1)
                drug_to_cnn_features[drug_id] = cnn_feat
            except:
                print(f'{cnn_feat.shape=}')
                print('error')
        # print('------------------------------------------------')
        # print('CNN Features Creation Completed Successfuly')
        # print('CNN Features Shape are: {}.'.format(data_f.shape))
        return drug_to_cnn_features

    def get_smiles_features(self, drug_to_smiles: Dict[str, str]) -> Dict[str, np.array]:
        # print(f'{drug_to_smiles=}')
        charset = sorted(set("".join(list(drug_to_smiles.values()))+"!E"))
        embed = max([len(smile) for smile in drug_to_smiles.values()]) + 2
        
        drug_to_smiles_features = {}
        
        char_to_int = dict((c, i) for i, c in enumerate(charset))
        for (drug_id, smiles) in tqdm(drug_to_smiles.items(), desc='one-hot'):
            one_hot =  np.zeros((embed , len(charset)), dtype=np.int8)
            #encode the startchar
            one_hot[0,char_to_int["!"]] = 1
            #encode the rest of the chars
            for j,c in enumerate(smiles):
                one_hot[j+1,char_to_int[c]] = 1
            #Encode endchar
            one_hot[len(smiles)+1:,char_to_int["E"]] = 1
            drug_to_smiles_features[drug_id] = one_hot

        return drug_to_smiles_features

class IntersectionDrugDataset(DrugDataset):
    
    def __init__(self, old_drug_bank: DrugBank, new_drug_bank: DrugBank, neg_pos_ratio: float=1.0, **kwargs):
        super().__init__(old_drug_bank, new_drug_bank, neg_pos_ratio)

    def get_positive_instances(self):
        pass

    def get_negative_instances(self):
        pass

    def smaple_data(self):
        pass

    def create_data(self):
        pass

    def build_dataset(self):
        return None

    def build_test_dataset(self):
        pass

def get_dataset(data_type: DatasetTypes, old_drug_bank: DrugBank,
                new_drug_bank: DrugBank, neg_pos_ratio: float, validation_size: float=0.2, **kwargs) -> Optional[Data]:
    if data_type == DatasetTypes.COLD_START:
        data = ColdStartDrugDataset(old_drug_bank, new_drug_bank, neg_pos_ratio, **kwargs).build_dataset(validation_size=validation_size)

    elif data_type == DatasetTypes.SMILES:
        data = SmilesDrugDataset(old_drug_bank, new_drug_bank, neg_pos_ratio, **kwargs).build_dataset()

    elif data_type == DatasetTypes.INTERSECTION:
        data = IntersectionDrugDataset(old_drug_bank, new_drug_bank, neg_pos_ratio, **kwargs).build_dataset()
    
    else:
        data = None
    
    return data
