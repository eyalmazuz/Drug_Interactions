from abc import ABC, abstractmethod
from enum import Enum
from itertools import product
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import re
from typing import List, Dict, Tuple, Optional, Any
import random

import numpy as np
# from rdkit import Chem
# from rdkit.Chem import rdchem
# from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.info')
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
    def sample_data(self, data: Any, new_drug_idxs: Any, neg_pos_ratio: float) -> Tuple[Any, List[int]]:
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

    def sample_data(self, negative_instances: List[Tuple[int, int]], len_positive: int) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Generate a sample from the data with equal number of positive and negative instances.

        Args:
            negative_instances: List of drug pairs which have no interaction in the data
            len_positive: the number of positive interaction in the dataset.

        Returns:
            x_train: A list of pairs of drug interactions from train_data.
            y_train: the value of train_data[i, j] for the i-th and j-th drug.
        """
        print(f'{len(negative_instances)=}')
        print(f'{len_positive=}')
        if len_positive < len(negative_instances) and self.neg_pos_ratio is not None:
            print('There are less positive cells so sampling from the negative cells')
            negative_indexes = random.sample(range(len(negative_instances)), k=int(self.neg_pos_ratio * len_positive))

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

        negative_samples, negative_labels = self.sample_data(negative_samples, len(positive_labels))

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
        # self.atom_info = kwargs['atom_info']
        # self.struct_info = kwargs['struct_info']

    def get_positive_instances(self, data, new_drug_idxs=None):

        data, labels = data
        print('getiing positive samples')
        idxs = np.where(np.array(labels) == 1)[0]
        pos_data = [data[i] for i in tqdm(idxs, 'positive')]
        labels = [1] * len(pos_data)

        return pos_data, labels

    def get_negative_instances(self, data, new_drug_idxs=None):

        data, labels = data
        print('getting negative samples')
        idxs = np.where(np.array(labels) == 0)[0]
        neg_data = [data[i] for i in tqdm(idxs, 'negative')]
        labels = [0] * len(neg_data)

        return neg_data, labels

    def sample_data(self, negative_instances: List[Tuple[int, int]], len_positive: int) -> Tuple[List[Tuple[int, int]], List[int]]:
        
        print(f'{len(negative_instances)=}')
        print(f'{len_positive=}')
        if len_positive < len(negative_instances) and self.neg_pos_ratio is not None:
            print('There are less positive cells so sampling from the negative cells')
            negative_indexes = random.sample(range(len(negative_instances)), k=int(self.neg_pos_ratio * len_positive))

            negative_instances = [negative_instances[i] for i in negative_indexes]
            print('done sampling')
        negative_labels = [0] * len(negative_instances)
        
        return negative_instances, negative_labels

    def create_data(self):
        
        train_drug_ids, test_drug_ids = set(self.old_drug_bank.id_to_drug.keys()), set(self.new_drug_bank.id_to_drug.keys())
        new_drug_ids = test_drug_ids - (train_drug_ids & test_drug_ids)


        train_drug_pairs = list(product(train_drug_ids, train_drug_ids))
        train_drug_pairs = list(set([tuple(sorted(t)) for t in train_drug_pairs if t[0] != t[1]]))

        test_drug_pairs = list(product(new_drug_ids, train_drug_ids))
        test_drug_pairs += list(set([tuple(sorted(t)) for t in list(product(new_drug_ids, new_drug_ids)) if t[0] != t[1]]))

        drug_to_smiles = {}
        for drug in self.old_drug_bank.drugs:
            drug_to_smiles[drug.id_] = drug.smiles

        for drug in self.new_drug_bank.drugs:
            drug_to_smiles[drug.id_] = drug.smiles

        drug_to_smiles_features = self.get_smiles_features(drug_to_smiles)

        train_smiles_features = []
        train_labels = []
        for drug_a, drug_b in tqdm(train_drug_pairs, desc='building train pairs'):
            train_smiles_features.append((drug_to_smiles_features[drug_a], drug_to_smiles_features[drug_b]))
            train_labels += [1] if self.old_drug_bank.id_to_drug[drug_a].interacts_with(self.old_drug_bank.id_to_drug[drug_b]) else [0]

        test_smiles_features = []
        test_labels = []
        for drug_a, drug_b in tqdm(test_drug_pairs, desc='building test pairs'):
            test_smiles_features.append((drug_to_smiles_features[drug_a], drug_to_smiles_features[drug_b]))
            try:
                test_labels += [1] if self.new_drug_bank.id_to_drug[drug_a].interacts_with(self.new_drug_bank.id_to_drug[drug_b]) else [0]
            except:
                test_labels += [1] if self.new_drug_bank.id_to_drug[drug_a].interacts_with(self.old_drug_bank.id_to_drug[drug_b]) else [0]
        
        return (train_smiles_features, train_labels), (test_smiles_features, test_labels), test_drug_pairs, {}

    def build_dataset(self, validation_size: float=0.2):
        
        self.old_drug_bank = self.get_smiles_drugs(self.old_drug_bank)
        self.new_drug_bank = self.get_smiles_drugs(self.new_drug_bank)

        train_data, test_data, _, metadata = self.create_data()

        positive_instances, positive_labels = self.get_positive_instances(train_data)
        negative_instances, negative_labels = self.get_negative_instances(train_data)

        negative_instances, negative_labels = self.sample_data(negative_instances, len(positive_instances))

        x = positive_instances + negative_instances
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

        # train_input = tf.data.Dataset.from_generator(self.data_generator, args=[x_train], output_types=(tf.int64, tf.int64))
        train_input = tf.data.Dataset.from_tensor_slices((x_train[0], x_train[1]))
        train_labels = tf.data.Dataset.from_tensor_slices(y_train)
        train_dataset = tf.data.Dataset.zip((train_input, train_labels))

        # validation_input = tf.data.Dataset.from_generator(self.data_generator, args=[x_val], output_types=(tf.int64, tf.int64))
        validation_input = tf.data.Dataset.from_tensor_slices((x_val[0], x_val[1]))
        validation_labels = tf.data.Dataset.from_tensor_slices(y_val)
        validation_dataset = tf.data.Dataset.zip((validation_input, validation_labels))

        test_dataset = self.build_test_dataset(test_data)

        return train_dataset, validation_dataset, test_dataset, metadata
    
    def build_test_dataset(self, test_data, new_drug_idxs: List[Tuple[int, int]]=None) -> tf.data.Dataset:
        """
        Creating test dataset from the new drug indexes and test matrix.
        """
        print('Creating input data.')

        test_instances, test_labels = test_data

        x_test = list(map(np.array, zip(*test_instances)))
        
        print('Creating labels.')
        y_test = test_labels
        print(f'{len(y_test)=}')

        print('Building dataset object.')
        test_input = tf.data.Dataset.from_tensor_slices((x_test[0], x_test[1]))
        # test_input = tf.data.Dataset.from_generator(self.data_generator, args=[x_test], output_types=(tf.int64, tf.int64))
        test_labels = tf.data.Dataset.from_tensor_slices(y_test)
        test_dataset = tf.data.Dataset.zip((test_input, test_labels))
        return test_dataset

    def data_generator(self, data):

        for drug_a, drug_b in data:
            yield drug_a, drug_b

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
            if drug.smiles is not None and len(drug.smiles) < self.atom_size:
                valid_drug_ids.append(drug.id_)

        drugs_with_smiles = [drug for drug in drug_bank.drugs if drug.id_ in valid_drug_ids]
        for drug in tqdm(drugs_with_smiles, desc='filtering interactions'):
            new_interactions = [(drug_id, interaction) for drug_id, interaction in drug.interactions if drug_id in valid_drug_ids]
            drug.interactions = set(new_interactions)

        print(f'{len(drugs_with_smiles)=}')
        drugs_with_smiles = [drug for drug in drugs_with_smiles if len(drug.interactions) > 0]
        print(f'{len(drugs_with_smiles)=}')
        new_bank = DrugBank(drug_bank.version, drugs_with_smiles)
        return new_bank

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

    def sample_data(self):
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
