from itertools import product
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import re
from typing import List, Dict, Tuple, Optional, Any
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from drug_interactions.reader.dal import DrugBank
from drug_interactions.datasets.DatasetUtils import Data
from drug_interactions.datasets.AbstractDataset import DrugDataset

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