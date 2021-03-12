from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any

import tensorflow as tf
from tqdm import tqdm

from drug_interactions.reader.dal import DrugBank
from drug_interactions.datasets.DatasetUtils import Data

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