import os
from typing import List, Tuple, Set, Dict, Any
from itertools import product

from drug_interactions.reader.dal import DrugBank
from drug_interactions.reader.preprocessor import DrugPreprocessor

import numpy as np
from tqdm import tqdm



class DrugReader():

    def __init__(self, path: str):

        self.path = path


    def get_drug_data(self, train_version: str, test_version: str) -> Tuple[DrugBank, DrugBank]:
        """
        Returning processed drug bank data for train and test versions.
        """
        train_preprocessor = DrugPreprocessor(f'{self.path}/{train_version}')
        train_data = train_preprocessor.get_preprocessed_drug_bank()

        test_preprocessor = DrugPreprocessor(f'{self.path}/{test_version}')
        test_data = test_preprocessor.get_preprocessed_drug_bank()

        return train_data, test_data
