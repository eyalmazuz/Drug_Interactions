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
from drug_interactions.datasets.dataset_builder import DrugDataset


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