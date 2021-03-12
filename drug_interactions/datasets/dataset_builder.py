from abc import ABC, abstractmethod
from enum import Enum
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
from drug_interactions.datasets.ColdStartDataset import ColdStartDrugDataset
from drug_interactions.datasets.DatasetUtils import DatasetTypes, Data
from drug_interactions.datasets.IntersectionDataset import IntersectionDrugDataset
from drug_interactions.datasets.OneHotSmilesDataset import OneHotSmilesDrugDataset
from drug_interactions.datasets.DeepSmilesDataset import DeepSmilesDrugDataset


def get_dataset(data_type: DatasetTypes, old_drug_bank: DrugBank,
                new_drug_bank: DrugBank, neg_pos_ratio: float, validation_size: float=0.2, **kwargs) -> Optional[Data]:
    if data_type == DatasetTypes.COLD_START:
        data = ColdStartDrugDataset(old_drug_bank, new_drug_bank, neg_pos_ratio, **kwargs).build_dataset(validation_size=validation_size)

    elif data_type == DatasetTypes.ONEHOT_SMILES:
        data = OneHotSmilesDrugDataset(old_drug_bank, new_drug_bank, neg_pos_ratio, **kwargs).build_dataset()

    elif data_type == DatasetTypes.INTERSECTION:
        # data = IntersectionDrugDataset(old_drug_bank, new_drug_bank, neg_pos_ratio, **kwargs).build_dataset()
        data = None

    elif data_type == DatasetTypes.DEEP_SMILES:
        data = DeepSmilesDrugDataset(old_drug_bank, new_drug_bank, neg_pos_ratio, **kwargs).build_dataset()

    elif data_type == DatasetTypes.CharVec:
        data = None
    
    return data
