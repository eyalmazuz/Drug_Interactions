from enum import Enum
from typing import List, Dict, Tuple, Optional, Any

import tensorflow as tf

Data = Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, Any]]

class DatasetTypes(Enum):
    COLD_START = 1
    ONEHOT_SMILES = 2
    DEEP_SMILES = 3
    CHAR_2_VEC = 4
    BINARY_SMILES = 5
