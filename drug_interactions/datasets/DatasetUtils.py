from enum import Enum
from typing import List, Dict, Tuple, Optional, Any

import tensorflow as tf

Data = Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, Any]]

class DatasetTypes(Enum):
    COLD_START = 1
    ONEHOT_SMILES = 2
    INTERSECTION = 3
    DEEP_SMILES = 4
    CHAR_2_VEC = 5
