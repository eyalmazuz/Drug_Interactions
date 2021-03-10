import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from typing import Dict, List, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Add, Dropout, Multiply, Conv2D, BatchNormalization, AveragePooling2D, \
                                Concatenate, Flatten, Dense, InputLayer, GRU, Bidirectional, GlobalMaxPool2D

from drug_interactions.datasets.DatasetUtils import DatasetTypes
from drug_interactions.models.AFMP import AFMP
from drug_interactions.models.CharSmiles import CharSmiles
from drug_interactions.models.DeepSmiles import DeepSmiles


def get_model(model_type: DatasetTypes, metadata:Dict[str, Any]):

    if model_type == DatasetTypes.COLD_START:
        return AFMP(metadata)

    if model_type == DatasetTypes.ONEHOT_SMILES:
        return CharSmiles(metadata)

    if model_type == DatasetTypes.DEEP_SMILES:
        return DeepSmiles(metadata)