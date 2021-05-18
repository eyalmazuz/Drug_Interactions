import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from drug_interactions.datasets.dataset_builder import DatasetTypes
from drug_interactions.models.AFMP import AFMP, AFMPConfig
from drug_interactions.models.CharSmiles import CharSmiles, CharSmilesConfig
from drug_interactions.models.DeepSmiles import DeepSmiles, DeepSmilesConfig


def get_config(model_type: DatasetTypes, **kwargs):

    if model_type == DatasetTypes.AFMP:
        return AFMPConfig(**kwargs)

    elif model_type == DatasetTypes.ONEHOT_SMILES or model_type == DatasetTypes.CHAR_2_VEC:
        return CharSmilesConfig(**kwargs)

    elif model_type == DatasetTypes.DEEP_SMILES:
        return DeepSmilesConfig(**kwargs)


def get_model(model_type: DatasetTypes, **kwargs):

    config = get_config(model_type, **kwargs)

    if model_type == DatasetTypes.AFMP:
        return AFMP(config)

    elif model_type == DatasetTypes.ONEHOT_SMILES or model_type == DatasetTypes.CHAR_2_VEC:
        return CharSmiles(config)

    elif model_type == DatasetTypes.DEEP_SMILES:
        return DeepSmiles(config)
