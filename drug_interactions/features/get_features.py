from drug_interactions.datasets.dataset_builder import DatasetTypes
from drug_interactions.features.Char2VecFeature import Char2VecFeature
from drug_interactions.features.EmbeddingFeature import EmbeddingFeature
from drug_interactions.features.CNNFeature import CNNFeature
from drug_interactions.features.OneHotFeature import OneHotFeature

def get_features(dataset_type, **kwargs):
    features = []
    if dataset_type == DatasetTypes.AFMP:
        features = [EmbeddingFeature(**kwargs)]

    elif dataset_type == DatasetTypes.ONEHOT_SMILES:
        features = [OneHotFeature(**kwargs)]

    elif dataset_type == DatasetTypes.CHAR_2_VEC:
        features = [Char2VecFeature(**kwargs)]

    elif dataset_type == DatasetTypes.DEEP_SMILES:
        features = [OneHotFeature(**kwargs), CNNFeature(**kwargs)]

    return features