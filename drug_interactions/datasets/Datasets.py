import json
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

class TrainDataset(Sequence):

    def __init__(self,
        pos: Tuple[List[Tuple[str, str]], List[int]],
        neg: Tuple[List[Tuple[str, str]], List[int]],
        features: Dict[str, Dict[str, np.ndarray]],
        batch_size: int,
        **kwargs):

        self.x_pos, self.y_pos = pos
        self.x_neg, self.y_neg = neg

        self.x = self.x_pos + self.x_neg
        self.y = self.y_pos + self.y_neg
        data = list(zip(self.x, self.y))
        random.shuffle(data)
        self.x, self.y = zip(*data)

        self.features = features
        self.batch_size = batch_size
        self.neg_pos_ratio = kwargs['neg_pos_ratio']

    def __len__(self,):
        return len(self.x) // self.batch_size + 1

    def __getitem__(self, idx):
        batch_drugs = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        drug_a, drug_b = list(zip(*batch_drugs))

        drug_a_features = []
        drug_b_features = []
        for _, feature in sorted(self.features.items()):
            drug_a_features.append(self.get_drug_features(drug_a, feature))
            drug_b_features.append(self.get_drug_features(drug_b, feature))

        if len(drug_a_features) == 1:
            drug_a_features = drug_a_features[0]

        if len(drug_b_features) == 1:
            drug_b_features = drug_b_features[0]

        batch_labels = np.array(batch_labels).reshape(-1, 1)

        return (drug_a_features, drug_b_features), batch_labels

    def get_drug_features(self, drugs, feature):
        drug_features = np.array([feature[drug] for drug in drugs])
        return drug_features
 
    def sample(self,):
        print("In epoch sample")
        negative_indexes = random.sample(range(len(self.x_neg)), k=int(self.neg_pos_ratio * len(self.x_pos)))

        negative_instances = [self.x_neg[i] for i in negative_indexes]
        negative_labels = [0] * len(negative_instances)
 
        self.x = self.x_pos + negative_instances
        self.y = self.y_pos + negative_labels

        data = list(zip(self.x, self.y))
        random.shuffle(data)
        self.x, self.y = zip(*data)
        print(f"len of data: {len(self.x)}")

class TestDataset(Sequence):

    def __init__(self,
        path: str,
        features: Dict[str, Dict[str, np.ndarray]],
        batch_size: int,
        similar_map_path: str,
        drug_a_similar: bool=False,
        drug_b_similar: bool=False,
        ):

        self.test_data = pd.read_csv(path)
        self.features = features
        self.batch_size = batch_size

        self.drug_a_similar = drug_a_similar
        self.drug_b_similar = drug_b_similar

        with open(similar_map_path, 'r') as f:
            self.similar_map = json.load(f)

        self.drug_a_list = self.test_data['Drug1_ID'].tolist()
        self.drug_b_list = self.test_data['Drug2_ID'].tolist()
        self.y_test = self.test_data['label']

    def __len__(self, ):
        return len(self.test_data) // self.batch_size + 1

    def __getitem__(self, idx):
        drug_a_batch = self.drug_a_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        drug_b_batch = self.drug_b_list[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.drug_a_similar:
            drug_a_new_batch = [self.similar_map[drug][0][0] if drug in self.similar_map and self.similar_map[drug] else drug for drug in drug_a_batch]
        else:
            drug_a_new_batch = drug_a_batch

        if self.drug_b_similar:
            drug_b_new_batch = [self.similar_map[drug][0][0] if drug in self.similar_map and self.similar_map[drug] else drug for drug in drug_b_batch]
        else:
            drug_b_new_batch = drug_b_batch
        
        batch_drugs = list(zip(drug_a_new_batch, drug_b_new_batch))
        batch_labels = self.y_test[idx * self.batch_size:(idx + 1) * self.batch_size]
 
        drug_a, drug_b = list(zip(*batch_drugs))

        drug_a_features = []
        drug_b_features = []
        for _, feature in sorted(self.features.items()):
            drug_a_features.append(self.get_drug_features(drug_a, feature))
            drug_b_features.append(self.get_drug_features(drug_b, feature))

        if len(drug_a_features) == 1:
            drug_a_features = drug_a_features[0]

        if len(drug_b_features) == 1:
            drug_b_features = drug_b_features[0]

        batch_labels = np.array(batch_labels).reshape(-1, 1)

        return (drug_a_batch, drug_b_batch), ((drug_a_features, drug_b_features), batch_labels)

    def get_drug_features(self, drugs, feature):
        drug_features = np.array([feature[drug] for drug in drugs])
        return drug_features

class TTATestDataset(Sequence):

    def __init__(self,
        path: str,
        features: Dict[str, Dict[str, np.ndarray]],
        similar_map_path: str,
        ):

        self.test_data = pd.read_csv(path)
        self.features = features

        with open(similar_map_path, 'r') as f:
            self.similar_map = json.load(f)

        self.drug_a_list = self.test_data['Drug1_ID'].tolist()
        self.drug_b_list = self.test_data['Drug2_ID'].tolist()
        self.y_test = self.test_data['label']

    def __len__(self, ):
        return len(self.drug_a_list)

    def __getitem__(self, idx):

        drug_a = self.drug_a_list[idx]
        drug_b = self.drug_b_list[idx]
        drug_a_id = drug_a
        drug_b_id = drug_b
        
        drug_a_similars = self.similar_map[drug_a]
        if drug_a_similars:
            num_tta = min(3, len(drug_a_similars))
            tta_ids, tta_weights = list(zip(*drug_a_similars[:num_tta]))

            drug_a = list(tta_ids)
            drug_b = [drug_b] * num_tta
            batch_labels = [self.y_test[idx]]
        else:
            drug_a = [drug_a]
            drug_b = [drug_b]
            tta_weights = []
            batch_labels = [self.y_test[idx]]

        drug_a_features = []
        drug_b_features = []
        for _, feature in sorted(self.features.items()):
            drug_a_features.append(self.get_drug_features(drug_a, feature))
            drug_b_features.append(self.get_drug_features(drug_b, feature))

        if len(drug_a_features) == 1:
            drug_a_features = drug_a_features[0]

        if len(drug_b_features) == 1:
            drug_b_features = drug_b_features[0]

        batch_labels = np.array(batch_labels).reshape(-1, 1)

        return (drug_a_id, drug_b_id), ((drug_a_features, drug_b_features), batch_labels), tta_weights

    def get_drug_features(self, drugs, feature):
        drug_features = np.array([feature[drug] for drug in drugs])
        return drug_features

class TTANNTestDataset(Sequence):

    def __init__(self,
        path: str,
        features: Dict[str, Dict[str, np.ndarray]],
        similar_map_path: str,
        k=3,
        ):

        self.test_data = pd.read_csv(path)
        self.features = features

        with open(similar_map_path, 'r') as f:
            self.similar_map = json.load(f)

        self.k = k

        self.drug_a_list = self.test_data['Drug1_ID'].tolist()
        self.drug_b_list = self.test_data['Drug2_ID'].tolist()
        self.y_test = self.test_data['label']

    def __len__(self, ):
        return len(self.drug_a_list)

    def __getitem__(self, idx):

        drug_a = self.drug_a_list[idx]
        drug_b = self.drug_b_list[idx]
        drug_a_id = drug_a
        drug_b_id = drug_b
        
        drug_a_similars = self.similar_map[drug_a]
        drug_b_similars = self.similar_map[drug_b]

        if drug_a_similars and drug_b_similars:
            num_tta = min(self.k, len(drug_a_similars), len(drug_b_similars))
            tta_a_ids, tta_a_weights = list(zip(*drug_a_similars[:num_tta]))
            tta_b_ids, tta_b_weights = list(zip(*drug_b_similars[:num_tta]))

            tta_weights = []
            for a, b in zip(tta_a_weights, tta_b_weights):
                h_mean = 2 / (1 / a + 1 / b)
                tta_weights.append(h_mean)

            drug_a = list(tta_a_ids)
            drug_b = list(tta_b_ids)

        elif drug_a_similars:
            num_tta = min(self.k, len(drug_a_similars))
            tta_ids, tta_weights = list(zip(*drug_a_similars[:num_tta]))

            drug_a = list(tta_ids)
            drug_b = [drug_b] * num_tta
            
        elif drug_b_similars:
            num_tta = min(self.k, len(drug_b_similars))
            tta_ids, tta_weights = list(zip(*drug_b_similars[:num_tta]))

            drug_a = [drug_a] * num_tta
            drug_b = list(tta_ids)
 
        else:
            drug_a = [drug_a]
            drug_b = [drug_b]
            tta_weights = []

        batch_labels = [self.y_test[idx]]

        drug_a_features = []
        drug_b_features = []
        for _, feature in sorted(self.features.items()):
            drug_a_features.append(self.get_drug_features(drug_a, feature))
            drug_b_features.append(self.get_drug_features(drug_b, feature))

        if len(drug_a_features) == 1:
            drug_a_features = drug_a_features[0]

        if len(drug_b_features) == 1:
            drug_b_features = drug_b_features[0]

        batch_labels = np.array(batch_labels).reshape(-1, 1)

        return (drug_a_id, drug_b_id), ((drug_a_features, drug_b_features), batch_labels), tta_weights

    def get_drug_features(self, drugs, feature):
        drug_features = np.array([feature[drug] for drug in drugs])
        return drug_features


class NewOldTestDataset(Sequence):

    def __init__(self,
        path: str,
        features: Dict[str, Dict[str, np.ndarray]],
        batch_size: int,):

        self.test_data = pd.read_csv(path)
        self.features = features
        self.batch_size = batch_size

        self.drug_a_list = self.test_data['Drug1_ID'].tolist()
        self.drug_b_list = self.test_data['Drug2_ID'].tolist()
        drug_a_similar_list = self.test_data['Drug1_ID_SIMILAR'].tolist()
        self.x_test = list(zip(drug_a_similar_list, self.drug_b_list))
        self.y_test = self.test_data['label']

    def __len__(self, ):
        return len(self.test_data) // self.batch_size + 1

    def __getitem__(self, idx):
        batch_drugs = self.x_test[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.y_test[idx * self.batch_size:(idx + 1) * self.batch_size]
 
        drug_a_ids = self.drug_a_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        drug_b_ids = self.drug_b_list[idx * self.batch_size:(idx + 1) * self.batch_size]

        drug_a, drug_b = list(zip(*batch_drugs))

        drug_a_features = []
        drug_b_features = []
        for _, feature in sorted(self.features.items()):
            drug_a_features.append(self.get_drug_features(drug_a, feature))
            drug_b_features.append(self.get_drug_features(drug_b, feature))

        if len(drug_a_features) == 1:
            drug_a_features = drug_a_features[0]

        if len(drug_b_features) == 1:
            drug_b_features = drug_b_features[0]

        batch_labels = np.array(batch_labels).reshape(-1, 1)

        return (drug_a_ids, drug_b_ids), ((drug_a_features, drug_b_features), batch_labels)

    def get_drug_features(self, drugs, feature):
        drug_features = np.array([feature[drug] for drug in drugs])
        return drug_features

class NewNewTestDataset(Sequence):

    def __init__(self,
        path: str,
        features: Dict[str, Dict[str, np.ndarray]],
        batch_size: int,):

        self.test_data = pd.read_csv(path)
        self.features = features
        self.batch_size = batch_size

        self.drug_a_list = self.test_data['Drug1_ID'].tolist()
        self.drug_b_list = self.test_data['Drug2_ID'].tolist()
        drug_a_similar_list = self.test_data['Drug1_ID_SIMILAR'].tolist()
        drug_b_similar_list = self.test_data['Drug2_ID_SIMILAR'].tolist()
        self.x_test = list(zip(drug_a_similar_list, drug_b_similar_list))
        self.y_test = self.test_data['label']

    def __len__(self, ):
        return len(self.test_data) // self.batch_size + 1

    def __getitem__(self, idx):
        batch_drugs = self.x_test[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.y_test[idx * self.batch_size:(idx + 1) * self.batch_size]
 
        drug_a_ids = self.drug_a_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        drug_b_ids = self.drug_b_list[idx * self.batch_size:(idx + 1) * self.batch_size]

        drug_a, drug_b = list(zip(*batch_drugs))

        drug_a_features = []
        drug_b_features = []
        for _, feature in sorted(self.features.items()):
            drug_a_features.append(self.get_drug_features(drug_a, feature))
            drug_b_features.append(self.get_drug_features(drug_b, feature))

        if len(drug_a_features) == 1:
            drug_a_features = drug_a_features[0]

        if len(drug_b_features) == 1:
            drug_b_features = drug_b_features[0]

        batch_labels = np.array(batch_labels).reshape(-1, 1)

        return (drug_a_ids, drug_b_ids), ((drug_a_features, drug_b_features), batch_labels)

    def get_drug_features(self, drugs, feature):
        drug_features = np.array([feature[drug] for drug in drugs])
        return drug_features
