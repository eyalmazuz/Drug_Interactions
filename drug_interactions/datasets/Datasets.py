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
    
    def epoch_sample(self,):
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
        batch_size: int,):

        self.test_data = pd.read_csv(path)
        self.features = features
        self.batch_size = batch_size

        drug_a_list = self.test_data['smile_a'].tolist()
        drug_b_list = self.test_data['smile_b'].tolist()
        self.x_test = list(zip(drug_a_list, drug_b_list))
        self.y_test = self.test_data['label']

    def __len__(self, ):
        return len(self.test_data) // self.batch_size + 1

    def __getitem__(self, idx):
        batch_drugs = self.x_test[idx * self.batch_size:(idx + 1) * self.batch_size]
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

        return (drug_a, drug_b), ((drug_a_features, drug_b_features), batch_labels)

    def get_drug_features(self, drugs, feature):
        drug_features = np.array([feature[drug] for drug in drugs])
        return drug_features
