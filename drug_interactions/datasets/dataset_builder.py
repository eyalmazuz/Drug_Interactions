from enum import Enum
from itertools import product
from typing import List, Any
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from drug_interactions.datasets.Datasets import TrainDataset, TestDataset, NewOldTestDataset, NewNewTestDataset, TTATestDataset, TTANNTestDataset
from drug_interactions.reader.dal import DrugBank

class DatasetTypes(Enum):
    AFMP = 1
    ONEHOT_SMILES = 2
    DEEP_SMILES = 3
    CHAR_2_VEC = 4
    BINARY_SMILES = 5

def get_dataset(old_drug_bank: DrugBank,
                new_drug_bank: DrugBank,
                feature_list: List[Any],
                **kwargs):

    metadata = {}
    old_drug_bank = get_smiles_drugs(old_drug_bank, kwargs["atom_size"])
    new_drug_bank = get_smiles_drugs(new_drug_bank, kwargs["atom_size"])

    metadata['old_drug_bank'] = old_drug_bank
    metadata['new_drug_bank'] = new_drug_bank

    features = {}
    for feature in feature_list:
        features[str(feature)] = feature(old_drug_bank, new_drug_bank)

    train_data = get_train_test_pairs(old_drug_bank, new_drug_bank, kwargs['data_path'])

    (pos_instances, pos_labels), (neg_instances, neg_labels) = split_positive_negative(train_data)

    print('Creating validation set.')

    if kwargs["validation_size"] is not None:
        x_train_pos, x_val_pos, y_train_pos, y_val_pos = train_test_split(pos_instances, pos_labels, test_size=kwargs["validation_size"],
        random_state=42, shuffle=True)

        x_train_neg, x_val_neg, y_train_neg, y_val_neg = train_test_split(neg_instances, neg_labels, test_size=kwargs["validation_size"],
        random_state=42, shuffle=True)

    train_dataset = TrainDataset(pos=(x_train_pos, y_train_pos),
                                 neg=(x_train_neg, y_train_neg),
                                 features=features,
                                 batch_size=kwargs['batch_size'],
                                 neg_pos_ratio=kwargs["neg_pos_ratio"])

    validation_dataset = TrainDataset(pos=(x_val_pos, y_val_pos),
                                      neg=(x_val_neg, y_val_neg),
                                      features=features,
                                      batch_size=kwargs['batch_size'],
                                      neg_pos_ratio=kwargs["neg_pos_ratio"])

    test_new_old_dataset = TTATestDataset(path=f'{kwargs["data_path"]}/test_new_old_similar.csv',
                               features=features,
                               similar_map_path='./data/jsons/similar_drugs_dict_only_old.json',
                               )

    test_new_new_dataset = TTANNTestDataset(path=f'{kwargs["data_path"]}/test_new_new_similar.csv',
                               features=features,
                               similar_map_path='./data/jsons/similar_drugs_dict_only_old.json',
                               )

    # test_new_old_dataset = NewOldTestDataset(path=f'{kwargs["data_path"]}/test_new_old_nlcs_similar.csv',
    #                            features=features,
    #                            batch_size=kwargs["batch_size"],)

    # test_new_new_dataset = NewNewTestDataset(path=f'{kwargs["data_path"]}/test_new_new_nlcs_similar.csv',
    #                            features=features,
    #                            batch_size=kwargs["batch_size"])

    # test_all_dataset = TestDataset(path=f'{kwargs["data_path"]}/test_all_similar.csv',
    #                            features=features,
    #                            batch_size=kwargs["batch_size"])


    return (train_dataset,
            validation_dataset,
            test_new_old_dataset,
            test_new_new_dataset,
            # test_all_dataset,
            metadata)


def get_train_test_pairs(old_drug_bank, new_drug_bank, save_path):
    
    train_drug_ids, new_drug_ids = get_train_test_ids(old_drug_bank, new_drug_bank)

    train_drug_pairs = list(product(train_drug_ids, train_drug_ids))
    train_drug_pairs = list(set([tuple(sorted(t)) for t in train_drug_pairs if t[0] != t[1]]))

    train_labels = [1 if old_drug_bank.id_to_drug[drug_a].interacts_with(old_drug_bank.id_to_drug[drug_b]) else 0 for drug_a, drug_b in tqdm(train_drug_pairs, desc='building train pairs')]

    if not os.path.exists(f'{save_path}/test_new_old.csv'):
        new_old_pairs = list(product(new_drug_ids, train_drug_ids))

        new_old_labels = []
        for drug_a, drug_b in tqdm(new_old_pairs, desc='building new-old pairs'):
            new_old_labels += [1] if new_drug_bank.id_to_drug[drug_a].interacts_with(old_drug_bank.id_to_drug[drug_b]) else [0]


        test_no_vals = []
        for (drug_a, drug_b), label in zip(new_old_pairs, new_old_labels):
            smile_a = new_drug_bank.id_to_drug[drug_a].smiles
            smile_b = old_drug_bank.id_to_drug[drug_b].smiles
            test_no_vals.append((drug_a, smile_a, drug_b, smile_b, label))

            test_no_df = pd.DataFrame(test_no_vals, columns=['Drug1_ID', 'Drug1_SMILES',
                                                        'Drug2_ID', 'Drug2_SMILES',
                                                        'label'])

        test_no_df.to_csv(f'{save_path}/test_new_old.csv', index=False)
        print(f'Number of new old positive samples: {len(list(filter(lambda x: x == 1, new_old_labels)))}')
        print(f'Number of new old negative samples: {len(list(filter(lambda x: x == 0, new_old_labels)))}')



    if not os.path.exists(f'{save_path}/test_new_new.csv'):
        new_new_pairs = list(set([tuple(sorted(t)) for t in list(product(new_drug_ids, new_drug_ids)) if t[0] != t[1]]))

        new_new_labels = []
        for drug_a, drug_b in tqdm(new_new_pairs, desc='building new-old pairs'):
            new_new_labels += [1] if new_drug_bank.id_to_drug[drug_a].interacts_with(new_drug_bank.id_to_drug[drug_b]) else [0]

        test_nn_vals = [] 
        for (drug_a, drug_b), label in zip(new_new_pairs, new_new_labels):
            smile_a = new_drug_bank.id_to_drug[drug_a].smiles
            smile_b = new_drug_bank.id_to_drug[drug_b].smiles
            test_nn_vals.append((drug_a, smile_a, drug_b, smile_b, label))

        test_nn_df = pd.DataFrame(test_nn_vals, columns=['Drug1_ID', 'Drug1_SMILES',
                                                        'Drug2_ID', 'Drug2_SMILES',
                                                        'label'])

        test_nn_df.to_csv(f'{save_path}/test_new_new.csv', index=False)

        print(f'Number of new new positive samples: {len(list(filter(lambda x: x == 1, new_new_labels)))}')
        print(f'Number of new new negative samples: {len(list(filter(lambda x: x == 0, new_new_labels)))}')

    return (train_drug_pairs, train_labels)

def get_train_test_ids(old_drug_bank, new_drug_bank):
    train_drug_ids = set(old_drug_bank.id_to_drug.keys())
    test_drug_ids = set(new_drug_bank.id_to_drug.keys())
    new_drug_ids = test_drug_ids - (train_drug_ids & test_drug_ids)
    print(f'Num of drug in train: {len(train_drug_ids)}')
    print(f'Num of new drug in test: {len(new_drug_ids)}')

    return train_drug_ids, new_drug_ids

def split_positive_negative(data):

    data, labels = data
    print('getiing positive samples')
    idxs = np.where(np.array(labels) == 1)[0]
    pos_data = [data[i] for i in tqdm(idxs, 'positive')]
    pos_labels = [1] * len(pos_data)

    print('getting negative samples')
    idxs = np.where(np.array(labels) == 0)[0]
    neg_data = [data[i] for i in tqdm(idxs, 'negative')]
    neg_labels = [0] * len(neg_data)

    return (pos_data, pos_labels), (neg_data, neg_labels)

def get_smiles_drugs(drug_bank: DrugBank,
                    atom_size: int):
    """
    Removes all the drugs that don't have smiles representation from the data.
    as well as the interactions of drugs without smiles.

    Args:
        drug_bank: Drug bank object containing drug data.

    Returns:
        A new drug bank which has only drugs with smiles and interaction between drugs with smiles.
    """
    valid_drug_ids = []
    for drug in drug_bank.drugs:
        if drug.smiles is not None and len(drug.smiles) <= atom_size:
            valid_drug_ids.append(drug.id_)

    drugs_with_smiles = [drug for drug in drug_bank.drugs if drug.id_ in valid_drug_ids]
    for drug in tqdm(drugs_with_smiles, desc='filtering interactions'):
        new_interactions = [(drug_id, interaction) for drug_id, interaction in drug.interactions if drug_id in valid_drug_ids]
        drug.interactions = set(new_interactions)

    print(f'{len(drugs_with_smiles)=}')
    drugs_with_smiles = [drug for drug in drugs_with_smiles if len(drug.interactions) > 0]
    print(f'{len(drugs_with_smiles)=}')
    new_bank = DrugBank(drug_bank.version, drugs_with_smiles)
    return new_bank
