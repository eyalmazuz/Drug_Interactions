from functools import partial
from itertools import product
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import re
from typing import List, Dict, Tuple, Optional, Any
import random

import numpy as np
import pandas as  pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm

from drug_interactions.reader.dal import DrugBank
from drug_interactions.datasets.DatasetUtils import Data
from drug_interactions.datasets.AbstractDataset import DrugDataset


class OneHotSmilesDrugDataset(DrugDataset):
    
    def __init__(self, old_drug_bank: DrugBank, new_drug_bank: DrugBank, neg_pos_ratio: float=1.0, **kwargs):
        super().__init__(old_drug_bank, new_drug_bank, neg_pos_ratio)
        self.atom_size = kwargs['atom_size']
        # self.atom_info = kwargs['atom_info']
        # self.struct_info = kwargs['struct_info']

    def get_positive_instances(self, data, new_drug_idxs=None):

        data, labels = data
        print('getiing positive samples')
        idxs = np.where(np.array(labels) == 1)[0]
        pos_data = [data[i] for i in tqdm(idxs, 'positive')]
        labels = [1] * len(pos_data)

        return pos_data, labels

    def get_negative_instances(self, data, new_drug_idxs=None):

        data, labels = data
        print('getting negative samples')
        idxs = np.where(np.array(labels) == 0)[0]
        neg_data = [data[i] for i in tqdm(idxs, 'negative')]
        labels = [0] * len(neg_data)

        return neg_data, labels

    def sample_data(self, negative_instances: List[Tuple[int, int]], len_positive: int) -> Tuple[List[Tuple[int, int]], List[int]]:
        
        print(f'{len(negative_instances)=}')
        print(f'{len_positive=}')
        if len_positive < len(negative_instances) and self.neg_pos_ratio is not None:
            print('There are less positive cells so sampling from the negative cells')
            negative_indexes = random.sample(range(len(negative_instances)), k=int(self.neg_pos_ratio * len_positive))

            negative_instances = [negative_instances[i] for i in negative_indexes]
            print('done sampling')
        negative_labels = [0] * len(negative_instances)
        
        return negative_instances, negative_labels

    def create_data(self):
        
        train_drug_ids, test_drug_ids = set(self.old_drug_bank.id_to_drug.keys()), set(self.new_drug_bank.id_to_drug.keys())
        new_drug_ids = test_drug_ids - (train_drug_ids & test_drug_ids)

        print(f'Number of new drug with smiles: {len(new_drug_ids)}')

        train_drug_pairs = list(product(train_drug_ids, train_drug_ids))
        train_drug_pairs = list(set([tuple(sorted(t)) for t in train_drug_pairs if t[0] != t[1]]))

        test_drug_pairs = list(product(new_drug_ids, train_drug_ids))
        test_drug_pairs += list(set([tuple(sorted(t)) for t in list(product(new_drug_ids, new_drug_ids)) if t[0] != t[1]]))

        drug_to_smiles = {}
        for drug_id in train_drug_ids:
            drug_to_smiles[drug_id] = self.old_drug_bank.id_to_drug[drug_id].smiles

        test_drug_to_smiles = {}
        for drug_id in new_drug_ids:
            test_drug_to_smiles[drug_id] = self.new_drug_bank.id_to_drug[drug_id].smiles


        drug_to_smiles_features = self.get_smiles_features(drug_to_smiles, test_drug_to_smiles)

        self.drug_to_smiles_features = drug_to_smiles_features

        # TODO remove this later
        # train_labels = [0] * int(len(train_drug_pairs) /2) + [1] * (len(train_drug_pairs) - int(len(train_drug_pairs) /2))
        train_labels = [1 if self.old_drug_bank.id_to_drug[drug_a].interacts_with(self.old_drug_bank.id_to_drug[drug_b]) else 0 for drug_a, drug_b in tqdm(train_drug_pairs, desc='building train pairs')]

        test_labels = []
        for drug_a, drug_b in tqdm(test_drug_pairs, desc='building test pairs'):
            try:
                test_labels += [1] if self.new_drug_bank.id_to_drug[drug_a].interacts_with(self.new_drug_bank.id_to_drug[drug_b]) else [0]
            except:
                test_labels += [1] if self.new_drug_bank.id_to_drug[drug_a].interacts_with(self.old_drug_bank.id_to_drug[drug_b]) else [0]
        
        print(f'Number of test positive samples: {len(list(filter(lambda x: x == 1, test_labels)))}')
        print(f'Number of test negative samples: {len(list(filter(lambda x: x == 0, test_labels)))}')
        # all_smiles = {**drug_to_smiles, **test_drug_to_smiles}

        # train_data = [(all_smiles[s_a], all_smiles[s_b], l) for (s_a, s_b), l in zip(train_drug_pairs, train_labels)]
        # test_data = [(all_smiles[s_a], all_smiles[s_b], l) for (s_a, s_b), l in zip(test_drug_pairs, test_labels)]

        # train_df = pd.DataFrame(train_data, columns=['smile_a', 'smile_b', 'label'])
        # test_df = pd.DataFrame(test_data, columns=['smile_a', 'smile_b', 'label'])

        # train_df.to_csv('./data/csvs/smiles_train.csv', index=False)
        # test_df.to_csv('./data/csvs/smiles_test.csv', index=False)

        # print('finished saving dataframes')

        return (train_drug_pairs, train_labels), (test_drug_pairs, test_labels), {}

    def build_dataset(self, validation_size: float=0.2):
        
        self.old_drug_bank = self.get_smiles_drugs(self.old_drug_bank)
        self.new_drug_bank = self.get_smiles_drugs(self.new_drug_bank)

        train_data, test_data, metadata = self.create_data()

        positive_instances, positive_labels = self.get_positive_instances(train_data)
        negative_instances, negative_labels = self.get_negative_instances(train_data)

        negative_instances, negative_labels = self.sample_data(negative_instances, len(positive_instances))

        x = positive_instances + negative_instances
        y = positive_labels + negative_labels
        metadata['data_size'] =len(y)
        print(f'All data len: {len(y)=}')

        print('Creating validation set.')

        if validation_size is not None:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_size,
                                                            random_state=42, shuffle=True, stratify=y)

        # if validation_size is not None:
        #     validation_indexes = random.sample(range(len(x)), k=int(validation_size * len(x)))
        #     train_indexes = list(set(range(len(x))) - set(validation_indexes))

        #     x_val = [x[i] for i in validation_indexes]
        #     y_val = [y[i] for i in validation_indexes]

        #     x_train = [x[i] for i in train_indexes]
        #     y_train = [y[i] for i in train_indexes]

        print('Creating test data')
        x_test, y_test = test_data

        print('shuffeling the data')

        train = list(zip(x_train, y_train))
        random.shuffle(train)
        x_train, y_train = zip(*train)

        val = list(zip(x_val, y_val))
        random.shuffle(val)
        x_val, y_val = zip(*val)

        test = list(zip(x_test, y_test))
        random.shuffle(train)
        x_test, y_test = zip(*test)

        print('Generating dataset objects')
 
        train_dataset = tf.data.Dataset.from_generator(self.data_generator,
                                                        args=[x_train, y_train],
                                                        output_types=((np.float32, np.float32), np.float32))
        print('finished building train dataset')
        if validation_size is not None:
            validation_dataset = tf.data.Dataset.from_generator(self.data_generator,
                                                            args=[x_val, y_val],
                                                            output_types=((np.float32, np.float32), np.float32))
            print('finished building validation dataset')
        else: 
            validation_dataset = None 

        test_dataset = tf.data.Dataset.from_generator(self.test_data_generator,
                                                        args=[x_test, y_test],
                                                        output_types=(tf.string, ((np.float32, np.float32), np.float32)))
        print('finished building test dataset')

        return train_dataset, validation_dataset, test_dataset, metadata
    
    def data_generator(self, x, y):
        for (a, b), label in zip(x, y):
            f_a, f_b = self.drug_to_smiles_features[a.decode()], self.drug_to_smiles_features[b.decode()] 
            yield (f_a, f_b), label

    def test_data_generator(self, x, y):
        for (a, b), label in zip(x, y):
            f_a, f_b = self.drug_to_smiles_features[a.decode()], self.drug_to_smiles_features[b.decode()] 
            yield (a.decode(), ((f_a, f_b), label))

    def get_smiles_drugs(self, drug_bank: DrugBank):
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
            if drug.smiles is not None and len(drug.smiles) <= self.atom_size:
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

    def get_smiles_features(self, drug_to_smiles: Dict[str, str], test_drug_to_smiles: Dict[str, str]) -> Dict[str, np.array]:
        # print(f'{drug_to_smiles=}')
        charset = sorted(set("".join(list(drug_to_smiles.values()))+"!E?"))
        embed = max([len(smile) for smile in {**drug_to_smiles, **test_drug_to_smiles}.values()]) + 2
        
        drug_to_smiles_features = {}
        
        char_to_int = dict((c, i) for i, c in enumerate(charset))
        for (drug_id, smiles) in tqdm({**drug_to_smiles, **test_drug_to_smiles}.items(), desc='one-hot'):
            one_hot =  np.zeros((embed , len(charset) + 1), dtype=np.float32)
            #encode the startchar
            one_hot[0,char_to_int["!"]] = 1
            #encode the rest of the chars
            for j, c in enumerate(smiles):
                c = c if c in char_to_int else '?'
                try:
                    one_hot[j+1,char_to_int[c]] = 1
                except IndexError:
                    print(f'{j+1=}, {c=}, {char_to_int[c]=}, {smiles=}, {len(smiles)=}, {embed=}, {len(charset)=}')
                    raise IndexError
            #Encode endchar
            one_hot[len(smiles)+1:,char_to_int["E"]] = 1
            drug_to_smiles_features[drug_id] = one_hot

        return drug_to_smiles_features