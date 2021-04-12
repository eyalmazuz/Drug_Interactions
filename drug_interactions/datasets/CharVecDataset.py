from functools import partial
from itertools import product
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import re
from typing import List, Dict, Tuple, Optional, Any
import random

from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from drug_interactions.reader.dal import DrugBank
from drug_interactions.datasets.DatasetUtils import Data
from drug_interactions.datasets.AbstractDataset import DrugDataset


class CharVecDrugDataset(DrugDataset):
    
    def __init__(self, old_drug_bank: DrugBank, new_drug_bank: DrugBank, neg_pos_ratio: float=1.0, **kwargs):
        super().__init__(old_drug_bank, new_drug_bank, neg_pos_ratio)
        self.atom_size = kwargs['atom_size']
        self.vector_size = kwargs['vector_size']

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

        train_labels = [1 if self.old_drug_bank.id_to_drug[drug_a].interacts_with(self.old_drug_bank.id_to_drug[drug_b]) else 0 for drug_a, drug_b in tqdm(train_drug_pairs, desc='building train pairs')]

        self.drug_to_smiles_features = drug_to_smiles_features


        test_labels = []
        for drug_a, drug_b in tqdm(test_drug_pairs, desc='building test pairs'):
            try:
                test_labels += [1] if self.new_drug_bank.id_to_drug[drug_a].interacts_with(self.new_drug_bank.id_to_drug[drug_b]) else [0]
            except:
                test_labels += [1] if self.new_drug_bank.id_to_drug[drug_a].interacts_with(self.old_drug_bank.id_to_drug[drug_b]) else [0]
        
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
        print(f'{len(y)=}')

        print('Creating validation set.')
        if validation_size is not None:
            validation_indexes = random.sample(range(len(x)), k=int(validation_size * len(x)))
            train_indexes = list(set(range(len(x))) - set(validation_indexes))

            x_val = [x[i] for i in validation_indexes]
            y_val = [y[i] for i in validation_indexes]

            x_train = [x[i] for i in train_indexes]
            y_train = [y[i] for i in train_indexes]

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

        validation_dataset = tf.data.Dataset.from_generator(self.data_generator,
                                                        args=[x_val, y_val],
                                                        output_types=((np.float32, np.float32), np.float32))
        print('finished building validation dataset')

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
        
        texts = [['!'] + list(smiles) + ['E'] for smiles in drug_to_smiles.values()] + ['?']
        
        print('training w2v model')
        model = Word2Vec(sentences=texts, size=self.vector_size, window=5, min_count=1, workers=8)
        model.train(total_examples=model.corpus_count, sentences=texts, epochs=5)
        embed = max([len(smile) for smile in {**drug_to_smiles, **test_drug_to_smiles}.values()]) + 2
        print('done training')
        drug_to_smiles_features = {}
        
        for (drug_id, smiles) in tqdm({**drug_to_smiles, **test_drug_to_smiles}.items(), desc='word vectors'):
            smiles_vector =  np.zeros((embed , self.vector_size), dtype=np.float32)
            #encode the startchar
            smiles_vector[0, :] = model.wv['!']
            #encode the rest of the chars
            for j, c in enumerate(smiles):
                try:
                    smiles_vector[j+1, :] = model.wv[c]
                except KeyError:
                    smiles_vector[j+1, :] = model.wv['?']
            #Encode endchar
            smiles_vector[len(smiles)+1:, :] =  model.wv['E']
            drug_to_smiles_features[drug_id] = smiles_vector
        return drug_to_smiles_features