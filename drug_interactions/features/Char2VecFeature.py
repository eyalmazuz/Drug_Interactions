from typing import Dict

from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm

class Char2VecFeature():

    def __init__(self, **kwargs):

        self.embedding_size = kwargs["embedding_size"]
        self.window = kwargs["window"]
        self.min_count = kwargs["min_count"]
        self.workers = kwargs["workers"]
        self.epochs = kwargs["epochs"]

    def __repr__(self, ):
        return "Char2VecFeature"

    def __call__(self, old_drug_bank, new_drug_bank) -> Dict[str, np.array]:
        train_drug_ids = set(old_drug_bank.id_to_drug.keys())
        test_drug_ids = set(new_drug_bank.id_to_drug.keys())
        new_drug_ids = test_drug_ids - (train_drug_ids & test_drug_ids)

        drug_to_smiles = {}
        for drug_id in train_drug_ids:
            drug_to_smiles[drug_id] = old_drug_bank.id_to_drug[drug_id].smiles

        test_drug_to_smiles = {}
        for drug_id in new_drug_ids:
            test_drug_to_smiles[drug_id] = new_drug_bank.id_to_drug[drug_id].smiles

    
        texts = [['!'] + list(smiles) + ['E'] for smiles in drug_to_smiles.values()] + ['?']
        
        print('training w2v model')
        model = Word2Vec(sentences=texts, size=self.embedding_size, window=self.window, min_count=self.min_count, workers=self.workers)
        model.train(total_examples=model.corpus_count, sentences=texts, epochs=self.epochs)
        embed = max([len(smile) for smile in {**drug_to_smiles, **test_drug_to_smiles}.values()]) + 2
        print('done training')
        drug_to_smiles_features = {}
        
        for (drug_id, smiles) in tqdm({**drug_to_smiles, **test_drug_to_smiles}.items(), desc='word vectors'):
            smiles_vector =  np.zeros((embed , self.embedding_size), dtype=np.float32)
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