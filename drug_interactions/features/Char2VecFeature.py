from typing import Dict

from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm

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