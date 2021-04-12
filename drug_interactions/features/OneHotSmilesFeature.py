from typing import Dict

import numpy as np
from tqdm import tqdm

def get_smiles_features(self, drug_to_smiles: Dict[str, str], test_drug_to_smiles: Dict[str, str]) -> Dict[str, np.array]:
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