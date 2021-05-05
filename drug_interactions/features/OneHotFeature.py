from typing import Dict

import numpy as np
from tqdm import tqdm

class OneHotFeature():

    def __init__(self, **kwargs):
        pass

    def __repr__(self,):
        return "OneHotFeature"

    def __call__(self, old_drug_bank, new_drug_bank) -> Dict[str, np.array]:
        print('in onehot call')
        train_drug_ids = set(old_drug_bank.id_to_drug.keys())
        test_drug_ids = set(new_drug_bank.id_to_drug.keys())
        new_drug_ids = test_drug_ids - (train_drug_ids & test_drug_ids)

        drug_to_smiles = {}
        for drug_id in train_drug_ids:
            drug_to_smiles[drug_id] = old_drug_bank.id_to_drug[drug_id].smiles

        test_drug_to_smiles = {}
        for drug_id in new_drug_ids:
            test_drug_to_smiles[drug_id] = new_drug_bank.id_to_drug[drug_id].smiles

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
                    raise IndexError from None
            #Encode endchar
            one_hot[len(smiles)+1:,char_to_int["E"]] = 1
            drug_to_smiles_features[drug_id] = one_hot
        print(type(drug_to_smiles_features))
        return drug_to_smiles_features