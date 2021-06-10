import json

import pandas as pd
from tqdm import tqdm, trange
tqdm.pandas()

def main():
    test = pd.read_csv('./data/csvs/data/test_new_old.csv')
    old_new = pd.read_csv('./data/csvs/data/test_new_old.csv')
    old_new = old_new[['Drug2_ID', 'Drug2_SMILES']] 

    with open('./data/jsons/chemprop_similar_drugs_dict_only_old.json', 'r') as f:
        similar_dict = json.load(f)

    old_drug_ids = test.Drug2_ID.tolist()
    have_similar = {}
    for id_, similar_list in similar_dict.items():
        if similar_list and id_ in test.Drug1_ID.tolist():
            have_similar[id_] = similar_list[0][0]

    new_drug_with_similar = have_similar.keys()

    id2id = {}
    smiles2smiles = {}
    for new_id, old_id in (t := tqdm(have_similar.items())):
        t.set_description(f'{new_id}')
        new_smiles = test[test.Drug1_ID == new_id].iloc[0].Drug1_SMILES
        old_smiles = old_new[old_new.Drug2_ID == old_id].iloc[0].Drug2_SMILES
        id2id[new_id] = old_id
        smiles2smiles[new_smiles] = old_smiles

    test_similar = test.copy()
    test_similar_only = test_similar.copy()

    test_similar_only = test_similar_only[test_similar_only.Drug1_ID.isin(new_drug_with_similar)]
 
    test_similar["Drug1_ID_SIMILAR"] = test_similar.Drug1_ID.progress_apply(lambda i: id2id[i] if i in id2id else i)
    test_similar["Drug1_SMILES_SIMILAR"] = test_similar.Drug1_SMILES.progress_apply(lambda s: smiles2smiles[s] if s in smiles2smiles else s)

    # test_similar["Drug2_ID_SIMILAR"] = test_similar.Drug2_ID.progress_apply(lambda i: id2id[i] if i in id2id else i)
    # test_similar["Drug2_SMILES_SIMILAR"] = test_similar.Drug2_SMILES.progress_apply(lambda s: smiles2smiles[s] if s in smiles2smiles else s)

    test_similar_only["Drug1_ID_SIMILAR"] = test_similar_only.Drug1_ID.progress_apply(lambda i: id2id[i] if i in id2id else i)
    test_similar_only["Drug1_SMILES_SIMILAR"] = test_similar_only.Drug1_SMILES.progress_apply(lambda s: smiles2smiles[s] if s in smiles2smiles else s)

    # test_similar_only["Drug2_ID_SIMILAR"] = test_similar_only.Drug2_ID.progress_apply(lambda i: id2id[i] if i in id2id else i)
    # test_similar_only["Drug2_SMILES_SIMILAR"] = test_similar_only.Drug2_SMILES.progress_apply(lambda s: smiles2smiles[s] if s in smiles2smiles else s)

    print(test_similar.shape)
    print(test_similar_only.shape)

    test_similar.to_csv('./data/csvs/data/test_new_old_chemprop_similar.csv', index=False)
    test_similar_only.to_csv('./data/csvs/data/test_new_old_chemprop_similar_only.csv', index=False)


    
if __name__ == "__main__":
   main()