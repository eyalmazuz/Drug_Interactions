import json
import pandas as pd
from tqdm import tqdm 

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def main():
	df = pd.read_csv('./data/csvs/data/test_new_old_similar.csv')

	new_drugs = df[["Drug1_ID", "Drug1_SMILES"]].drop_duplicates(subset=["Drug1_ID", "Drug1_SMILES"])
	old_drugs = df[["Drug2_ID", "Drug2_SMILES"]].drop_duplicates(subset=["Drug2_ID", "Drug2_SMILES"])

	similars = {}
	for new_drug_id, new_drug_smiles in tqdm(new_drugs.itertuples(index=False)):
		similars[new_drug_id] = []
		for old_drug_id, old_drug_smiles in old_drugs.itertuples(index=False):
			distance = levenshteinDistance(new_drug_smiles, old_drug_smiles)
			distance = 1 - (distance / max(len(old_drug_smiles), len(new_drug_smiles)))
			similars[new_drug_id].append((old_drug_id, round(distance, 4)))
		similars[new_drug_id] = sorted(similars[new_drug_id], key=lambda x: x[1], reverse=True)
	with open('./data/jsons/edit_distance.json', 'w') as f:
		json.dump(similars, f)

if __name__ == "__main__":
	main()