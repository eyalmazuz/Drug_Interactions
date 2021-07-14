import json
import pandas as pd
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm 

def main():
	df = pd.read_csv('./data/csvs/data/test_new_old_similar.csv')

	new_drugs = df[["Drug1_ID", "Drug1_SMILES"]].drop_duplicates(subset=["Drug1_ID", "Drug1_SMILES"])
	old_drugs = df[["Drug2_ID", "Drug2_SMILES"]].drop_duplicates(subset=["Drug2_ID", "Drug2_SMILES"])

	new_drug_smiles = new_drugs.Drug1_SMILES.unique()
	old_drug_smiles = old_drugs.Drug2_SMILES.unique()


	old_vectorizer = TfidfVectorizer(use_idf=False, analyzer='char', lowercase=False)
	old_vectorizer.fit(old_drug_smiles)
	

	similars = {}
	for new_drug_id, new_drug_smiles in tqdm(new_drugs.itertuples(index=False)):
		similars[new_drug_id] = []
		for old_drug_id, old_drug_smiles in old_drugs.itertuples(index=False):
			new_vector = old_vectorizer.transform([new_drug_smiles]).toarray()
			old_vector = old_vectorizer.transform([old_drug_smiles]).toarray()
			distance = 1 - spatial.distance.cosine(new_vector, old_vector)
			similars[new_drug_id].append((old_drug_id, round(distance, 4)))
		similars[new_drug_id] = sorted(similars[new_drug_id], key=lambda x: x[1], reverse=True)
	with open('./data/jsons/tf_cosine.json', 'w') as f:
		json.dump(similars, f)

if __name__ == "__main__":
	main()