import json
import pandas as pd
from tqdm import tqdm

def LCSubStr(X, Y):
 
	# Create a table to store lengths of
	# longest common suffixes of substrings.
	# Note that LCSuff[i][j] contains the
	# length of longest common suffix of
	# X[0...i-1] and Y[0...j-1]. The first
	# row and first column entries have no
	# logical meaning, they are used only
	# for simplicity of the program.

	# LCSuff is the table with zero
	# value initially in each cell
	m, n = len(X), len(Y)
	LCSuff = [[0 for k in range(n+1)] for l in range(m+1)]

	# To store the length of
	# longest common substring
	result = 0

	# Following steps to build
	# LCSuff[m+1][n+1] in bottom up fashion
	for i in range(m + 1):
		for j in range(n + 1):
			if (i == 0 or j == 0):
				LCSuff[i][j] = 0
			elif (X[i-1] == Y[j-1]):
				LCSuff[i][j] = LCSuff[i-1][j-1] + 1
				result = max(result, LCSuff[i][j])
			else:
				LCSuff[i][j] = 0
	return result

def main():
	df = pd.read_csv('./data/csvs/data/test_new_old_similar.csv')

	new_drugs = df[["Drug1_ID", "Drug1_SMILES"]].drop_duplicates(subset=["Drug1_ID", "Drug1_SMILES"])
	old_drugs = df[["Drug2_ID", "Drug2_SMILES"]].drop_duplicates(subset=["Drug2_ID", "Drug2_SMILES"])

	similars = {}
	for new_drug_id, new_drug_smiles in tqdm(new_drugs.itertuples(index=False)):
		similars[new_drug_id] = []
		for old_drug_id, old_drug_smiles in old_drugs.itertuples(index=False):
			distance = LCSubStr(new_drug_smiles, old_drug_smiles)
			distance = distance**2 / (len(old_drug_smiles) * len(new_drug_smiles))
			similars[new_drug_id].append((old_drug_id, round(distance, 4)))
		similars[new_drug_id] = sorted(similars[new_drug_id], key=lambda x: x[1], reverse=True)
	with open('./data/jsons/nlcs.json', 'w') as f:
		json.dump(similars, f)

if __name__ == "__main__":
	main()