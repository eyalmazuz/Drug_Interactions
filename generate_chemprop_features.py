import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

with open("./sorted_ids.txt", 'r') as f:
    sorted_ids = eval(f.read())

df = pd.read_csv('./data/csvs/data/test_new_new_chemprop_similar.csv')

drug_emb = np.load('./drug_emb.npy')[0]
drug_bias = np.load('./drug_bias.npy')[0]

with open('./data/csvs/chemprop/additional_features_new_new_chemprop_similar.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')

    n_features = drug_emb.shape[1]
    header = [f'f{i}' for i in range(n_features)]

    writer.writerow(header)

    drug1_idxs = [sorted_ids.index(drug) for drug in tqdm(df.Drug1_ID.tolist())]
    drug2_idxs = [sorted_ids.index(drug) for drug in tqdm(df.Drug2_ID.tolist())]

    for drug1_idx, drug2_idx in tqdm(zip(drug1_idxs, drug2_idxs)):

        drug1_emb = drug_emb[drug1_idx]
        drug1_bias = drug_bias[drug1_idx]

        drug2_emb = drug_emb[drug2_idx]
        drug2_bias = drug_bias[drug2_idx]

        emb = drug1_emb * drug1_emb + drug1_bias + drug2_bias
        writer.writerow(drug1_emb.tolist() + drug1_bias.tolist() + drug2_emb.tolist() + drug2_bias.tolist())