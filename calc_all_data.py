import os

import numpy as np
import pandas as pd

from drug_interactions.utils.calc_metrics import calc_metrics



def main():

    # dataset_type = "CHEMPROP_AFMP"
    old_path = 'NewOldSimilar'
    new_path = 'NewNewSimilar'
    save_path = 'AllSimilar'
    path = './data/csvs/results/All_Data/'
    # calc_metrics(path, dataset_type)
    for no, nn in zip(sorted(os.listdir(f'{path}/{old_path}')), sorted(os.listdir(f'{path}/{new_path}'))):
        print(no, nn)
        if no != 'Metrics.csv':
            print(no)
            d1 = pd.read_csv(f'{path}/{old_path}/{no}')
            d2 = pd.read_csv(f'{path}/{new_path}/{nn}')
            df = d1.append(d2)
            df['prediction'] = np.array(df['prediction'].tolist()).astype(np.float32)
            dataset_type = no.split('.')[0]

            calc_metrics(f'{path}/{save_path}', dataset_type, df=df)
            df.to_csv(f'{path}/{save_path}/{no}', index=False)
if __name__ == "__main__":
    main()