import os

import pandas as pd

from drug_interactions.utils.calc_metrics import calc_metrics



def main():

    # dataset_type = "CHEMPROP_AFMP"
    old_path = 'NewOld'
    new_path = 'NewNew'
    path = './data/csvs/results/All_Data/'
    # calc_metrics(path, dataset_type)
    for no, nn in zip(sorted(os.listdir(f'{path}/{old_path}')), sorted(os.listdir(f'{path}/{new_path}'))):
        if no != 'Metrics.csv':
            print(no)
            d1 = pd.read_csv(f'{path}/{old_path}/{no}')
            d2 = pd.read_csv(f'{path}/{new_path}/{nn}')
            df = d1.append(d2)
            dataset_type = no.split('.')[0]

            calc_metrics(f'{path}/All', dataset_type, df=df)
            df.to_csv(f'{path}/All/{no}', index=False)
if __name__ == "__main__":
    main()