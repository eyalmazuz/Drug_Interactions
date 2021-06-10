import os

import pandas as pd

from drug_interactions.utils.calc_metrics import calc_metrics



def main():

    # dataset_type = "CHEMPROP_AFMP"
    path = './data/csvs/results/All_Data/'
    # calc_metrics(path, dataset_type)
    for no, nn in zip(sorted(os.listdir(f'{path}/NewOldChempropSimilar')), sorted(os.listdir(f'{path}/NewNewChempropSimilar'))):
        if no != 'Metrics.csv':
            print(no)
            d1 = pd.read_csv(f'{path}/NewOldSimilar/{no}')
            d2 = pd.read_csv(f'{path}/NewNewSimilar/{nn}')
            df = d1.append(d2)
            dataset_type = no.split('.')[0]

            calc_metrics(f'{path}/AllChempropSimilar', dataset_type, df=df)
            df.to_csv(f'{path}/AllChempropSimilar/{no}', index=False)
if __name__ == "__main__":
    main()