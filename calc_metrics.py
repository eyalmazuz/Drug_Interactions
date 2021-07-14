import os
from drug_interactions.utils.calc_metrics import calc_metrics



def main():

    # dataset_type = "CHEMPROP"
    for folder in ['NewOldTF', 'NewNewTF']:
        path = f'./data/csvs/results/All_Data/{folder}'
        print(folder)
        for dataset_type in os.listdir(path):
            print(dataset_type)
            if dataset_type != 'Metrics.csv' and dataset_type == "CHEMPROP.csv":
                calc_metrics(path, dataset_type.split('.')[0])

if __name__ == "__main__":
    main()