from drug_interactions.utils.calc_metrics import calc_metrics



def main():

    dataset_type = "CHEMPROP_AFMP_SIMILAR_OLD"
    path = './data/csvs/results/All_Data/Tricks'
    calc_metrics(path, dataset_type)

if __name__ == "__main__":
    main()