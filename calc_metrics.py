from drug_interactions.utils.calc_metrics import calc_metrics



def main():

    dataset_type = "CHEMPROP_AFMP"
    path = './data/csvs/results/All_Data/NewNewChempropSimilar'
    calc_metrics(path, dataset_type)

if __name__ == "__main__":
    main()