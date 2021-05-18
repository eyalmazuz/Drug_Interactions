import json
import requests

from bs4 import BeautifulSoup
from tqdm import tqdm

from drug_interactions.reader.reader import DrugReader
from drug_interactions.datasets.dataset_builder import get_smiles_drugs, get_train_test_ids

URL = "https://go.drugbank.com/structures/search/small_molecule_drugs/structure?database_id="

def load_drugs():

    reader = DrugReader('./data/DrugBankReleases')

    old_drug_bank, new_drug_bank = reader.get_drug_data('5.1.3', '5.1.6')

    return old_drug_bank, new_drug_bank

def filter_drugs(old_drug_bank, new_drug_bank):

    old_drug_bank = get_smiles_drugs(old_drug_bank, atom_size=300)
    new_drug_bank = get_smiles_drugs(new_drug_bank, atom_size=300)

    return old_drug_bank, new_drug_bank

def get_similar_drugs(new_drug_ids):

    similar_drugs_dict = {}
    bad_drugs = 0

    for id_ in (t := tqdm(new_drug_ids)):
        t.set_description(f'Drug: {id_}')

        similar_drugs_dict[id_] = []

        drug_path = f'{URL}{id_}#results'
        # print(drug_path)
        r = requests.get(drug_path)
        soup = BeautifulSoup(r.text, 'html.parser')
        try: 
            similar_drug_table = soup.find('table', {"class": "table table-striped"})
            similar_drug_table = similar_drug_table.find('tbody')
            similar_drugs = similar_drug_table.find_all('tr')

            for drug in similar_drugs:
                result = drug.find_all('td')[0].text
                similar_drug_id, score = result.split('\nScore: ')
                score = float(score)
                similar_drugs_dict[id_].append((similar_drug_id, score))
        
        except AttributeError:
            print(f'Failed to find similar drug for {id_}')
            bad_drugs += 1

    print(f'Total Bad Drugs: {bad_drugs}')    

    return similar_drugs_dict

def save_dict(similar_drugs, path):
    with open(path, 'w') as f:
        json.dump(similar_drugs, f)

def filter_new_drugs(similar_drugs, old_drug_ids):
    new_dict = {}
    for id_, similar_drug_list in similar_drugs.items():
        new_dict[id_] = [(drug_id, score) for drug_id, score in similar_drug_list if drug_id in old_drug_ids]

    return new_dict

def main():

    old_drug_bank, new_drug_bank = load_drugs()

    old_drug_bank, new_drug_bank = filter_drugs(old_drug_bank, new_drug_bank)

    old_drug_ids, new_drug_ids = get_train_test_ids(old_drug_bank, new_drug_bank)

    all_drugs = set(old_drug_ids) | set(new_drug_ids)

    similar_drugs = get_similar_drugs(all_drugs)

    save_dict(similar_drugs, path='./data/jsons/similar_drugs_dict_all.json')

    similar_drugs = filter_new_drugs(similar_drugs, old_drug_ids)

    save_dict(similar_drugs, path='./data/jsons/similar_drugs_dict_only_old.json')


if __name__ == "__main__":
    main()