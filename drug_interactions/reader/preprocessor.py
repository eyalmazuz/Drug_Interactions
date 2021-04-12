import os
import pickle
from copy import deepcopy
from typing import Tuple

from drug_interactions.reader.dal import DrugBank, DrugDAL, Drug

from tqdm import tqdm

class DrugPreprocessor():
    """
    An object that contains all the preprocessing methods for the drug bank.

    Attributes:
        path: A string of the path to load the data from.
    """
    def __init__(self, path: str):
        """
        Inits the class with the data path.
        """
        self.path = path

    def get_preprocessed_drug_bank(self) -> DrugBank:
        """
        Loads the processed drug bank object if exists.
        If the object doesn't exists it creates one by loading the preprocessed drug bank.

        Retruns:
            A drug bank object after removing all the illegal drugs. 
        """
        if os.path.exists(os.path.join(self.path, 'ValidBank.pickle')):
            with open(os.path.join(self.path, 'ValidBank.pickle'), 'rb') as f:
                valid_bank = pickle.load(f)
            
        else:
            print('No Preprocessed Drug Bank found, creating one')
            drug_dal = DrugDAL(self.path)
            bank = drug_dal.read_data_from_file()

            valid_bank = self._validate_drugs(bank)
            with open(os.path.join(self.path, 'ValidBank.pickle'), 'wb') as f:
                pickle.dump(valid_bank, f)
            
        return valid_bank

    def _validate_drugs(self, bank: DrugBank) -> DrugBank:
        """
        Checks that any drug in the drug bank has a least one interaction.
        If a drug doesn't have any interactions with any other drugs it's removed.
        Saves the new bank object as a pickle for future use.

        Args:
            bank: An object containing all the drugs from the Drug Bank data.

        Returns
            A new DrugBank object after removing the drugs that were found to be invalid.
        """
        invalid_drugs = []
        valid_drugs = []
        valid = 0
        invalid = 0
        for drug in tqdm(bank.drugs):
            if not bank.has_symmetric_interaction(drug):
                invalid_drugs.append(drug.id_)
                invalid += 1
            else:
                valid_drugs.append(drug.id_)
                valid += 1

        valid_bank = deepcopy(bank)

        valid_bank.remove_invalid_drugs_and_interactions(set(valid_drugs))
    
        return valid_bank

    @staticmethod
    def find_intersections(old_bank: DrugBank, new_bank: DrugBank) -> Tuple[DrugBank, DrugBank]:
        
        old_drug_ids = set(map(lambda drug: drug.id_, old_bank.drugs))
        new_drug_ids = set(map(lambda drug: drug.id_, new_bank.drugs))

        drug_intersection = old_drug_ids & new_drug_ids
        old_drugs_to_remove = old_drug_ids - drug_intersection
        new_drugs_to_remove = new_drug_ids - drug_intersection

        print(f'drug_intersection', len(drug_intersection))
        print(f'old_drugs_to_remove', len(old_drugs_to_remove))
        print(f'new_drugs_to_remove', len(new_drugs_to_remove))

        old_bank.remove_invalid_drugs_and_interactions(drug_intersection)
        new_bank.remove_invalid_drugs_and_interactions(drug_intersection)
        
        return old_bank, new_bank