


class EmbeddingFeature():

    def __init__(self, **kwargs):
        pass

    def __repr__(self,):
        return "EmbeddingFeature"


    def __call__(self, old_drug_bank, new_drug_bank):
        train_drug_ids = set(old_drug_bank.id_to_drug.keys())
        test_drug_ids = set(new_drug_bank.id_to_drug.keys())
        sorted_drug_ids = sorted(list(train_drug_ids | test_drug_ids))

        drug_to_id = dict(zip(sorted_drug_ids, range(len(sorted_drug_ids))))

        return drug_to_id