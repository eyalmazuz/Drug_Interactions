from unittest import TestCase

from drug_interactions.reader.dal import Drug, DrugBank
from drug_interactions.reader.preprocessor import DrugPreprocessor

class DrugPreprocessorTestCase(TestCase):

    def setUp(self):
        self.drug1 = Drug(**{'id_': '1', 'interactions': {('3', 'bar')}})
        self.drug2 = Drug(**{'id_': '2', 'interactions': {('3', 'baz'), ('5', 'goo')}})
        self.drug3 = Drug(**{'id_': '3', 'interactions': {('1', 'bar'), ('2', 'baz')}})
        self.drug4 = Drug(**{'id_': '4', 'interactions': {('1', 'bar')}})

        self.drugBank = DrugBank('12', [self.drug1, self.drug2, self.drug3, self.drug4])

        self.drug2_cleaned = Drug(**{'id_': '2', 'interactions': {('3', 'baz')}})

        self.drugBankRemoved = DrugBank('12', [self.drug1, self.drug2_cleaned, self.drug3])

        self.preprocessor = DrugPreprocessor("foo/bar")

        self.drug1_new = Drug(**{'id_': '1', 'interactions': {('3', 'bar'), ('4', 'bar')}})

        self.drugBank2 = DrugBank('15', [self.drug1_new, self.drug2, self.drug4])
        self.drugBank3 = DrugBank('15', [self.drug1_new, self.drug3, self.drug4])


    def testValidateDrugs(self):
        drugBankCleaned = self.preprocessor._validate_drugs(self.drugBank)
        self.assertEqual(drugBankCleaned, self.drugBankRemoved)
        self.assertNotIn(('5', 'goo'), drugBankCleaned.id_to_drug['2'].interactions)
        self.assertNotIn('4', drugBankCleaned.id_to_drug)

    def testFindIntersections(self):
        d1, d2 = DrugPreprocessor.find_intersections(self.drugBank2, self.drugBank3)
        test_bank = self.drugBank = DrugBank('12', [self.drug1_new, self.drug4])

        self.assertEqual(d1, test_bank)
        self.assertEqual(d2, test_bank)