"""
Containing anything realted to drug bank.
"""
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional
import zipfile
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import xml.etree.ElementTree

from tqdm import tqdm

class DrugDAL():
    """
    Drug reader from the Drug Bank XML.
    """
    xml_file_name = 'full database.xml'
    zip_file_name = 'drugbank_all_full_database.xml.zip'

    def __init__(self, path: str):

        self.path = path

    def read_data_from_file(self):
        """
        Loads Drug Bank data from pickle
        if Drug Bank pickle doesn't exists it creates the data and saves the pickle.
        """
        DrugBank = None
        if os.path.exists(os.path.join(self.path, 'DrugBank.pickle')):
            with open(os.path.join(self.path, 'DrugBank.pickle'), 'rb') as f:
                DrugBank = pickle.load(f)
        else:
            print('No Drug Bank was found, creating one')
            DrugBank = self.__create_bank_data()

        return DrugBank

    def __create_bank_data(self):

        all_drugs = []

        archive = zipfile.ZipFile(os.path.join(self.path, DrugDAL.zip_file_name), 'r')
        db_file = archive.open(DrugDAL.xml_file_name)


        root = xml.etree.ElementTree.parse(db_file).getroot()
        ns = '{http://www.drugbank.ca}'

        for drug in tqdm(root):

            drug_info = {}

            drug_info['id_'] = drug.findtext("{ns}drugbank-id[@primary='true']".format(ns=ns))

            try:
                drug_info['type_'] = drug.attrib['type']
            except:
                drug_info['type_'] = None

            drug_info.setdefault('groups', set())
            for _, group in enumerate(drug.findall("{ns}groups".format(ns=ns))[0].getchildren()):
                drug_info['groups'].add(group.text)

            drug_info.setdefault('categories', set())
            for _, cat in enumerate(drug.findall("{ns}categories".format(ns=ns))[0].getchildren()):
                cat_text = cat.findtext('{ns}category'.format(ns=ns))
                drug_info['categories'].add(cat_text)


            mass = drug.findall("{ns}average-mass".format(ns=ns)) #resides in the main data about the drug
            weight = None
            if len(mass) > 0:
                drug_info['weight'] = float(mass[0].text)


            drug_info['name'] = drug.findtext("{ns}name".format(ns=ns))

            interactions = []
            for interaction in drug.findall("{ns}drug-interactions".format(ns=ns))[0].getchildren():
                other_drug_id = interaction.findtext('{ns}drugbank-id'.format(ns=ns))
                interaction_description = interaction.findtext('{ns}description'.format(ns=ns))
                interactions.append((other_drug_id, interaction_description))

            drug_info['interactions'] = set(interactions)

            enzymes = []
            targets = []
            carriers = []
            transporters = []

            for t in drug.iter("{ns}target".format(ns=ns)):
                for gn in t.iter("{ns}gene-name".format(ns=ns)):
                    if gn is not None and gn.text is not None:
                        targets.append(gn.text)
            for t in drug.iter("{ns}enzymes".format(ns=ns)):
                for gn in t.iter("{ns}gene-name".format(ns=ns)):
                    if gn is not None and gn.text is not None:
                        enzymes.append(gn.text)
            for t in drug.iter("{ns}carriers".format(ns=ns)):
                for gn in t.iter("{ns}gene-name".format(ns=ns)):
                    if gn is not None and gn.text is not None:
                        carriers.append(gn.text)
            for t in drug.iter("{ns}transporters".format(ns=ns)):
                for gn in t.iter("{ns}gene-name".format(ns=ns)):
                    if gn is not None and gn.text is not None:
                        transporters.append(gn.text)

            drug_info['targets'] = set(targets)
            drug_info['enzymes'] = set(enzymes)
            drug_info['carriers'] = set(carriers)
            drug_info['transporters'] = set(transporters)


            tax = drug.find("{ns}classification".format(ns=ns))
            try:
                drug_info['tax_description'] = tax.find("{ns}description".format(ns=ns)).text
            except:
                drug_info['tax_description'] = None
            try:
                drug_info['direct_parent'] = tax.find("{ns}direct-parent".format(ns=ns)).text
            except:
                drug_info['direct_parent'] = None
            try:
                drug_info['kingdom'] = tax.find("{ns}kingdom".format(ns=ns)).text
            except:
                drug_info['kingdom'] = None
            try:
                drug_info['superclass'] = tax.find("{ns}superclass".format(ns=ns)).text
            except:
                drug_info['superclass'] = None
            try:
                drug_info['tax_class'] = tax.find("{ns}class".format(ns=ns)).text
            except:
                drug_info['tax_class'] = None
            try:
                drug_info['subclass'] = tax.find("{ns}subclass".format(ns=ns)).text
            except:
                drug_info['subclass'] = None




            if len(drug.findall("{ns}calculated-properties".format(ns=ns))) > 0:
                for _, prop in enumerate(drug.findall("{ns}calculated-properties".format(ns=ns))[0].getchildren()):
                    if prop.find('{http://www.drugbank.ca}kind').text == 'SMILES':
                        smiles = prop.find('{ns}value'.format(ns=ns)).text
                        drug_info['smiles'] = smiles

            for _, prop in enumerate(drug.findall("{ns}experimental-properties".format(ns=ns))[0].getchildren()):
                if prop.find('{http://www.drugbank.ca}kind').text == 'Molecular Weight':
                    assert weight == None or float(prop.find('{ns}value'.format(ns=ns)).text) == weight, 'found weight twice'
                    weight = float(prop.find('{ns}value'.format(ns=ns)).text)
                    drug_info['weight'] = weight




            drug_info['atc'] = set()
            drug_info['atc2text'] = {}
            for atc in drug.findall("{ns}atc-codes".format(ns=ns))[0].getchildren():
                drug_info['atc'].add(atc.get('code'))
                for atc_child in atc.getchildren():
                    code = atc_child.get('code')
                    text = atc_child.text
                    drug_info['atc2text'][code] = text

            d = Drug(**drug_info)
            all_drugs.append(d)

        bank = DrugBank(self.path.split('/')[-2], all_drugs)

        with open(os.path.join(self.path, 'DrugBank.pickle'), 'wb') as f:
            pickle.dump(bank, f)

        return bank

@dataclass(init=True, repr=True, eq=True)
class Drug():
    """
    Drug object, contains all the data in the drug bank xml about the drug.

    Attributes:
        id_: The ID of the drug in the drug bank.
        name: The name of the drug.
        type: The type of the drug
        approved: A str indicating if the durg is approved or not.
        text: A description of the drug
        smiles: A str of the smiles representation of the drug
        tax_description:
        direct_parent:
        kingdom:
        superclass:
        tax_class:
        subclass:
        weight: float indicating the weight of the drug
        groups:
        categories:
        interactions: List of tuples indicating the drugs and the type of interactions the drug have with other drugs.
        targets:
        enzymes: List of enzymes the drug have.
        carries:
        transporters:
        atc:
        atc2text:

    """
    id_: Optional[str] = None
    name: Optional[str] = None
    type_: Optional[str] = None
    approved: Optional[str] = None
    text: Optional[str] = None
    smiles: Optional[str] = None
    tax_description: Optional[str] = None
    direct_parent: Optional[str] = None
    kingdom: Optional[str] = None
    superclass: Optional[str] = None
    tax_class: Optional[str] = None
    subclass: Optional[str] = None

    weight: Optional[float] = None

    groups: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    interactions: Set[Tuple[str, str]] = field(default_factory=set)
    targets: Set[str] = field(default_factory=set)
    enzymes: Set[str] = field(default_factory=set)
    carriers: Set[str] = field(default_factory=set)
    transporters: Set[str] = field(default_factory=set)
    atc: Set[str] = field(default_factory=set)
    atc2text: Dict[str, str] = field(default_factory=dict)

    def interacts_with(self, other) -> bool:
        """
        Return true if this drug has an interaction with other

        Args:
            other: Drug objects.

        Returns:
            Boolean indicating if the drug has an interation with the other drug.
        """
        for other_id, _ in self.interactions:
            if other_id == other.id_:
                return True

        return False

    def interaction_type(self, other) -> Optional[str]:
        """
        Return string of the interaction the drug has with the other drug if exists
        else retun None

        Args:
            other: Drug objects.

        Returns:
            String description of the interaction or None.
        """
        for other_id, interaction in self.interactions:
            if other_id == other.id_:
                return interaction

        return None

class DrugBank():
    """
    A drug database object containing all the drugs from a specific drug bank release.


    Attributes:
        version: A string of the version the data was extracted from.
        drugs: A list of Drug objects.
        id_to_drug: A dict mapping between drug id and drug for easy access to a specific drug.
    """
    def __init__(self, version: str, drugs: List[Drug]):
        """
        Inits the class with the version of the drug bank used, and the list of drugs fetched from
        the drug bank xml file.
        Creates a dict containing mapping from drug ids to the drug object for easy access for a specific drug data.
        """
        self.version = version
        self.drugs = drugs
        self.id_to_drug = dict(zip(list(map(lambda drug: drug.id_, drugs)), drugs))

    def has_symmetric_interaction(self, drug: Drug) -> bool:
        """
        Checks if drug has interactions with other drugs in the bank.
        Returns true if the drug has at least one interaction and the interaction is symmetric (i.e, other drug has interaction is the drug)

        Args:
            drug: A drug object containing information on the drug from the https://drugbank.ca

        Retuns:
            A boolean indicating if the drug has any symmetric of interaction with other drugs in the bank.
        """
        for drug_id, _ in drug.interactions:
            if drug_id in self.id_to_drug and self.id_to_drug[drug_id].interacts_with(drug):
                return True

        return False

    def remove_invalid_drugs_and_interactions(self, valid_drug_ids: Set[Optional[str]]) -> None:
        """
        Removes all invalid drugs from the drug list.
        Then iterates over all the valid drugs are remove from their interaction list the invalid drugs.

        Args:
            valid_drug_ids: A list of drug ids that are valid (have at least one interaction with other drugs).
        """
        drugs = [drug for drug in self.drugs if drug.id_ in valid_drug_ids]

        for drug in tqdm(drugs):
            drug.interactions = set((drug_id, interaction) for drug_id, interaction in drug.interactions if drug_id in valid_drug_ids)

        self.drugs = drugs
        self.id_to_drug = dict(zip(list(map(lambda drug: drug.id_, drugs)), drugs))


    def __eq__(self, other: object):
        if not isinstance(other, DrugBank):
            raise NotImplementedError
        for (drug_id, drug), (other_id, other_drug) in zip(sorted(self.id_to_drug.items()), sorted(other.id_to_drug.items())):
            if drug_id != other_id or drug != other_drug:
                return False
        
        return True
