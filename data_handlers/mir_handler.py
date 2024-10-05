import pandas as pd
from typing import Dict
from os.path import join
from utils.utils_general import read_df
import logging
from data_handlers.data_handler import DataHandler
logger = logging.getLogger(__name__)

class DataHandler_Mirna_Mrna(DataHandler):
    """
    Class to ...

    Parameters
    ----------

    Attributes
    ----------
    Same as parameters
    """
    def __init__(self, data_path: str, train_fragments_file: str, added_neg: bool = False):
        DataHandler.__init__(self, data_path).__init__()
        del self.test_filtered_file
        del self.test_complete_file

        # interactions data file
        self.train_fragments_file = train_fragments_file

        # RNA data files
        # it's called srna because the parent class
        self.srna_data_file = "DATA_mirna_eco.csv"
        # TODO ?
        self.mrna_data_with_rbp_file = "DATA_mrna_eco.csv"
        self.mrna_data_with_srna_file = "DATA_mrna_eco.csv"
        self.mrna_accession_id_col_with_rbp = "EcoCyc_accession_id"
        self.mrna_accession_id_col_with_srna = "EcoCyc_accession_id"

        self.rbp_data_file = "DATA_rbp_eco.csv"
        self.rbp_accession_id_col = "EcoCyc_accession_id"

        # interactions features and label columns
        # with open(join(self.data_path,'features_cols.txt'), 'r') as file:
        #     self.features_cols = [line.strip() for line in file]
        self.features_cols=["Seed_match_A"]
        self.label_col = "interaction_label"

    def load_interactions_datasets(self, added_neg):
        train_fragments = read_df(join(self.data_path, self.train_fragments_file))
        if not added_neg:
            train_fragments[self.label_col] = 1
        train_fragments['is_synthetic'] = False
        # print("added_neg: ", added_neg)
        # print("train_fragments: ", train_fragments)
        train_fragments = DataHandler.split_dataset(self, dataset=train_fragments)
        
        return train_fragments
    
