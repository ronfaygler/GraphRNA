from typing import Dict
from os.path import join
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils import shuffle

from utils.utils_general import get_logger_config_dict, write_df
from data_handlers.data_handler import DataHandler
from data_handlers.mir_handler import DataHandler_Mirna_Mrna

# --- not rbp:
from models_handlers.graph_rna_model_handler import GraphRNAModelHandler

import shap
import matplotlib.pyplot as plt
import os

from models_handlers.xgboost_model_handler import XGBModelHandler
from models_handlers.rf_model_handler import RFModelHandler
import logging.config
logging.config.dictConfig(get_logger_config_dict(filename="main", file_level='DEBUG'))
logger = logging.getLogger(__name__)

# --- for rbp:
# from models_handlers.three_graph_rna_model_handler import GraphRNAModelHandler



def main():

# ------ triple: mirna mrna rbp:
    # ----- configuration
    # data="triple"
    # data_path = "/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp"
    # outputs_path = "/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir_rbp"
    # print("paths")

    # # ----- load data for GraphRNA:
    # train_fragments, kwargs = load_data_triple(data_path=data_path, added_neg=False, is_rbp=True)

    # # ----- run GraphRNA
    # model_name = "GNN"
    # graph_rna = GraphRNAModelHandler()
    # test = None
    # cv_predictions_dfs = train_and_evaluate(model_h=graph_rna, train_fragments=train_fragments, test=test, model_name=model_name , data=data, **kwargs)
    # # write cv results to folds dfs
    # for fold, fold_df in cv_predictions_dfs.items():
    #     write_df(df=fold_df, file_path=join(join(outputs_path, 'GNN'), f"cv_fold{fold}_predictions_GraphRNA.csv"))


# # ------ mirna mrna:
#     # ----- configuration
    data="mirna"
    data_path = "/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir"
    outputs_path = "/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir"
    neg_dir = "/home/ronfay/Data_bacteria/graphNN/GraphRNA/neg_data"
    print("paths")

    # -- train-test split 
    # Define the root directories for train and test
    train_root = "Train_Test_files/DATA TRAIN"
    test_root = "Train_Test_files/DATA TEST"

    train_test=True

    # Number of directories (from 0 to 19)
    num_folders = 20

    # Function to get the file that starts with 'NPS_CLIP_Random' from a directory
    def get_random_clip_file(dir_path, prefix="NPS_CLIP_Random"):
        for filename in os.listdir(dir_path):
            if filename.startswith(prefix):
                return os.path.join(dir_path, filename)
        return None  # Return None if no matching file is found

    model_name = "GNN"
    graph_rna = GraphRNAModelHandler()

    dummy_x_train, dummy_x_val = pd.DataFrame(), pd.DataFrame()
    dummy_y_train, dummy_y_val = list(), list()
    dummy_meta_train, dummy_meta_val = pd.DataFrame(), pd.DataFrame()

    # 6 - predict on folds
    cv_training_history = {}
    # cv_prediction_dfs = {}
    train_neg_sampling = False  # negatives were already added to cv_data_unq
    cv_predictions_dfs = [] # To collect CV predictions

    # Loop through 20 folders for both train and test
    for i in range(num_folders):
        # Construct paths to the current train and test directories
        train_dir = os.path.join(train_root, str(i))
        test_dir = os.path.join(test_root, str(i))

        # Get the train and test file starting with 'NPS_CLIP_Random'
        train_file = get_random_clip_file(train_dir)
        test_file = get_random_clip_file(test_dir)
        print( "train_file: ", train_file, "test_file: ", test_file)
        if train_file and test_file:
            train_fragments, test, kwargs = load_data_mir(data_path=data_path, neg_path='', added_neg=False, train_file=train_file, 
                                                        test_file=test_file)
            print(f"Loop {i}:")

        else:
            print(f"Loop {i}: Train or Test file not found.")

        cv_n_splits = 1
    
        test_predictions_df = train_and_evaluate(model_h=graph_rna, train_fragments=train_fragments, test=test, model_name=model_name , data=data, train_test=train_test, **kwargs)
        cv_predictions_dfs.append(test_predictions_df)
        
        break

    all_folds_predictions = pd.concat(cv_predictions_dfs).reset_index(drop=True)
    return cv_predictions_dfs, all_folds_predictions

        # cv_predictions_dfs, cv_training_history = \
        #     model_h.run_cross_validation(X=train_fragments['X'], y=train_fragments['y'], 
        #     metadata=train_fragments['metadata'], n_splits=cv_n_splits, model_args=model_args, 
        #     srna_acc_col='miRNA ID', mrna_acc_col='Gene_ID',neg_df=neg_df, **kwargs)
                
        # dummy_x_train, dummy_x_val, dummy_y_train, dummy_y_val, dummy_meta_train, dummy_meta_val, 
        # cv_training_history, cv_prediction_dfs = \
        #             train_and_evaluate(dummy_x_train, dummy_x_val, dummy_y_train, dummy_y_val,
        #             dummy_meta_train, dummy_meta_val, cv_training_history, cv_prediction_dfs, train_neg_sampling, 
        #             model_h=graph_rna, train_fragments=train_fragments, test=test, 
        #             model_name=model_name , data=data, iteration=i+1, train_test=train_test, **kwargs)

        # for fold, fold_data_unq in cv_data_unq.items():
            
        #     # 6.1 - predict on validation set (pos + random sampled neg)
        #     predictions, training_history = \
        #         cls.train_and_test(X_train=dummy_x_train, y_train=dummy_y_train, X_test=dummy_x_val, y_test=dummy_y_val,
        #                            model_args=model_args, metadata_train=dummy_meta_train, metadata_test=dummy_meta_val,
        #                            unq_train=fold_data_unq['unq_train'], unq_test=fold_data_unq['unq_val'],
        #                            train_neg_sampling=train_neg_sampling, srna_acc_col=srna_acc_col, mrna_acc_col=mrna_acc_col, **kwargs)
        #     # 6.2 - fold's training history
        #     cv_training_history[fold] = training_history
        #     # 6.3 - fold's predictions df
        #     y_val_graph_score = predictions['test_y_graph_score']
        #     unq_val = fold_data_unq['unq_val'][[cls.srna_nid_col, cls.mrna_nid_col]]
        #     y_val = fold_data_unq['unq_val'][cls.binary_intr_label_col]
        #     cv_pred_df = cls.get_predictions_df(unq_intr=unq_val, y_true=y_val, y_score=y_val_graph_score)
        #     cv_prediction_dfs[fold] = cv_pred_df

        # return cv_prediction_dfs, cv_training_history


    # add neg for the first time . data for XGBoost / RandomForest:
    # combine_pos_neg_samples(data_path=data_path , pos_path="h3.csv",neg_dir=neg_dir,  neg_path="Mock_miRNA.csv", ratio=1, _shuffle=True)
    
    # ----- load data without neg:
    train_fragments, kwargs = load_data_mir(data_path=data_path, neg_path='', added_neg=False)

    # ----- load data for XGBoost / RandomForest (with neg)
    # train_fragments, kwargs = load_data_mir(data_path=data_path, added_neg=True)

    # ----- run GraphRNA
    model_name = "GNN"
    graph_rna = GraphRNAModelHandler()
    test = None
    cv_predictions_dfs = train_and_evaluate(model_h=graph_rna, train_fragments=train_fragments, test=test, model_name=model_name , data=data, **kwargs)

    # write cv results to folds dfs
    for fold, fold_df in cv_predictions_dfs.items():
        write_df(df=fold_df, file_path=join(join(outputs_path, 'GNN-Random_neg'), f"cv_fold{fold}_predictions_GraphRNA.csv"))

    # # # ----- run XGBoost
    # model_name = "XGB"
    # xgb = XGBModelHandler()
    # test = None
    # cv_predictions_dfs = train_and_evaluate(model_h=xgb, train_fragments=train_fragments, test=test, model_name=model_name, data=data, **kwargs)
    # for fold, fold_df in cv_predictions_dfs.items():
    #     write_df(df=fold_df, file_path=join(join(outputs_path, 'XGB'), f"cv_fold{fold}_predictions_XGBoost.csv"))

    # # # ----- run RandomForest
    # model_name = "RF"
    # rf = RFModelHandler()
    # test = None
    # cv_predictions_dfs = train_and_evaluate(model_h=rf, train_fragments=train_fragments, test=test, model_name=model_name, data=data, **kwargs)
    # for fold, fold_df in cv_predictions_dfs.items():
    #     write_df(df=fold_df, file_path=join(join(outputs_path, 'RF'), f"cv_fold{fold}_predictions_RandomForest.csv"))

    # return
    

# ------ srna mrna:

    # ----- configuration
    # data="srna"

    # data_path = "/home/ronfay/Data_bacteria/graphNN/GraphRNA/data"
    # outputs_path = "/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs"
    # print("paths")

    # # ----- load data
    # train_fragments, test_complete, test_filtered, kwargs = load_data(data_path=data_path)

    # # ----- run GraphRNA
    # model_name = "GNN"
    # graph_rna = GraphRNAModelHandler()
    # test = test_complete
    # cv_predictions_dfs, test_predictions = \
    #     train_and_evaluate(model_h=graph_rna, train_fragments=train_fragments, test=test,  model_name=model_name, data=data, **kwargs)
    # write_df(df=test_predictions, file_path=join(outputs_path, f"test_predictions_GraphRNA.csv"))
    # write cv results to folds dfs
    # for fold, fold_df in cv_predictions_dfs.items():
    #     write_df(df=fold_df, file_path=join(outputs_path, f"cv_fold{fold}_predictions_GraphRNA.csv"))

    # # ----- run XGBoost
    # model_name = "XGB"
    # xgb = XGBModelHandler()
    # test = test_filtered
    # cv_predictions_dfs, test_predictions = \
    #     train_and_evaluate(model_h=xgb, train_fragments=train_fragments, test=test,  model_name=model_name, data=data, **kwargs)
    # # write_df(df=test_predictions, file_path=join(outputs_path, f"test_predictions_XGBoost.csv"))

    # # ----- run RandomForest
    # model_name = "RF"
    # rf = RFModelHandler()
    # test = test_filtered
    # cv_predictions_dfs, test_predictions = \
    #     train_and_evaluate(model_h=rf, train_fragments=train_fragments, test=test,  model_name=model_name, data=data, **kwargs)
    # # write_df(df=test_predictions, file_path=join(outputs_path, f"test_predictions_RandomForest.csv"))

    # return


#----------- combine pos+neg
def combine_pos_neg_samples(data_path: pd.DataFrame, pos_path: str, neg_dir: str, neg_path: str, ratio: float, _shuffle: bool = False):
    train_df = pd.read_csv(join(data_path, pos_path))
    # remove type row
    train_df = train_df.iloc[1:].reset_index(drop=True)
    train_df['interaction_label'] = 1

    neg_df =  pd.read_csv(join(neg_dir, neg_path))
    # remove type row
    neg_df = neg_df.iloc[1:].reset_index(drop=True)
    neg_df['interaction_label'] = 0

    assert len(train_df) <= len(neg_df), "neg df is smaller than pos df"
    
    # create df neg+pos 
    n = max(int(len(train_df) * ratio), 1)
    _neg_samples = neg_df.sample(n=n, random_state=20)
    out = pd.concat(objs=[train_df, _neg_samples], axis=0, ignore_index=True).reset_index(drop=True)
    if _shuffle:
        out = pd.DataFrame(shuffle(out)).reset_index(drop=True)
    out.to_csv(join(data_path, f"combined_train_{neg_path[:-4]}.csv"))
    return out, neg_df


def load_data_mir(data_path: str, neg_path: str='', added_neg: bool = False, is_rbp: bool = False, train_file: str="", test_file: str = ""):
    if train_file!="":
        train_fragments_file = train_file
    elif added_neg:
        train_fragments_file = f"combined_train_{neg_path[:-4]}.csv"
    else:
        train_fragments_file = "h3.csv"
    dhm = DataHandler_Mirna_Mrna(data_path=data_path, train_fragments_file=train_fragments_file, added_neg=added_neg, is_rbp=is_rbp, test_file=test_file)
    train_fragments, test = dhm.load_interactions_datasets(added_neg=added_neg)
    mirna_eco, mrna_eco, mirna_eco_accession_id_col, mrna_eco_accession_id_col = dhm.load_rna_data()

    # 2.1 - update kwargs
    # when running model that is not GNN - this kwargs doesn't correspond to the pos+neg data, but it doesn't matter in the code, 
    # because there is no use of the kwargs in this models.
    kwargs = {
        'srna_eco': mirna_eco,
        'mrna_eco': mrna_eco,
        'se_acc_col': mirna_eco_accession_id_col,
        'me_acc_col': mrna_eco_accession_id_col
    }

    return train_fragments, test, kwargs

def load_data_triple(data_path: str, added_neg: bool = False, is_rbp:bool = True):
    # if added_neg:
    #     train_fragments_file = "combined_train.csv"
    # else:
    #     train_fragments_file = "h3.csv"
    train_fragments_file = "combined_rbp_mirna_interactions.csv"

    dhm = DataHandler_Mirna_Mrna(data_path=data_path, train_fragments_file=train_fragments_file, added_neg=added_neg, is_rbp=is_rbp)
    train_fragments = dhm.load_interactions_datasets(added_neg=added_neg)
    mirna_eco, mirna_eco_accession_id_col, mrna_eco_with_rbp, mrna_eco_with_rbp_accession_id_col, mrna_eco_with_mirna, mrna_eco_with_mirna_accession_id_col, rbp_eco, rbp_eco_accession_id_col = dhm.load_rna_triple_data()

    # 2.1 - update kwargs
    # when running model that is not GNN - this kwargs doesn't correspond to the pos+neg data, but it doesn't matter in the code, 
    # because there is no use of the kwargs in this models.
    kwargs = {
        'srna_eco': mirna_eco,
        'me_acc_col_with_srna': mrna_eco_with_mirna_accession_id_col,
        'me_acc_col_with_rbp' : mrna_eco_with_rbp_accession_id_col,
        'mrna_eco_with_srna': mrna_eco_with_mirna,
        'mrna_eco_with_rbp': mrna_eco_with_rbp,
        'rbp_eco': rbp_eco,
        're_acc_col': rbp_eco_accession_id_col,
        'se_acc_col': mirna_eco_accession_id_col,
    }

    return train_fragments, kwargs

def load_data(data_path: str):
    """
    :param data_path: str
    :return:
    train_fragments, test_complete, test_filtered - dicts in the following format:
        {
            'X': pd.DataFrame,
            'y': List[int],
            'metadata': pd.DataFrame
        }
    kwargs - dict in the following format:
        {
            'srna_eco': pd.DataFrame,
            'mrna_eco': pd.DataFrame,
            'se_acc_col': str (the column in srna_eco containing unique id per sRNA),
            'me_acc_col': str (the column in mrna_eco containing unique id per mRNA)
        }
    """
    # ----- data
    # 1 - load interactions datasets
    """
    train_fragments: includes 14806 interactions: 7403 positives, 7403 synthetic negatives (see evaluation 1).
                     (synthetic samples are ignored when training GraphRNA)
                     RNA sequences are chimeric fragments
    test_complete: includes 391 interactions: 227 positives, 164 negatives (see evaluation 4).
    test_filtered: includes 342 interactions: 199 positives, 143 negatives (see evaluation 3).
    """
    dh = DataHandler(data_path=data_path)
    train_fragments, test_complete, test_filtered = dh.load_interactions_datasets()
    # 2 - load RNA data  (for GraphRNA)
    """
    srna_eco: includes 94 unique sRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.
    mrna_eco: includes 4300 unique mRNAs of Escherichia coli K12 MG1655 (NC_000913) from EcoCyc.

    Note: sRNA/mRNA accession ids in the RNA data must match the accession ids in the interactions datasets.
    """
    srna_eco, mrna_eco, srna_eco_accession_id_col, mrna_eco_accession_id_col = dh.load_rna_data()
    # 2.1 - update kwargs
    kwargs = {
        'srna_eco': srna_eco,
        'mrna_eco': mrna_eco,
        'se_acc_col': srna_eco_accession_id_col,
        'me_acc_col': mrna_eco_accession_id_col
    }

    return train_fragments, test_complete, test_filtered, kwargs


def train_and_evaluate(model_h, train_fragments: Dict[str, object], 
                test: Dict[str, object], model_name: str="", data: str="mirna", neg_df: pd.DataFrame=None, 
                train_test: bool=False, **kwargs) -> (Dict[int, pd.DataFrame], pd.DataFrame):
    """
    Returns
    -------

    cv_prediction_dfs - Dict in the following format:  {
        <fold>: pd.DataFrame including the following information:
                sRNA accession id, mRNA accession id, interaction label (y_true),
                model's prediction score (y_score OR y_graph_score), metadata columns.
        }
    }

    test_predictions_df - pd.DataFrame including the following information:
                          sRNA accession id, mRNA accession id, interaction label (y_true),
                          model's prediction score (y_score OR y_graph_score), metadata columns.
    """
    # 1 - define model args
    model_args = model_h.get_model_args()
    # 2 - run cross validation
    cv_n_splits = 1

    # ------------mrna mirna:
    if model_name == "XGB" or model_name == "RF":
        # Identify columns to be removed - strings
        for i in range(1, 21):
            name_col = "miRNAMatchPosition_" + str(i)
            train_fragments['X'][name_col] = train_fragments['X'][name_col].astype('category').cat.codes
        
        def save_dataset(dataset, file_prefix):
            dataset['X'].to_csv(f"{file_prefix}_features.csv", index=False)
            pd.DataFrame(dataset['y'], columns=['y']).to_csv(f"{file_prefix}_labels.csv", index=False)
            dataset['metadata'].to_csv(f"{file_prefix}_metadata.csv", index=False)

        save_dataset(train_fragments, join(data_path,"efrat_h3_and_Mock_miRNA"))
        # Remove identified columns from the DataFrame
        # train_fragments['X'] = train_fragments['X'].drop(columns=miRNA_columns)

    # 2 - train
    
# ------------mrna mirna:
    if data == "mirna":
        if train_test:
            # 3 - train and test
            predictions, training_history = \
                model_h.train_and_test(X_train=train_fragments['X'], y_train=train_fragments['y'], X_test=test['X'],
                                    y_test=test['y'], model_args=model_args, train_neg_sampling = False,
                                    metadata_train=train_fragments['metadata'], metadata_test=test['metadata'],
                                    srna_acc_col='miRNA ID', mrna_acc_col='Gene_ID', **kwargs)
            test_predictions_df = predictions['out_test_pred']
            
            return test_predictions_df

            # dummy_x_train, dummy_x_val, dummy_y_train, dummy_y_val, dummy_meta_train, dummy_meta_val, 
            # cv_training_history, cv_prediction_dfs = \
            #     model_h.use_train_test(X=train_fragments['X'], y=train_fragments['y'], 
            #     metadata=train_fragments['metadata'], n_splits=cv_n_splits, model_args=model_args, 
            #     srna_acc_col='miRNA ID', mrna_acc_col='Gene_ID',neg_df=neg_df, iteration=iteration, **kwargs)
                    
            # return dummy_x_train, dummy_x_val, dummy_y_train, dummy_y_val, dummy_meta_train, dummy_meta_val, cv_training_history, cv_prediction_dfs
        else:
            cv_predictions_dfs, cv_training_history = \
                model_h.run_cross_validation(X=train_fragments['X'], y=train_fragments['y'], 
                metadata=train_fragments['metadata'], n_splits=cv_n_splits, model_args=model_args, 
                train_neg_sampling = False, srna_acc_col='miRNA ID', mrna_acc_col='Gene_ID',neg_df=neg_df, **kwargs)
                    
            return cv_predictions_dfs

# ------------triple:
    if data == "triple":
        cv_predictions_dfs, cv_training_history = \
            model_h.run_cross_validation(X=train_fragments['X'], y_srna=train_fragments['y_srna'], 
                y_rbp=train_fragments['y_rbp'],
                metadata=train_fragments['metadata'], n_splits=cv_n_splits, model_args=model_args, 
                srna_acc_col='miRNA ID', rbp_acc_col='RBP', 
                mrna_acc_with_srna_col='mRNA_ID_with_sRNA' , mrna_acc_with_rbp_col='mRNA_ID_with_RBP',
                **kwargs)
                
        return cv_predictions_dfs

# ----------srna mrna:
    else:
        cv_predictions_dfs, cv_training_history = \
            model_h.run_cross_validation(X=train_fragments['X'], y=train_fragments['y'],
                                        metadata=train_fragments['metadata'], n_splits=cv_n_splits,
                                        model_args=model_args, **kwargs)
        # 3 - train and test
        predictions, training_history = \
            model_h.train_and_test(X_train=train_fragments['X'], y_train=train_fragments['y'], X_test=test['X'],
                                y_test=test['y'], model_args=model_args,
                                metadata_train=train_fragments['metadata'], metadata_test=test['metadata'], **kwargs)
        test_predictions_df = predictions['out_test_pred']
        
        return cv_predictions_dfs , test_predictions_df
    



if __name__ == "__main__":
    main()
