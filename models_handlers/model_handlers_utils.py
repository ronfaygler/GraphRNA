# Authors: Shani Cohen (ShaniCohen)
# Python version: 3.8
# Last update: 19.11.2023

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from os.path import join
from sklearn.metrics import det_curve, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, auc, \
    log_loss, top_k_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, jaccard_score, \
    matthews_corrcoef, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from utils import write_df
from visualization.figures_generator import FiguresGenerator
import numbers
import logging
logger = logging.getLogger(__name__)


def get_element_closest_to_val(numbers_arr: np.array, val: numbers.Number) -> (numbers.Number, int):
    """

    Parameters
    ----------
    floats_arr - of floats
    val

    Returns
    -------
    _value: float
    _index: int

    """

    assert sum([isinstance(x, numbers.Number) for x in numbers_arr]) == len(numbers_arr), \
        "numbers_arr contains invalid elements"
    assert isinstance(val, numbers.Number), "val is not a number"
    _index = np.argmin(np.abs(numbers_arr - val))
    _value = numbers_arr[_index]

    return _value, _index


def calc_binary_classification_metrics_using_y_pred(y_true: list, y_pred: np.array, dataset_nm: str = None) -> \
        Dict[str, object]:
    """
    https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

    Parameters
    ----------
    y_true: true binary labels
    y_pred: predicted binary labels
    dataset_nm

    Returns
    -------
    scores - Dict in the following format: {
                '<dataset_nm>_MCC': <score>,
                '<dataset_nm>_ACCURACY': <score>,
                '<dataset_nm>_PRECISION': <score>,
                '<dataset_nm>_RECALL': <score>,
                '<dataset_nm>_F1': <score>,
            }
    """
    logger.debug(f"calculating binary classification metrics using y_pred over {len(y_true)} samples"
                 f"{' of dataset = ' + dataset_nm if dataset_nm else ''}")
    _pref = f"{dataset_nm}_" if dataset_nm else ""
    # 1 - validate inputs
    _valid_input = True
    _valid_y_ture = sorted(set(y_true)) == [0, 1]
    if not _valid_y_ture:
        logger.error(f"y_true values are {sorted(set(y_true))} (expected [0, 1])  ->  metrics are set to None")
        _valid_input = False
    _valid_y_pred = sorted(set(y_pred)) == [0, 1]
    if not _valid_y_pred:
        logger.error(f"y_pred values are {sorted(set(y_pred))} (expected [0, 1])  ->  metrics are set to None")
        _valid_input = False
    # 2 - calc scores
    scores = {
        f"{_pref}ACCURACY": accuracy_score(y_true=y_true, y_pred=y_pred) if _valid_input else None,
        f"{_pref}PRECISION": precision_score(y_true=y_true, y_pred=y_pred) if _valid_input else None,
        f"{_pref}RECALL": recall_score(y_true=y_true, y_pred=y_pred) if _valid_input else None,
        f"{_pref}F1": f1_score(y_true=y_true, y_pred=y_pred) if _valid_input else None
    }
    return scores


def aupr_score(y_true, y_score) -> float:
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(recall, precision)


def calc_binary_classification_metrics_using_y_score(y_true: list, y_score: np.array, roc_max_fpr: float = None,
                                                     dataset_nm: str = None) -> Dict[str, object]:
    """
    https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

    Parameters
    ----------
    y_true: true binary labels
    y_score: positive numeric scores
    dataset_nm: Optional

    Returns
    -------
    scores - Dict in the following format: {
                '<dataset_nm>_ROC_AUC': <score>,
                '<dataset_nm>_AUPR': <score>,
                '<dataset_nm>_PARTIAL_ROC_AUC_max_fpr_{roc_max_fpr}': <score> - optional
            }
    """
    logger.debug(f"calculating binary classification metrics using y_score over {len(y_true)} samples"
                 f"{' of dataset = ' + dataset_nm if dataset_nm else ''}")
    _pref = f"{dataset_nm}_" if dataset_nm else ""
    # validate inputs
    _valid_input = True
    _valid_y_ture = sorted(set(y_true)) == [0, 1]
    if not _valid_y_ture:
        logger.error(f"y_true values are {sorted(set(y_true))} (expected [0, 1])  ->  metrics are set to None")
        _valid_input = False
    _valid_y_score = sum([pd.notnull(x) and 0 <= x for x in y_score]) == len(y_score)
    if not _valid_y_score:
        logger.error(f"{len(y_score)- sum([pd.notnull(x) and 0 <= x for x in y_score])} "
                     f"elements in y_score are invalid (expected: 0 <= element)")
        _valid_input = False
    # calculate metrics
    scores = {
        f"{_pref}ROC_AUC": roc_auc_score(y_true=y_true, y_score=y_score) if _valid_input else None,
        f"{_pref}AUPR": aupr_score(y_true=y_true, y_score=y_score) if _valid_input else None
    }

    if roc_max_fpr:
        _valid_max_fpr = 0 < roc_max_fpr < 1
        if not _valid_max_fpr:
            logger.error(f"max_fpr = {roc_max_fpr} is invalid")
        p_auc_score = roc_auc_score(y_true=y_true, y_score=y_score, max_fpr=roc_max_fpr) if _valid_max_fpr else None
        scores.update({
            f"{_pref}PARTIAL_ROC_AUC_max_fpr_{roc_max_fpr}": p_auc_score
        })

    return scores


def calc_binary_classification_metrics_using_prob_y_score_and_y_pred_thresholds(
        y_true: list, y_score: np.array, y_pred_thresholds: List[float], dataset_nm: str = None) -> Dict[str, object]:
    """
    https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

    Parameters
    ----------
    y_true: true binary labels
    y_score: predicted scores (probabilities)
    y_pred_thresholds: list of thresholds, where   0 < threshold < 1
    dataset_nm: Optional

    Returns
    -------
    scores - Dict in the following format: {
                '<dataset_nm>_ROC_AUC': <score>,
                '<dataset_nm>_AUPR': <score>,
            }
    """

    logger.debug(f"calculating binary classification metrics using y_score and y_pred thresholds = {y_pred_thresholds} "
                 f"over {len(y_true)} samples"
                 f"{' of dataset = ' + dataset_nm if dataset_nm else ''}")
    _pref = f"{dataset_nm}_" if dataset_nm else ""
    scores = {}
    # 1 - validate inputs
    _valid_input = True
    _valid_y_ture = sorted(set(y_true)) == [0, 1]
    if not _valid_y_ture:
        logger.error(f"y_true values are {sorted(set(y_true))} (expected [0, 1])  ->  metrics are set to None")
        _valid_input = False
    _valid_y_score = sum([pd.notnull(x) and 0 <= x <= 1 for x in y_score]) == len(y_score)
    if not _valid_y_score:
        logger.error(f"{len(y_score) - sum([pd.notnull(x) and 0 <= x <= 1 for x in y_score])} "
                     f"elements in y_score are invalid (expected: 0 <= element <= 1)")
        _valid_input = False
    _valid_y_pred_thresholds = sum([0 < x < 1 for x in y_pred_thresholds]) == len(y_pred_thresholds)
    if not _valid_y_pred_thresholds:
        logger.error(f"{len(y_pred_thresholds) - sum([0 < x < 1 for x in y_pred_thresholds])} "
                     f"elements in y_pred_thresholds are invalid (expected: 0 < element < 1)")
        _valid_input = False
    # 2 - define y_pred per threshold
    for thresh in y_pred_thresholds:
        y_pred = np.array(y_score > thresh).astype(int)
        t_scores = calc_binary_classification_metrics_using_y_pred(y_true=y_true, y_pred=y_pred,
                                                                   dataset_nm=f'{_pref}thresh_{thresh}')
        scores.update(t_scores)
    # calculate metrics
    scores.update({
        f"{_pref}ROC_AUC": roc_auc_score(y_true=y_true, y_score=y_score) if _valid_input else None,
        f"{_pref}AUPR": aupr_score(y_true=y_true, y_score=y_score) if _valid_input else None
    })

    return scores


def get_predictions_df(X: pd.DataFrame, y_true: list, y_score: np.array, out_col_y_true: str = "y_true",
                       out_col_y_score: str = "y_score", metadata: pd.DataFrame = None, sort_df: bool = True) \
        -> pd.DataFrame:
    is_length_compatible = len(X) == len(y_true) == len(y_score) if metadata is None \
        else len(X) == len(y_true) == len(y_score) == len(metadata)
    assert is_length_compatible, "X, y_true, y_score and metadata are not compatible in length"

    _df = pd.DataFrame(X)
    if metadata is not None:
        _df = pd.concat(objs=[_df, metadata], axis=1).reset_index(drop=True)
    _df[out_col_y_true] = y_true
    _df[out_col_y_score] = y_score
    if sort_df:
        _df = _df.sort_values(by=out_col_y_score, ascending=False).reset_index(drop=True)

    return _df


def split_cv_data(X: pd.DataFrame, y: np.array, metadata: pd.DataFrame, fold_col: str) -> \
        Dict[int, Dict[str, object]]:
    """

    Parameters
    ----------
    X - pd.DataFrame (m_samples, N_features)
    y - np.array (m_samples,)
    metadata - Optional - pd.DataFrame (m_samples, T_features)

    Returns
    -------
    cv_folds - Dict in the following format: {
        1:   {
            "X_train": pd.DataFrame (n_samples, N_features),
            "y_train": list (n_samples,),
            "X_val": pd.DataFrame (k_samples, K_features),
            "y_val": list (k_samples,),
            "metadata_train": pd.DataFrame (n_samples, T_features),
            "metadata_val": list (k_samples,)
            },
        2:  {
        }, ...
    }
    """
    # logger.debug(f"getting stratified {n_splits} cv folds")
    is_length_compatible = len(X) == len(y) if metadata is None else len(X) == len(y) == len(metadata)
    assert is_length_compatible, "X, y, and metadata are compatible in length"

    # 1 - data preparation
    # 1.1 - reset df indexes
    X = X.reset_index(drop=True)
    metadata = metadata.reset_index(drop=True)

    # 2 - split data into folds
    cv_folds = {}
    folds = sorted(set(metadata[fold_col]))
    for i in folds:
        mask_val = metadata[fold_col] == i
        fold_data = {
                "X_train": X.iloc[list(~mask_val), :],
                "y_train": list(y[~mask_val]),
                "metadata_train": metadata.iloc[list(~mask_val), :],
                "X_val": X.iloc[list(mask_val), :],
                "y_val": list(y[mask_val]),
                "metadata_val": metadata.iloc[list(mask_val), :]
        }
        cv_folds[i] = fold_data

    return cv_folds


def get_stratified_cv_folds(X: pd.DataFrame, y: np.array, n_splits: int, metadata: pd.DataFrame = None) -> \
        Dict[int, Dict[str, object]]:
    """

    Parameters
    ----------
    X - pd.DataFrame (m_samples, N_features)
    y - np.array (m_samples,)
    n_splits - cv splits, int
    metadata - Optional - pd.DataFrame (m_samples, T_features)

    Returns
    -------
    cv_folds - Dict in the following format: {
        1:   {
            "X_train": pd.DataFrame (n_samples, N_features),
            "y_train": list (n_samples,),
            "X_val": pd.DataFrame (k_samples, K_features),
            "y_val": list (k_samples,),
            "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional
            "metadata_val": list (k_samples,)    - Optional
            },
        2:  {
        }, ...
    }
    """
    # logger.debug(f"getting stratified {n_splits} cv folds")
    is_length_compatible = len(X) == len(y) if metadata is None else len(X) == len(y) == len(metadata)
    assert is_length_compatible, "X, y, and metadata are compatible in length"

    # 1 - data preparation
    # 1.1 - encode y labels
    lbl = LabelEncoder()
    lbl.fit(y)
    y_enc = lbl.transform(y)
    # 1.2 - reset df indexes
    X = X.reset_index(drop=True)
    if metadata is not None:
        metadata = metadata.reset_index(drop=True)

    # 2 - split data into folds
    cv_folds = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    # skf = StratifiedKFold(n_splits=val_set_n_splits, shuffle=True, random_state=val_set_seed)
    for i, (train_index, val_index) in enumerate(skf.split(X=np.array(X), y=y_enc)):
        fold_data = {
                "X_train": X.iloc[list(train_index), :],
                "y_train": list(y[train_index]),
                "X_val": X.iloc[list(val_index), :],
                "y_val": list(y[val_index])
        }
        if metadata is not None:
            fold_data.update({
                "metadata_train": metadata.iloc[list(train_index), :],
                "metadata_val": metadata.iloc[list(val_index), :]
            })

        cv_folds[i] = fold_data

    return cv_folds


def get_stratified_cv_folds_for_unique(unq_intr_data: pd.DataFrame, unq_y: np.array, n_splits: int, label_col: str,
                                       shuffle: bool = False, seed: int = None) -> Dict[int, Dict[str, object]]:
    """

    Parameters
    ----------
    unq_intr_data - pd.DataFrame (u_samples, [edge_index_0, edge_index_1])
    unq_y - np.array (u_samples,)
    n_splits - cv splits, int


    Returns
    -------
    cv_folds - Dict in the following format: {
        1:   {
            "unq_train": pd.DataFrame (t_samples, [edge_index_0, edge_index_1, edge_label])
            "unq_val": pd.DataFrame (v_samples, [edge_index_0, edge_index_1, edge_label])
            },
        2:  {
        }, ...
    }
    """
    # logger.debug(f"getting stratified {n_splits} cv folds")
    is_length_compatible = len(unq_y) == len(unq_intr_data)
    assert is_length_compatible, "unq_y and unq_intr_data are compatible in length"
    assert unq_intr_data.shape[1] == 2, "unq_intr_data format is pd.DataFrame (u_samples, [edge_index_0, edge_index_1])"

    # 1 - data preparation
    # 1.1 - encode y labels
    lbl = LabelEncoder()
    lbl.fit(unq_y)
    y_enc = lbl.transform(unq_y)
    # 1.2 - reset df indexes
    unq_intr_data = unq_intr_data.reset_index(drop=True)

    # 2 - split data into folds
    cv_folds = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    for i, (train_index, val_index) in enumerate(skf.split(X=np.array(unq_intr_data), y=y_enc)):
        # train
        unq_train = pd.DataFrame(unq_intr_data.iloc[list(train_index), :]).reset_index(drop=True)
        unq_train[label_col] = list(unq_y[train_index])
        # val
        unq_val = pd.DataFrame(unq_intr_data.iloc[list(val_index), :]).reset_index(drop=True)
        unq_val[label_col] = list(unq_y[val_index])

        fold_data = {
                "unq_train": unq_train,
                "unq_val": unq_val
        }
        cv_folds[i] = fold_data

    return cv_folds
