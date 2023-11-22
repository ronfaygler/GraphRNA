# Authors: Shani Cohen (ShaniCohen)
# Python version: 3.8
# Last update: 19.11.2023

import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
from typing import Dict, List
from models_handlers.model_handlers_utils import calc_binary_classification_metrics_using_y_score,\
    calc_binary_classification_metrics_using_prob_y_score_and_y_pred_thresholds, \
    get_stratified_cv_folds, get_predictions_df, split_cv_data
import logging
logger = logging.getLogger(__name__)


class XGBModelHandler(object):
    """
    Class to handle XGBoost model training and evaluation.
    xgboost.__version__ = '1.5.0'
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier

    Parameters
    ----------

    Attributes
    ----------
    Same as parameters
    """
    def __init__(self):
        super(XGBModelHandler, self).__init__()

    @staticmethod
    def get_const_hyperparams(**kwargs) -> Dict[str, object]:
        """   xgboost.__version__ = '1.5.0'  """
        logger.debug(f"getting const XGBoost hyperparams")
        if xgboost.__version__ != '1.5.0':
            logger.warning(f"This model handler supports xgboost version 1.5.0, but got version {xgboost.__version__}")
        const_model_hyperparams = {}
        return const_model_hyperparams

    @staticmethod
    def get_model_args(objective='binary:logistic', seed: int = 22, verbosity: int = 1, eval_metric='logloss',
                       early_stopping_rounds: int = 3, verbose: bool = False, **kwargs) -> Dict[str, object]:
        """   xgboost.__version__ = '1.5.0'  """
        logger.debug(f"getting XGBoost model_args")
        if xgboost.__version__ != '1.5.0':
            logger.warning(f"This model handler supports xgboost version 1.5.0, but got version {xgboost.__version__}")
        model_args = {
            # 1 - constructor args
            'objective': objective,
            'random_state': seed,
            'verbosity': verbosity,
            # 2 - fit args
            'eval_metric': eval_metric,
            'early_stopping_rounds': early_stopping_rounds,
            'verbose': verbose
        }
        return model_args

    @staticmethod
    def get_model_args(objective='binary:logistic', seed: int = 22, verbosity: int = 1, eval_metric='logloss',
                       early_stopping_rounds: int = 3, verbose: bool = False, **kwargs) -> Dict[str, object]:
        """   xgboost.__version__ = '1.5.0'  """
        logger.debug(f"getting XGBoost model_args")
        if xgboost.__version__ != '1.5.0':
            logger.warning(f"This model handler supports xgboost version 1.5.0, but got version {xgboost.__version__}")

        model_args = {
            # 1 - constructor args
            'objective': objective,  # the learning task and the corresponding learning objective or a custom objective function to be used.  optins: 'binary:hinge', 'binary:logistic', 'binary:logitraw'
            'random_state': seed,
            'verbosity': verbosity,
            # 2 - fit args
            'eval_metric': eval_metric,
            'early_stopping_rounds': early_stopping_rounds,
            'verbose': verbose
        }
        return model_args

    @staticmethod
    def eval_dataset(trained_model: XGBClassifier, X: pd.DataFrame, y: list, is_binary: bool = True,
                     num_classes: int = 2, dataset_nm: str = None, **kwargs) -> (Dict[str, object], np.array):
        """   xgboost.__version__ = '1.5.0'  """
        assert pd.isnull(X).sum().sum() == 0, "X contains Nan values"
        y_pred = trained_model.predict_proba(X=X)
        if is_binary:
            y_score = y_pred[:, 1]
            thresholds = kwargs.get('y_pred_thresholds_features', None)
            if thresholds is not None:
                scores = calc_binary_classification_metrics_using_prob_y_score_and_y_pred_thresholds(
                    y_true=y, y_score=y_score, y_pred_thresholds=thresholds, dataset_nm=dataset_nm)
            else:
                roc_max_fpr = kwargs.get('roc_max_fpr', None)
                scores = calc_binary_classification_metrics_using_y_score(y_true=y, y_score=y_score,
                                                                          roc_max_fpr=roc_max_fpr,
                                                                          dataset_nm=dataset_nm)
        else:
            y_score: np.array = None
            scores = None
        return scores, y_score

    @classmethod
    def predict_on_folds(cls, model_args: dict, cv_data: Dict[int, Dict[str, object]], **kwargs) -> \
            (Dict[int, Dict[str, Dict[str, float]]], Dict[int, pd.DataFrame], Dict[int, dict], Dict[int, dict]):
        """
        Returns
        -------
        cv_scores - Dict in the following format:  {
            <fold>: {
                'val': {
                    <score_nm>: score
                    }
        }
        cv_predictions_dfs - Dict in the following format:  {
            <fold>: pd.DataFrame
        }
        cv_training_history - Dict in the following format:  {
            <fold>: {
                'train': OrderedDict
                'validation': OrderedDict
            }
        }
        """
        logger.debug("predicting on folds")
        cv_scores = {}
        cv_training_history = {}
        cv_predictions_dfs = {}
        for fold, fold_data in cv_data.items():
            logger.debug(f"starting fold {fold}")
            scores, predictions, training_history, _ = \
                cls.train_and_test(X_train=fold_data['X_train'], y_train=fold_data['y_train'],
                                   X_test=fold_data['X_val'], y_test=fold_data['y_val'], model_args=model_args,
                                   metadata_train=fold_data['metadata_train'], **kwargs)
            # 2.1 - fold's val scores
            cv_scores[fold] = {'val': scores['test']}
            # 2.2 - fold's training history
            cv_training_history[fold] = training_history
            # 2.3 - fold's predictions df
            y_val_pred = predictions['test_pred']
            X_val = fold_data['X_val']
            y_val = fold_data['y_val']
            metadata_val = fold_data.get('metadata_val')
            cv_predictions_dfs[fold] = get_predictions_df(X=X_val, y_true=y_val, y_score=y_val_pred,
                                                          metadata=metadata_val)

        return cv_scores, cv_predictions_dfs, cv_training_history

    @classmethod
    def run_cross_validation_with_random_negatives(cls, model_args: dict, feature_cols: List[str], **kwargs) -> \
            Dict[str, Dict[str, object]]:
        """
        Returns
        -------
        cv_outs - Dict in the following format:  {
            'neg_syn': {
                'cv_scores': <cv_scores>,
                'cv_prediction_dfs': <cv_prediction_dfs>,
                'cv_training_history': <cv_training_history>
            },
            'neg_rnd': {
                'cv_scores': <cv_scores>,
                'cv_prediction_dfs': <cv_prediction_dfs>,
                'cv_training_history': <cv_training_history>
            }
        }

        Where:
        cv_scores - Dict in the following format:  {
            <fold>: {
                'val': {
                    <score_nm>: score
                    }
        }
        cv_prediction_dfs - Dict in the following format:  {
            <fold>: pd.DataFrame
        }
        cv_training_history - Dict in the following format:  {
            <fold>: {
                'train': OrderedDict
                'validation': OrderedDict
            }
        }
        cv_data - Dict in the following format:  {
            <fold>: {
                "X_train": pd.DataFrame (n_samples, N_features),
                "y_train": list (n_samples,),
                "X_val": pd.DataFrame (k_samples, K_features),
                "y_val": list (k_samples,),
                "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata is not None)
                "metadata_val": list (k_samples,)    - Optional (in case metadata is not None)
            }
        }
        """
        logger.debug(f"running additional cross validation")
        cv_outs = {}

        # 1 - get cv_data for synthetic negatives
        cv_train_neg_syn = kwargs['cv_train_neg_syn']
        if pd.notnull(cv_train_neg_syn):
            logger.debug(f"\n getting cv_data for synthetic negatives")
            df = cv_train_neg_syn
            label_col = 'interaction_label'
            n_splits = 10
            X = pd.DataFrame(df[feature_cols])
            y = np.array(df[label_col])
            meta_cols = [c for c in df.columns.values if c not in feature_cols + [label_col]]
            meta = pd.DataFrame(df[meta_cols])
            cv_data_neg_syn = get_stratified_cv_folds(X=X, y=y, n_splits=n_splits, metadata=meta)
            cv_scores, cv_predictions_dfs, cv_training_history = \
                cls.predict_on_folds(model_args=model_args, cv_data=cv_data_neg_syn, **kwargs)
            cv_outs['neg_syn'] = {
                "cv_scores": cv_scores,
                "cv_predictions_dfs": cv_predictions_dfs,
                "cv_training_history": cv_training_history,
            }
        else:
            cv_outs['neg_syn'] = None

        # 2 - get cv_data for random negatives
        logger.debug(f"\n getting cv_data for random negatives")
        df = kwargs['cv_train_neg_rnd']
        label_col = 'y_true'
        fold_col = "fold"
        meta_cols_to_rem = ['y_score', 'COPRA_sRNA_is_missing', 'COPRA_sRNA', 'COPRA_mRNA', 'COPRA_mRNA_locus_tag',
                            'COPRA_pv', 'COPRA_fdr', 'COPRA_NC_000913', 'COPRA_mRNA_not_in_output',
                            'COPRA_validated_pv', 'COPRA_validated_score']
        X = pd.DataFrame(df[feature_cols])
        y = np.array(df[label_col])
        meta_cols = [c for c in df.columns.values if c not in feature_cols + [label_col] + meta_cols_to_rem]
        meta = pd.DataFrame(df[meta_cols])
        cv_data_neg_rnd = split_cv_data(X=X, y=y, metadata=meta, fold_col=fold_col)
        cv_scores, cv_predictions_dfs, cv_training_history = \
            cls.predict_on_folds(model_args=model_args, cv_data=cv_data_neg_rnd, **kwargs)
        cv_outs['neg_rnd'] = {
            "cv_scores": cv_scores,
            "cv_predictions_dfs": cv_predictions_dfs,
            "cv_training_history": cv_training_history,
        }

        return cv_outs

    @classmethod
    def run_cross_validation(cls, X: pd.DataFrame, y: List[int], n_splits: int, model_args: dict,
                             metadata: pd.DataFrame = None, **kwargs) -> \
            (Dict[int, Dict[str, Dict[str, float]]], Dict[int, pd.DataFrame], Dict[int, dict], Dict[int, dict]):
        """
        Returns
        -------

        cv_scores - Dict in the following format:  {
            <fold>: {
                'val': {
                    <score_nm>: score
                    }
        }
        cv_predictions_dfs - Dict in the following format:  {
            <fold>: pd.DataFrame
        }
        cv_training_history - Dict in the following format:  {
            <fold>: {
                'train': OrderedDict
                'validation': OrderedDict
            }
        }
        cv_data - Dict in the following format:  {
            <fold>: {
                "X_train": pd.DataFrame (n_samples, N_features),
                "y_train": list (n_samples,),
                "X_val": pd.DataFrame (k_samples, K_features),
                "y_val": list (k_samples,),
                "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata is not None)
                "metadata_val": list (k_samples,)    - Optional (in case metadata is not None)
            }
        }
        """
        # 1 - split dataset into folds
        cv_data = get_stratified_cv_folds(X=X, y=np.array(y), n_splits=n_splits, metadata=metadata)
        # 2 - predict on folds
        cv_scores = {}
        cv_training_history = {}
        cv_predictions_dfs = {}
        for fold, fold_data in cv_data.items():
            logger.debug(f"starting fold {fold}")
            scores, predictions, training_history, _ = \
                cls.train_and_test(X_train=fold_data['X_train'], y_train=fold_data['y_train'],
                                   X_test=fold_data['X_val'], y_test=fold_data['y_val'], model_args=model_args,
                                   metadata_train=fold_data['metadata_train'], **kwargs)
            # 2.1 - fold's val scores
            cv_scores[fold] = {'val': scores['test']}
            # 2.2 - fold's training history
            cv_training_history[fold] = training_history
            # 2.3 - fold's predictions df
            y_val_pred = predictions['test_pred']
            X_val = fold_data['X_val']
            y_val = fold_data['y_val']
            metadata_val = fold_data.get('metadata_val')
            cv_predictions_dfs[fold] = get_predictions_df(X=X_val, y_true=y_val, y_score=y_val_pred,
                                                          metadata=metadata_val)
            
        return cv_scores, cv_predictions_dfs, cv_training_history, cv_data

    @classmethod
    def train_and_test(cls, X_train: pd.DataFrame, y_train: List[int], X_test: pd.DataFrame, y_test: List[int],
                       model_args: dict, metadata_train: pd.DataFrame = None, metadata_test: pd.DataFrame = None,
                       **kwargs) -> (Dict[str, dict], Dict[str, object], Dict[str, object], Dict[str, object]):
        """
        xgboost.__version__ = '1.5.0'

        Parameters
        ----------
        X_train: pd.DataFrame (n_samples, N_features),
        y_train: list (n_samples,),
        X_test: pd.DataFrame (t_samples, N_features),
        y_test: list (t_samples,)
        model_args: Dict of model's constructor and fit() arguments
        metadata_train: Optional - pd.DataFrame (n_samples, T_features),
        metadata_test: Optional - pd.DataFrame (t_samples, T_features)
        kwargs

        Returns
        -------

        scores - Dict in the following format:  {
            'test': {
                <score_nm>: score
                }
        }
        predictions - Dict in the following format:  {
            'test_pred': array-like (t_samples,)
        }
        training_history - Dict in the following format:  {
            'train': OrderedDict
            'validation': OrderedDict
        }
        train_val_data - Dict in the following format:  {
            "X_train": pd.DataFrame (n_samples, N_features),
            "y_train": list (n_samples,),
            "X_val": pd.DataFrame (k_samples, K_features),
            "y_val": list (k_samples,),
            "metadata_train": pd.DataFrame (n_samples, T_features),    - Optional (in case metadata_train is not None)
            "metadata_val": list (k_samples,)    - Optional (in case metadata_train is not None)
        }
        """
        logger.debug("training an XGBoost classifier")
        if xgboost.__version__ != '1.5.0':
            logger.warning(f"This model handler supports xgboost version 1.5.0, but got version {xgboost.__version__}")

        # 1 - construct
        model = XGBClassifier(
            objective=model_args['objective'],
            random_state=model_args['random_state'],
            use_label_encoder=False,
            verbosity=model_args['verbosity']
        )
        # 2 - define fit() params
        # 2.1 - split train into training and validation
        val_size = 0.1
        folds_data = get_stratified_cv_folds(X=X_train, y=np.array(y_train), n_splits=int(1/val_size),
                                             metadata=metadata_train)
        train_val_data = folds_data[0]  # randomly select a fold
        del folds_data
        eval_set = [(np.array(train_val_data['X_train']), train_val_data['y_train']),
                    (np.array(train_val_data['X_val']), train_val_data['y_val'])]
        # 2.2 - extract additional fit() params form config
        eval_metric = model_args['eval_metric']
        early_stopping_rounds = model_args['early_stopping_rounds']  # if not None - early stopping is activated
        verbose = model_args['verbose']  # if not None - early stopping is activated

        # 3 - train
        trained_model = model.fit(X=np.array(train_val_data['X_train']), y=train_val_data['y_train'], eval_set=eval_set,
                                  eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        training_history = {
            "train": trained_model.evals_result()['validation_0'],
            "validation": trained_model.evals_result()['validation_1']
        }

        # 4 - calc scores
        predictions, scores = {}, {}
        num_classes = len(np.unique(y_train))
        is_binary = sorted(np.unique(y_train)) == [0, 1]
        # 4.1 - test
        logger.debug("evaluating test set")
        test_scores, test_y_pred = cls.eval_dataset(trained_model=trained_model, X=X_test, y=y_test,
                                                    is_binary=is_binary, num_classes=num_classes, **kwargs)
        scores.update({'test': test_scores})
        predictions.update({'test_pred': test_y_pred})

        return scores, predictions, training_history, train_val_data
