# Authors: Shani Cohen (ShaniCohen)
# Python version: 3.8
# Last update: 19.11.2023

import pandas as pd
import numpy as np
import os
import random
import itertools
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List
from models.graph_rna import GraphRNA
import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

import torch_geometric.transforms as T
from models_handlers.model_handlers_utils import calc_binary_classification_metrics_using_y_score, \
    calc_binary_classification_metrics_using_prob_y_score_and_y_pred_thresholds, get_stratified_cv_folds_for_unique
from utils import order_df, split_df_samples
import logging
logger = logging.getLogger(__name__)


class HGAModelHandler(object):
    """
    Class to ...
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

    Parameters
    ----------

    Attributes
    ----------
    Same as parameters
    """
    # ------  Nodes  ------
    nodes_are_defined = False
    ##  mRNA
    mrna_nodes = None  # init in _prepare_data
    mrna_nid_col = 'mrna_node_id'
    mrna_eco_acc_col = None
    mrna = 'mrna'

    ##  sRNA
    srna_nodes = None  # init in _prepare_data
    srna_nid_col = 'srna_node_id'
    srna_eco_acc_col = None
    srna = 'srna'

    # ------  Edges  ------
    # ---  interactions (sRNA - mRNA)
    srna_mrna_val_col = None  # in case additional edge features are requested
    srna_to_mrna = 'targets'  # ("rates")
    binary_intr_label_col = 'interaction_label'

    # ------  Params  ------
    # --- random dataset split
    random_split = False
    # Generate fixed negative edges (in val_data) for evaluation with a ratio of x:1
    val_and_test_neg_sampling_ratio = 1.0

    # --- train data
    # for train data, sample negative edges with a ratio of x:1
    cv_neg_sampling_ratio_data = 1.0
    train_neg_sampling_ratio_data = 1.0
    sampling_seed = 20
    # Across the training edges, we use 70% of edges for message passing, and 30% of edges for supervision.
    train_supervision_ratio = 0.3

    # --- train loader
    train_w_loader = False
    # in the first hop, sample at most 20 neighbors
    # in the second hop, sample at most 10 neighbors
    train_num_neighbors = [20, 10]
    # during training, sample negative edges on-the-fly with a ratio of x:1
    train_neg_sampling_ratio_loader = 2.0
    train_batch_size = 128
    # set to True to have the data reshuffled at every epoch
    train_shuffle = True

    # --- test data
    eval_num_neighbors = [20, 0]
    eval_batch_size = 128
    eval_shuffle = False

    shuffle_test = True
    debug_logs = False

    def __init__(self, seed: int = 100):
        super(HGAModelHandler, self).__init__()
        self.seed = seed
        # set random seeds
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # # When running on the CuDNN backend, two further options must be set
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def get_const_hyperparams(**kwargs) -> Dict[str, object]:
        """   torch_geometric.__version__ = '2.1.0'  """
        logger.debug(f"getting const HeteroGraph Advanced hyperparams")
        const_model_hyperparams = {
        }
        return const_model_hyperparams

    @staticmethod
    def get_model_args(**kwargs) -> Dict[str, object]:
        """   torch_geometric.__version__ = '2.1.0'  """
        logger.debug("getting HeteroGraph Advanced model_args")
        model_args = {
            # 1 - constructor args
            'hidden_channels': 64,
            'num_sim_feat': 50,
            # 2 - fit (train) args
            'learning_rate': 0.001,
            'epochs': 25
        }
        return model_args

    @staticmethod
    def log_df_stats(df: pd.DataFrame, label_col: str, df_nm: str = None):
        df_nm = 'df' if pd.isnull(df_nm) else df_nm
        _pos_df = sum(df[label_col])
        logger.debug(f' {df_nm}: {len(df)} interactions (P: {_pos_df}, N: {len(df) - _pos_df})')
        return

    @staticmethod
    def log_df_rna_eco(rna_name: str, rna_eco_df: pd.DataFrame, acc_col: str):
        logger.debug(f' {rna_name} data from EcoCyc: {len(set(rna_eco_df[acc_col]))} unique {rna_name}s of E.coli')
        return

    @staticmethod
    def _map_rna_nodes_and_edges(out_rna_node_id_col: str, rna_eco: pd.DataFrame, e_acc_col: str, s_acc_1_col: str,
                                 s_acc_2_col: str, s_score_col: str, rna_sim: pd.DataFrame = None) -> \
            (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """ map accession to node id """
        # 1 - Nodes = EcoCyc data
        rna_nodes = rna_eco
        # 1.1 - map from original id (acc) to node id (consecutive values)
        rna_nodes[out_rna_node_id_col] = np.arange(len(rna_nodes))
        rna_nodes = order_df(df=rna_nodes, first_cols=[out_rna_node_id_col])
        rna_map = rna_eco[[out_rna_node_id_col, e_acc_col]]
        # 2 - Edges = similarity data
        if type(rna_sim) is pd.DataFrame:
            _len = len(rna_sim)
            rna_edges = pd.merge(rna_sim, rna_map, left_on=s_acc_1_col, right_on=e_acc_col, how='left').rename(
                columns={out_rna_node_id_col: f'{out_rna_node_id_col}_1'})
            rna_edges = pd.merge(rna_edges, rna_map, left_on=s_acc_2_col, right_on=e_acc_col, how='left').rename(
                columns={out_rna_node_id_col: f'{out_rna_node_id_col}_2'})
            assert len(rna_edges) == _len, "duplications post merge"
            _cols = [f'{out_rna_node_id_col}_1', f'{out_rna_node_id_col}_2', s_score_col, s_acc_1_col, s_acc_2_col]
            rna_edges = rna_edges[_cols]
        else:
            rna_edges = None

        return rna_map, rna_nodes, rna_edges

    @staticmethod
    def _pos_neg_split(df: pd.DataFrame, binary_label_col: str) -> (pd.DataFrame, pd.DataFrame):
        df_pos = df[df[binary_label_col] == 1].reset_index(drop=True)
        df_neg = df[df[binary_label_col] == 0].reset_index(drop=True)
        return df_pos, df_neg

    @staticmethod
    def _remove_synthetic_samples(X: pd.DataFrame, y: List[int], metadata: pd.DataFrame, is_syn_col: str) -> \
            (pd.DataFrame, List[int], pd.DataFrame):
        '''

        Parameters
        ----------
        X: pd.DataFrame (n_samples, N_features)
        y: list (n_samples,)
        metadata: pd.DataFrame (n_samples, T_features)
        is_syn_col: is synthetic indicator columns in metadata df

        Returns
        -------

        '''
        logger.debug("removing synthetic samples from train")
        assert len(X) == len(y) == len(metadata), "X, y, metadata mismatch"
        assert sorted(set(y)) in [[0, 1], [1]], "y is not binary"

        mask_not_synthetic = ~metadata[is_syn_col].reset_index(drop=True)
        metadata = metadata.reset_index(drop=True)[mask_not_synthetic]
        X = X.reset_index(drop=True)[mask_not_synthetic]
        y = list(pd.Series(y)[mask_not_synthetic])
        logger.debug(f"removed {len(mask_not_synthetic) - len(X)} synthetic samples from X "
                     f"(before: {len(mask_not_synthetic)}, after: {len(X)})")
        logger.debug(f"removed {len(mask_not_synthetic) - len(metadata)} synthetic samples from metadata "
                     f"(before: {len(mask_not_synthetic)}, after: {len(metadata)})")
        return X, y, metadata

    @classmethod
    def _add_neg_samples(cls, unq_intr_pos: pd.DataFrame, ratio: float, _shuffle: bool = False) -> HeteroData:
        assert sum(unq_intr_pos[cls.binary_intr_label_col]) == len(unq_intr_pos), "unq_intr_pos has negatives"
        _pos = list(zip(list(unq_intr_pos[cls.srna_nid_col]), list(unq_intr_pos[cls.mrna_nid_col])))
        _all = list(itertools.product(list(cls.srna_nodes[cls.srna_nid_col]), list(cls.mrna_nodes[cls.mrna_nid_col])))
        _unknown = pd.Series(list(set(_all) - set(_pos)))
        _unknown_df = pd.DataFrame({
            cls.binary_intr_label_col: 0,
            cls.srna_nid_col: _unknown.apply(lambda x: x[0]),
            cls.mrna_nid_col: _unknown.apply(lambda x: x[1])
        })

        n = max(int(len(_pos) * ratio), 1)
        _neg_samples = _unknown_df.sample(n=n, random_state=cls.sampling_seed)
        out = pd.concat(objs=[unq_intr_pos, _neg_samples], axis=0, ignore_index=True).reset_index(drop=True)
        if _shuffle:
            out = pd.DataFrame(shuffle(out)).reset_index(drop=True)

        return out

    @classmethod
    def _get_unique_inter(cls, metadata: pd.DataFrame, y: List[int], srna_acc_col: str, mrna_acc_col: str,
                          df_nm: str = None) -> pd.DataFrame:
        # 1 - data validation
        _len = len(metadata)
        srna_acc = metadata[srna_acc_col]
        mrna_acc = metadata[mrna_acc_col]
        assert sorted(set(y)) in [[0, 1], [1]], "y is not binary"
        assert sum(pd.isnull(srna_acc)) + sum(pd.isnull(mrna_acc)) == 0, "some acc id are null"
        # 2 - get unique sRNA-mRNA interactions
        unq_intr = pd.DataFrame({
            srna_acc_col: metadata[srna_acc_col],
            mrna_acc_col: metadata[mrna_acc_col],
            cls.binary_intr_label_col: y
        })
        cls.log_df_stats(df=unq_intr, label_col=cls.binary_intr_label_col, df_nm=df_nm)
        unq_intr = unq_intr.drop_duplicates().reset_index(drop=True)
        # 2 - log unique
        cls.log_df_stats(df=unq_intr, label_col=cls.binary_intr_label_col, df_nm=f"unique_{df_nm}")

        return unq_intr

    @classmethod
    def _assert_no_data_leakage(cls, unq_train: pd.DataFrame, unq_test: pd.DataFrame, srna_acc_col: str,
                                mrna_acc_col: str):
        logger.debug("assert no data leakage")
        train_tup = set(zip(unq_train[srna_acc_col], unq_train[mrna_acc_col], unq_train[cls.binary_intr_label_col]))
        test_tup = set(zip(unq_test[srna_acc_col], unq_test[mrna_acc_col], unq_test[cls.binary_intr_label_col]))
        dupl = sorted(train_tup - (train_tup - test_tup))
        assert len(dupl) == 0, f"{len(dupl)} duplicated interactions in train and test"
        return

    @classmethod
    def _remove_train_inter_from_test_all_pairs(cls, unq_train: pd.DataFrame, unq_test: pd.DataFrame, srna_acc_col: str,
                                                mrna_acc_col: str) -> pd.DataFrame:
        logger.debug("remove train inter_from_test_all_pairs")
        train_tup = set(zip(unq_train[srna_acc_col], unq_train[mrna_acc_col]))
        test_tup = set(zip(unq_test[srna_acc_col], unq_test[mrna_acc_col]))
        test_no_dup = pd.DataFrame(list(test_tup - train_tup), columns=[srna_acc_col, mrna_acc_col])
        logger.debug(f"from {len(test_tup)} interactions {len(test_tup) - len(test_no_dup)} were filtered since appear "
                     f"in train --> {len(test_no_dup)} left")
        return test_no_dup

    @classmethod
    def _map_inter(cls, intr: pd.DataFrame, mrna_acc_col: str, srna_acc_col: str, mrna_map: pd.DataFrame,
                   m_map_acc_col: str, srna_map: pd.DataFrame, s_map_acc_col: str) -> pd.DataFrame:
        _len, _cols = len(intr), list(intr.columns.values)
        intr = pd.merge(intr, mrna_map, left_on=mrna_acc_col, right_on=m_map_acc_col, how='left')
        intr = pd.merge(intr, srna_map, left_on=srna_acc_col, right_on=s_map_acc_col, how='left')
        assert len(intr) == _len, "duplications post merge"
        intr = intr[[cls.srna_nid_col, cls.mrna_nid_col] + _cols]
        return intr

    @classmethod
    def _define_nodes_and_edges(cls, **kwargs):
        # 1 - mRNA
        # 1.1 - get map, nodes and edges
        m_eco_acc_col = kwargs['me_acc_col']
        m_sim_score_col = kwargs['ms_score_col']
        cls.log_df_rna_eco(rna_name='mRNA', rna_eco_df=kwargs['mrna_eco'], acc_col=m_eco_acc_col)
        _, mrna_nodes, _ = \
            cls._map_rna_nodes_and_edges(out_rna_node_id_col=cls.mrna_nid_col, rna_eco=kwargs['mrna_eco'],
                                         e_acc_col=m_eco_acc_col, s_acc_1_col=kwargs['ms_acc_1_col'],
                                         s_acc_2_col=kwargs['ms_acc_2_col'], s_score_col=m_sim_score_col)
        # 1.2 - set
        # nodes
        cls.mrna_nodes = mrna_nodes
        cls.mrna_eco_acc_col = m_eco_acc_col

        # 2 - sRNA
        # 2.1 - get map, nodes and edges
        s_eco_acc_col = kwargs['se_acc_col']
        s_sim_score_col = kwargs['ss_score_col']
        cls.log_df_rna_eco(rna_name='sRNA', rna_eco_df=kwargs['srna_eco'], acc_col=s_eco_acc_col)
        _, srna_nodes, _ = \
            cls._map_rna_nodes_and_edges(out_rna_node_id_col=cls.srna_nid_col, rna_eco=kwargs['srna_eco'],
                                         e_acc_col=s_eco_acc_col, s_acc_1_col=kwargs['ss_acc_1_col'],
                                         s_acc_2_col=kwargs['ss_acc_2_col'], s_score_col=s_sim_score_col)
        # 2.2 - set
        # nodes
        cls.srna_nodes = srna_nodes
        cls.srna_eco_acc_col = s_eco_acc_col
        # indicator
        cls.nodes_are_defined = True

        return

    @classmethod
    def _map_interactions_to_edges(cls, unique_intr: pd.DataFrame, srna_acc_col: str, mrna_acc_col: str) -> \
            (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        logger.debug("mapping interactions to edges")
        mrna_map = cls.mrna_nodes[[cls.mrna_nid_col, cls.mrna_eco_acc_col]]
        srna_map = cls.srna_nodes[[cls.srna_nid_col, cls.srna_eco_acc_col]]

        unique_intr = cls._map_inter(intr=unique_intr, mrna_acc_col=mrna_acc_col, srna_acc_col=srna_acc_col,
                                     mrna_map=mrna_map, m_map_acc_col=cls.mrna_eco_acc_col, srna_map=srna_map,
                                     s_map_acc_col=cls.srna_eco_acc_col)
        unique_intr = unique_intr.sort_values(by=[cls.srna_nid_col, cls.mrna_nid_col]).reset_index(drop=True)

        return unique_intr

    @classmethod
    def _generate_train_and_test(cls, edges: dict) -> (HeteroData, HeteroData):
        # ------  Train Data  ------
        train_data = HeteroData()
        # ------  Nodes
        train_data[cls.srna].node_id = torch.arange(len(cls.srna_nodes))
        train_data[cls.mrna].node_id = torch.arange(len(cls.mrna_nodes))
        train_data[cls.mrna].x = None
        # ------  Edges
        # edges for message passing
        train_edge_index = torch.stack([torch.from_numpy(np.array(edges['train']['message_passing']['label_index_0'])),
                                       torch.from_numpy(np.array(edges['train']['message_passing']['label_index_1']))],
                                       dim=0)
        train_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_index = train_edge_index
        train_data = T.ToUndirected()(train_data)
        # edges for supervision
        train_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label = \
            torch.from_numpy(np.array(edges['train']['supervision']['label'])).float()
        train_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label_index = \
            torch.stack([torch.from_numpy(np.array(edges['train']['supervision']['label_index_0'])),
            torch.from_numpy(np.array(edges['train']['supervision']['label_index_1']))], dim=0)

        # ------  Test Data  ------
        test_data = HeteroData()
        # ------  Nodes
        test_data[cls.srna].node_id = torch.arange(len(cls.srna_nodes))
        test_data[cls.mrna].node_id = torch.arange(len(cls.mrna_nodes))
        test_data[cls.mrna].x = None
        # ------  Edges
        # all train edges for message passing
        test_edge_index = torch.stack([torch.from_numpy(np.array(edges['train']['all']['label_index_0'])),
                                       torch.from_numpy(np.array(edges['train']['all']['label_index_1']))], dim=0)
        test_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_index = test_edge_index
        test_data = T.ToUndirected()(test_data)
        # test edges for supervision
        test_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label = \
            torch.from_numpy(np.array(edges['test']['label'])).float()
        test_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label_index = \
            torch.stack([torch.from_numpy(np.array(edges['test']['label_index_0'])),
            torch.from_numpy(np.array(edges['test']['label_index_1']))], dim=0)
        if cls.debug_logs:
            logger.debug(f"\n Train data:\n ============== \n{train_data}\n"
                         f"\n Test data:\n ============== \n{test_data}\n")

        return train_data, test_data

    @classmethod
    def _init_train_test_hetero_data(cls, unq_train: pd.DataFrame, unq_test: pd.DataFrame, train_neg_sampling: bool) \
            -> (HeteroData, HeteroData):
        """

        Parameters
        ----------
        unq_train
        unq_test
        train_neg_sampling

        Returns
        -------

        """
        logger.debug("initializing train and test hetero data")
        df = unq_train
        unq_train_pos, unq_train_neg = cls._pos_neg_split(df=unq_train, binary_label_col=cls.binary_intr_label_col)
        # 1 - random negative sampling - train
        if train_neg_sampling and not cls.train_w_loader:
            # 1.1 - take only positive samples from train and add random negatives
            _shuffle_train = True
            df = cls._add_neg_samples(unq_intr_pos=unq_train_pos, ratio=cls.train_neg_sampling_ratio_data,
                                      _shuffle=_shuffle_train)
            _shuffle_test = True
            if _shuffle_test:
                unq_test = pd.DataFrame(shuffle(unq_test)).reset_index(drop=True)
            if cls.debug_logs:
                logger.debug(f"_shuffle_train = {_shuffle_train}, _shuffle_test = {_shuffle_test}")

        # 2 - split train edges into message passing & supervision
        unq_train_spr, unq_train_mp = split_df_samples(df=df, ratio=cls.train_supervision_ratio)
        edges = {
            'train': {
                'all': {
                    'label': list(df[cls.binary_intr_label_col]),
                    'label_index_0': list(df[cls.srna_nid_col]),
                    'label_index_1': list(df[cls.mrna_nid_col])
                },
                'message_passing': {
                    'label': list(unq_train_mp[cls.binary_intr_label_col]),
                    'label_index_0': list(unq_train_mp[cls.srna_nid_col]),
                    'label_index_1': list(unq_train_mp[cls.mrna_nid_col])
                },
                'supervision': {
                    'label': list(unq_train_spr[cls.binary_intr_label_col]),
                    'label_index_0': list(unq_train_spr[cls.srna_nid_col]),
                    'label_index_1': list(unq_train_spr[cls.mrna_nid_col])
                }
            },
            'test': {
                'label': list(unq_test[cls.binary_intr_label_col]),
                'label_index_0': list(unq_test[cls.srna_nid_col]),
                'label_index_1': list(unq_test[cls.mrna_nid_col])
            }
        }

        logger.debug(f"\n{len(cls.srna_nodes)} sRNA nodes, {len(cls.mrna_nodes)} mRNA nodes \n"
                     f"Train: {len(edges['train']['all']['label'])} interactions, "
                     f"P: {sum(edges['train']['all']['label'])}, "
                     f"N: {len(edges['train']['all']['label']) - sum(edges['train']['all']['label'])} \n"
                     f"Test: {len(edges['test']['label'])} interactions, "
                     f"P: {sum(edges['test']['label'])}, "
                     f"N: {len(edges['test']['label']) - sum(edges['test']['label'])}")

        # 3 - initialize data sets
        train_data, test_data = cls._generate_train_and_test(edges=edges)

        return train_data, test_data

    @classmethod
    def _get_train_mini_batches_loader(cls, train_data: HeteroData) -> LinkNeighborLoader:
        # Define seed edges:
        edge_label_index = train_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label_index
        edge_label = train_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label
        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=cls.train_num_neighbors,
            neg_sampling_ratio=cls.train_neg_sampling_ratio_loader,
            edge_label_index=((cls.srna, cls.srna_to_mrna, cls.mrna), edge_label_index),
            edge_label=edge_label,
            batch_size=cls.train_batch_size,
            shuffle=cls.train_shuffle,
        )
        for sampled_data in tqdm.tqdm(train_loader):
            s_edge_label_index = sampled_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label_index
            s_edge_label = sampled_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label

        return train_loader

    @classmethod
    def _train_hgnn_with_loader(cls, hg_model: GraphRNA, train_data: HeteroData, model_args: dict) -> GraphRNA:
        logger.debug(f"training HeteroGraph Advanced model with loader")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if cls.debug_logs:
            logger.debug(f"Device: '{device}'")

        train_loader = cls._get_train_mini_batches_loader(train_data=train_data)
        hg_model = hg_model.to(device)
        optimizer = torch.optim.Adam(hg_model.parameters(), lr=model_args['learning_rate'])

        for epoch in range(1, model_args['epochs']):
            total_loss = total_examples = 0
            for sampled_data in tqdm.tqdm(train_loader):
                optimizer.zero_grad()
                sampled_data.to(device)
                pred = hg_model(sampled_data)
                ground_truth = sampled_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
            if cls.debug_logs:
                logger.debug(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

        return hg_model

    @classmethod
    def _train_hgnn(cls, hga_model: GraphRNA, train_data: HeteroData, model_args: dict) -> GraphRNA:
        logger.debug(f"training HeteroGraph Advanced model (without loader) -> epochs = {model_args['epochs']}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if cls.debug_logs:
            logger.debug(f"Device: '{device}'")

        hga_model = hga_model.to(device)
        optimizer = torch.optim.Adam(hga_model.parameters(), lr=model_args['learning_rate'], weight_decay=0.001)

        for epoch in range(1, model_args['epochs']):
            total_loss = total_examples = 0
            optimizer.zero_grad()
            train_data.to(device)
            pred = hga_model(train_data, model_args)
            ground_truth = train_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            if cls.debug_logs:
                logger.debug(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

        return hga_model

    @classmethod
    def _get_eval_mini_batches_loader(cls, eval_data: HeteroData) -> LinkNeighborLoader:
        # Define the validation seed edges:
        edge_label_index = eval_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label_index
        edge_label = eval_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label

        # Define hyper-params
        bs_const = cls.val_and_test_neg_sampling_ratio + 1 if cls.random_split else 1

        eval_loader = LinkNeighborLoader(
            data=eval_data,
            num_neighbors=cls.eval_num_neighbors,
            edge_label_index=((cls.srna, cls.srna_to_mrna, cls.mrna), edge_label_index),
            edge_label=edge_label,
            batch_size=int(bs_const * cls.eval_batch_size),
            shuffle=cls.eval_shuffle,
        )

        sampled_data = next(iter(eval_loader))
        i1 = sampled_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_index[0]
        i2 = sampled_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_index[1]
        lbl = sampled_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label
        test_df = pd.DataFrame({
            'edge_label_index_0': torch.cat(i1, dim=0).cpu().numpy(),
            'edge_label_index_1': torch.cat(i2, dim=0).cpu().numpy(),
            'edge_label_': torch.cat(lbl, dim=0).cpu().numpy()
        })

        return eval_loader

    @classmethod
    def _eval_dataset_w_loader(cls, trained_model: GraphRNA, dataset_loader: LinkNeighborLoader,
                               dataset_nm: str = None) -> (Dict[str, object], pd.DataFrame):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.debug(f"Device: '{device}'")
        # 1 - predict on dataset
        preds = []
        ground_truths = []
        srna_nids, mrna_nids = [], []
        for sampled_data in tqdm.tqdm(dataset_loader):
            with torch.no_grad():
                sampled_data.to(device)
                preds.append(trained_model(sampled_data))
                ground_truths.append(sampled_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label)
                srna_nids.append(sampled_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label_index[0])
                mrna_nids.append(sampled_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label_index[1])
        # 2 - process outputs
        scores_arr = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        y_true = list(ground_truth)
        # 2.1 - scaling scores_arr to [0 ,1]
        minmax_scaler = MinMaxScaler()
        y_score = minmax_scaler.fit_transform(scores_arr.reshape(-1, 1)).reshape(-1)
        # 2.2 - save
        eval_data_preds = pd.DataFrame({
            cls.srna_nid_col: torch.cat(srna_nids, dim=0).cpu().numpy(),
            cls.mrna_nid_col: torch.cat(mrna_nids, dim=0).cpu().numpy(),
            'y_original_score': list(scores_arr),
            'y_score': y_score,
            'y_true': y_true
        })
        eval_data_preds = eval_data_preds.sort_values(by=[cls.srna_nid_col, cls.mrna_nid_col])

        # 3 - calc evaluation metrics
        scores = calc_binary_classification_metrics_using_y_score(y_true=y_true, y_score=y_score, dataset_nm=dataset_nm)
        if cls.debug_logs:
            for k, v in scores.items():
                print(f"{k}: {v}")

        return scores, eval_data_preds

    @classmethod
    def _eval_hgnn_w_loader(cls, trained_model: GraphRNA, eval_data: HeteroData, dataset_nm: str = None) -> \
            (Dict[str, object], pd.DataFrame):
        eval_loader = cls._get_eval_mini_batches_loader(eval_data=eval_data)
        scores, eval_data_preds = cls._eval_dataset_w_loader(trained_model=trained_model, dataset_loader=eval_loader,
                                                             dataset_nm=dataset_nm)
        return scores, eval_data_preds

    @classmethod
    def _predict_hgnn(cls, trained_model: GraphRNA, eval_data: HeteroData, model_args: dict = None) -> pd.DataFrame:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if cls.debug_logs:
            logger.debug(f"Device: '{device}'")
        # 1 - predict on dataset
        preds = []
        # ground_truths = []
        srna_nids, mrna_nids = [], []
        with torch.no_grad():
            eval_data.to(device)
            preds.append(trained_model(eval_data, model_args))
            srna_nids.append(eval_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label_index[0])
            mrna_nids.append(eval_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label_index[1])
        # 2 - process outputs
        scores_arr = torch.cat(preds, dim=0).cpu().numpy()
        # 2.1 - scaling scores_arr to [0 ,1]
        minmax_scaler = MinMaxScaler()
        y_score = minmax_scaler.fit_transform(scores_arr.reshape(-1, 1)).reshape(-1)
        # 2.2 - save
        eval_data_preds = pd.DataFrame({
            cls.srna_nid_col: torch.cat(srna_nids, dim=0).cpu().numpy(),
            cls.mrna_nid_col: torch.cat(mrna_nids, dim=0).cpu().numpy(),
            'y_original_score': list(scores_arr),
            'y_score': y_score
        })
        eval_data_preds = eval_data_preds.sort_values(by='y_score', ascending=False).reset_index(drop=True)

        return eval_data_preds

    @classmethod
    def _eval_hgnn(cls, trained_model: GraphRNA, eval_data: HeteroData, model_args: dict = None,
                   dataset_nm: str = None, **kwargs) -> (Dict[str, object], pd.DataFrame):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if cls.debug_logs:
            logger.debug(f"Device: '{device}'")
        # 1 - predict on dataset
        preds = []
        ground_truths = []
        srna_nids, mrna_nids = [], []
        with torch.no_grad():
            eval_data.to(device)
            preds.append(trained_model(eval_data, model_args))

            ground_truths.append(eval_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label)
            srna_nids.append(eval_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label_index[0])
            mrna_nids.append(eval_data[cls.srna, cls.srna_to_mrna, cls.mrna].edge_label_index[1])
        # 2 - process outputs
        scores_arr = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        y_true = list(ground_truth)
        # 2.1 - scaling scores_arr to [0 ,1]
        minmax_scaler = MinMaxScaler()
        y_score = minmax_scaler.fit_transform(scores_arr.reshape(-1, 1)).reshape(-1)
        y_original_score = list(scores_arr)
        # 2.2 - save
        eval_data_preds = pd.DataFrame({
            cls.srna_nid_col: torch.cat(srna_nids, dim=0).cpu().numpy(),
            cls.mrna_nid_col: torch.cat(mrna_nids, dim=0).cpu().numpy(),
            'y_original_score': y_original_score,
            'y_score': y_score,
            'y_true': y_true
        })
        roc_max_fpr = kwargs.get('roc_max_fpr', None)

        if kwargs.get('use_original_score'):
            eval_data_preds = eval_data_preds.sort_values(by='y_original_score', ascending=False).reset_index(drop=True)
            # 3 - calc evaluation metrics
            scores = calc_binary_classification_metrics_using_y_score(y_true=y_true, y_score=y_original_score,
                                                                      roc_max_fpr=roc_max_fpr, dataset_nm=dataset_nm)
        else:
            eval_data_preds = eval_data_preds.sort_values(by='y_score', ascending=False).reset_index(drop=True)
            # 3 - calc evaluation metrics
            _thresholds = kwargs.get('y_pred_thresholds_gnn', None)
            if _thresholds is not None:
                scores = calc_binary_classification_metrics_using_prob_y_score_and_y_pred_thresholds(
                    y_true=y_true, y_score=y_score, y_pred_thresholds=_thresholds, dataset_nm=dataset_nm)
            else:
                scores = calc_binary_classification_metrics_using_y_score(y_true=y_true, y_score=y_score,
                                                                          roc_max_fpr=roc_max_fpr,
                                                                          dataset_nm=dataset_nm)
        for k, v in scores.items():
            print(f"{k}: {v}")

        return scores, eval_data_preds

    @classmethod
    def train_and_test(cls, X_train: pd.DataFrame, y_train: List[int], X_test: pd.DataFrame, y_test: List[int],
                       model_args: dict, metadata_train: pd.DataFrame, metadata_test: pd.DataFrame,
                       unq_train: pd.DataFrame = None, unq_test: pd.DataFrame = None,
                       train_neg_sampling: bool = True, srna_acc_col: str = 'sRNA_accession_id_Eco',
                       mrna_acc_col: str = 'mRNA_accession_id_Eco',
                       is_syn_col: str = 'is_synthetic', **kwargs) -> \
            (Dict[str, dict], Dict[str, object], Dict[str, object], Dict[str, object]):
        """
        torch_geometric.__version__ = '2.1.0'

        Parameters
        ----------
        X_train: pd.DataFrame (n_samples, N_features),
        y_train: list (n_samples,),
        X_test: pd.DataFrame (t_samples, N_features),
        y_test: list (t_samples,)
        model_args: Dict of model's constructor and fit() arguments
        metadata_train: pd.DataFrame (n_samples, T_features)
        metadata_test: pd.DataFrame (t_samples, T_features)
        unq_train: pd.DataFrame (_samples, T_features)
        unq_test: pd.DataFrame (t_samples, T_features)
        train_neg_sampling: whether to add random negative sampling to train HeteroData
        srna_acc_col: str  sRNA EcoCyc accession id col in metadata_train and metadata_test
        mrna_acc_col: str  mRNA EcoCyc accession id col in metadata_train and metadata_test
        is_syn_col: is synthetic indicator col in metadata_train
        kwargs:
            mrna_eco: all mRNA metadata from EcoCyc

        Returns
        -------

        scores - Dict in the following format:  {
            'test': {
                <score_nm>: score
                }
        }
        predictions - Dict in the following format:  {
            'test_pred': array-like (t_samples,)  - ordered as y test
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
        logger.debug(f"training an HeteroGraph model  ->  train negative sampling = {train_neg_sampling}")
        # 1 - define graph nodes
        if not cls.nodes_are_defined:
            cls._define_nodes_and_edges(**kwargs)

        # if not cv
        if unq_train is None or unq_test is None:
            out_test_pred = pd.DataFrame({
                srna_acc_col: metadata_test[srna_acc_col],
                mrna_acc_col: metadata_test[mrna_acc_col]
            })
            # 2 - remove synthetic data from train
            if sum(metadata_train[is_syn_col]) > 0:
                logger.warning("removing synthetic samples from train")
                X_train, y_train, metadata_train = \
                    cls._remove_synthetic_samples(X=X_train, y=y_train, metadata=metadata_train, is_syn_col=is_syn_col)

            # 3 - get unique interactions data (train and test)
            unq_train = cls._get_unique_inter(metadata=metadata_train, y=y_train, srna_acc_col=srna_acc_col,
                                              mrna_acc_col=mrna_acc_col, df_nm='train')
            unq_test = cls._get_unique_inter(metadata=metadata_test, y=y_test, srna_acc_col=srna_acc_col,
                                             mrna_acc_col=mrna_acc_col, df_nm='test')

            # 4 - assert no data leakage between train and test
            cls._assert_no_data_leakage(unq_train=unq_train, unq_test=unq_test, srna_acc_col=srna_acc_col,
                                        mrna_acc_col=mrna_acc_col)

            # 5 - map interactions to edges
            unq_train = cls._map_interactions_to_edges(unique_intr=unq_train, srna_acc_col=srna_acc_col,
                                                       mrna_acc_col=mrna_acc_col)
            unq_test = cls._map_interactions_to_edges(unique_intr=unq_test, srna_acc_col=srna_acc_col,
                                                      mrna_acc_col=mrna_acc_col)
            # 4.1 - update output df
            _len = len(out_test_pred)
            out_test_pred = pd.merge(out_test_pred, unq_test, on=[srna_acc_col, mrna_acc_col], how='left')
            assert len(out_test_pred) == _len
        else:
            out_test_pred = pd.DataFrame({
                cls.srna_nid_col: unq_test[cls.srna_nid_col],
                cls.mrna_nid_col: unq_test[cls.mrna_nid_col]
            })
        # 5 - init train & test sets (HeteroData)
        train_data, test_data = cls._init_train_test_hetero_data(unq_train=unq_train, unq_test=unq_test,
                                                                 train_neg_sampling=train_neg_sampling)
        model_args['add_sim'] = False

        # 7 - init HeteroGraph model
        # 7.1 - set params
        srna_num_emb = len(cls.srna_nodes)
        mrna_num_emb = len(cls.mrna_nodes)
        metadata = train_data.metadata()
        # 7.2 - init model
        hga_model = GraphRNA(srna=cls.srna, mrna=cls.mrna, srna_to_mrna=cls.srna_to_mrna,
                             srna_num_embeddings=srna_num_emb, mrna_num_embeddings=mrna_num_emb,
                             metadata=metadata, model_args=model_args)
        if cls.debug_logs:
            logger.debug(hga_model)

        # 8 - train HeteroGraph model
        if cls.train_w_loader:
            hg_model = cls._train_hgnn_with_loader(hg_model=hga_model, train_data=train_data, model_args=model_args)
        else:
            hg_model = cls._train_hgnn(hga_model=hga_model, train_data=train_data, model_args=model_args)
        training_history, train_val_data = {}, {}

        # 9 - evaluate - calc scores
        logger.debug("evaluating test set")
        predictions, scores = {}, {}
        test_scores, test_pred_df = cls._eval_hgnn(trained_model=hg_model, eval_data=test_data, model_args=model_args,
                                                   **kwargs)
        assert pd.isnull(test_pred_df).sum().sum() == 0, "some null predictions"

        # 10 - update outputs
        _len = len(out_test_pred)
        out_test_pred = pd.merge(out_test_pred, test_pred_df, on=[cls.srna_nid_col, cls.mrna_nid_col], how='left')
        assert len(out_test_pred) == _len
        test_y_pred = out_test_pred['y_score']
        test_original_score = out_test_pred['y_original_score']

        scores.update({'test': test_scores})
        predictions.update({'test_pred': test_y_pred, 'test_original_score': test_original_score,
                            'test_pred_df': test_pred_df, 'out_test_pred': out_test_pred})

        return scores, predictions, training_history, train_val_data

    @classmethod
    def train_and_test_all_pairs(cls, X_train: pd.DataFrame, y_train: List[int], X_test_all_unq_pairs: pd.DataFrame,
                                 model_args: dict, metadata_train: pd.DataFrame, train_neg_sampling: bool = True,
                                 srna_acc_col: str = 'sRNA_accession_id_Eco',
                                 mrna_acc_col: str = 'mRNA_accession_id_Eco', is_syn_col: str = 'is_synthetic',
                                 **kwargs) -> (pd.DataFrame, Dict[str, object], Dict[str, object]):
        """
        torch_geometric.__version__ = '2.1.0'

        Parameters
        ----------
        X_train: pd.DataFrame (n_samples, N_features),
        y_train: list (n_samples,),
        X_bench: pd.DataFrame (t_samples, N_features),
        y_bench: list (t_samples,)
        X_test_all_unq_pairs: pd.DataFrame
        model_args: Dict of model's constructor and fit() arguments
        metadata_train: pd.DataFrame (n_samples, T_features)
        metadata_bench: pd.DataFrame (t_samples, T_features)
        unq_train: pd.DataFrame (_samples, T_features)
        unq_test: pd.DataFrame (t_samples, T_features)
        train_neg_sampling: whether to add random negative sampling to train HeteroData
        srna_acc_col: str  sRNA EcoCyc accession id col in metadata_train and metadata_test
        mrna_acc_col: str  mRNA EcoCyc accession id col in metadata_train and metadata_test
        is_syn_col: is synthetic indicator col in metadata_train
        kwargs:
            mrna_eco: all mRNA metadata from EcoCyc

        Returns
        -------


        out_test_pred: pd.DataFrame
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
        logger.debug(f"training an HeteroGraph model  ->  train negative sampling = {train_neg_sampling}")
        # 1 - define graph nodes
        if not cls.nodes_are_defined:
            cls._define_nodes_and_edges(**kwargs)

        # 2 - remove synthetic data from train
        if sum(metadata_train[is_syn_col]) > 0:
            logger.warning("removing synthetic samples from train")
            X_train, y_train, metadata_train = \
                cls._remove_synthetic_samples(X=X_train, y=y_train, metadata=metadata_train, is_syn_col=is_syn_col)

        # 3 - get unique interactions data (train)
        unq_train = cls._get_unique_inter(metadata=metadata_train, y=y_train, srna_acc_col=srna_acc_col,
                                          mrna_acc_col=mrna_acc_col, df_nm='train')
        unq_test = X_test_all_unq_pairs

        # 4 - remove train interactions from all pairs
        unq_test = cls._remove_train_inter_from_test_all_pairs(unq_train=unq_train, unq_test=unq_test,
                                                               srna_acc_col=srna_acc_col, mrna_acc_col=mrna_acc_col)
        unq_test[cls.binary_intr_label_col] = 1

        # 5 - map interactions to edges
        unq_train = cls._map_interactions_to_edges(unique_intr=unq_train, srna_acc_col=srna_acc_col,
                                                   mrna_acc_col=mrna_acc_col)
        unq_test = cls._map_interactions_to_edges(unique_intr=unq_test, srna_acc_col=srna_acc_col,
                                                  mrna_acc_col=mrna_acc_col)
        # 4.1 - update output df
        out_test_pred = unq_test

        # 5 - init train & test sets (HeteroData)
        train_data, test_data = cls._init_train_test_hetero_data(unq_train=unq_train, unq_test=unq_test,
                                                                 train_neg_sampling=train_neg_sampling)
        model_args['add_sim'] = False

        # 7 - init HeteroGraph model
        # 7.1 - set params
        srna_num_emb = len(cls.srna_nodes)
        mrna_num_emb = len(cls.mrna_nodes)
        metadata = train_data.metadata()
        # 7.2 - init model
        hga_model = GraphRNA(srna=cls.srna, mrna=cls.mrna, srna_to_mrna=cls.srna_to_mrna,
                             srna_num_embeddings=srna_num_emb, mrna_num_embeddings=mrna_num_emb,
                             metadata=metadata, model_args=model_args)
        if cls.debug_logs:
            logger.debug(hga_model)

        # 8 - train HeteroGraph model
        if cls.train_w_loader:
            hg_model = cls._train_hgnn_with_loader(hg_model=hga_model, train_data=train_data, model_args=model_args)
        else:
            hg_model = cls._train_hgnn(hga_model=hga_model, train_data=train_data, model_args=model_args)
        training_history, train_val_data = {}, {}

        # 9 - evaluate - calc scores
        logger.debug("predicting on all pairs")
        test_pred_df = cls._predict_hgnn(trained_model=hg_model, eval_data=test_data, model_args=model_args, **kwargs)
        assert pd.isnull(test_pred_df).sum().sum() == 0, "some null predictions"

        # 10 - update outputs
        _len = len(out_test_pred)
        out_test_pred = pd.merge(out_test_pred, test_pred_df, on=[cls.srna_nid_col, cls.mrna_nid_col], how='left')
        out_test_pred = out_test_pred[[x for x in out_test_pred.columns.values if x != cls.binary_intr_label_col]]
        assert len(out_test_pred) == _len

        return out_test_pred, training_history, train_val_data

    @classmethod
    def get_predictions_df(cls, unq_intr: pd.DataFrame, y_true: list, y_score: np.array, y_original_score: np.array,
                           out_col_y_true: str = "y_true", out_col_y_score: str = "y_score",
                           out_col_y_original_score: str = "y_original_score",
                           sort_df: bool = True) -> pd.DataFrame:
        is_length_compatible = len(unq_intr) == len(y_true) == len(y_score) == len(y_original_score)
        assert is_length_compatible, "unq_intr, y_true, y_score and metadata are not compatible in length"
        assert pd.isnull(unq_intr).sum().sum() == sum(pd.isnull(y_true)) == sum(pd.isnull(y_score)) \
               == sum(pd.isnull(y_original_score)) == 0, "nulls in data"

        _df = unq_intr
        _df[out_col_y_true] = y_true
        _df[out_col_y_score] = y_score
        _df[out_col_y_original_score] = y_original_score

        # add metadata
        srna_meta_cols = [c for c in cls.srna_nodes.columns.values if c != cls.srna_nid_col]
        mrna_meta_cols = [c for c in cls.mrna_nodes.columns.values if c != cls.mrna_nid_col]
        _len = len(_df)
        _df = pd.merge(_df, cls.srna_nodes, on=cls.srna_nid_col, how='left').rename(
            columns={c: f"sRNA_{c}" for c in srna_meta_cols})
        _df = pd.merge(_df, cls.mrna_nodes, on=cls.mrna_nid_col, how='left').rename(
            columns={c: f"mRNA_{c}" for c in mrna_meta_cols})
        assert len(_df) == _len, "duplications post merge"

        if sort_df:
            _df = _df.sort_values(by=out_col_y_score, ascending=False).reset_index(drop=True)

        return _df

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
        n_splits_rnd = 10
        cv_outs = {}

        # 1 - define X, y and metadata
        train_for_cv_rnd = kwargs['train_for_cv_rnd']
        cols_to_rem = ['y_score', 'COPRA_sRNA_is_missing', 'COPRA_sRNA', 'COPRA_mRNA', 'COPRA_mRNA_locus_tag',
                            'COPRA_pv', 'COPRA_fdr', 'COPRA_NC_000913', 'COPRA_mRNA_not_in_output',
                            'COPRA_validated_pv', 'COPRA_validated_score']
        X_rnd  = train_for_cv_rnd[feature_cols]
        y_rnd = list(train_for_cv_rnd[cls.binary_intr_label_col])
        meta_cols = [c for c in train_for_cv_rnd.columns.values if c not in [cls.binary_intr_label_col] + feature_cols + cols_to_rem]
        metadata_rnd = train_for_cv_rnd[meta_cols]

        # 2 - run CV with random sampling of negatives
        cv_scores, cv_predictions_dfs, cv_training_history, cv_data = \
            cls.run_cross_validation(X=X_rnd, y=y_rnd, metadata=metadata_rnd, n_splits=n_splits_rnd,
                                     model_args=model_args, **kwargs)

        # 3 - save results
        cv_outs['neg_rnd'] = {
            "cv_scores": cv_scores,
            "cv_predictions_dfs": cv_predictions_dfs,
            "cv_training_history": cv_training_history,
        }

        return cv_outs

    @classmethod
    def run_cross_validation(cls, X: pd.DataFrame, y: List[int], n_splits: int, model_args: dict,
                             metadata: pd.DataFrame = None, srna_acc_col: str = 'sRNA_accession_id_Eco',
                             mrna_acc_col: str = 'mRNA_accession_id_Eco', is_syn_col: str = 'is_synthetic', **kwargs) \
            -> (Dict[int, Dict[str, Dict[str, float]]], Dict[int, pd.DataFrame], Dict[int, dict], Dict[int, dict]):
        """
        Returns
        -------

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
        logger.debug(f"running cross validation with {n_splits} folds")

        # 1 - remove all synthetic samples
        logger.warning("removing all synthetic samples")
        X_no_syn, y_no_syn, metadata_no_syn = \
            cls._remove_synthetic_samples(X=X, y=y, metadata=metadata, is_syn_col=is_syn_col)

        # 2 - get unique interactions data (train + val)
        unq_intr = cls._get_unique_inter(metadata=metadata_no_syn, y=y_no_syn, srna_acc_col=srna_acc_col,
                                         mrna_acc_col=mrna_acc_col, df_nm='all')
        unq_intr_pos, unq_intr_neg = cls._pos_neg_split(df=unq_intr, binary_label_col=cls.binary_intr_label_col)

        # 3 - define graph nodes (if needed) and map interaction
        if not cls.nodes_are_defined:
            cls._define_nodes_and_edges(**kwargs)
        unq_intr_pos = cls._map_interactions_to_edges(unique_intr=unq_intr_pos, srna_acc_col=srna_acc_col,
                                                      mrna_acc_col=mrna_acc_col)
        # 4 - random negative sampling - all cv data
        _shuffle = True
        unq_data = cls._add_neg_samples(unq_intr_pos=unq_intr_pos, ratio=cls.cv_neg_sampling_ratio_data,
                                        _shuffle=_shuffle)
        unq_y = np.array(unq_data[cls.binary_intr_label_col])
        unq_intr_data = unq_data[[cls.srna_nid_col, cls.mrna_nid_col]]

        # 5 - split data into folds
        cv_data_unq = get_stratified_cv_folds_for_unique(unq_intr_data=unq_intr_data, unq_y=unq_y, n_splits=n_splits,
                                                         label_col=cls.binary_intr_label_col)
        dummy_x_train, dummy_x_val = pd.DataFrame(), pd.DataFrame()
        dummy_y_train, dummy_y_val = list(), list()
        dummy_meta_train, dummy_meta_val = pd.DataFrame(), pd.DataFrame()

        # 6 - predict on folds
        cv_scores = {}
        cv_training_history = {}
        cv_prediction_dfs = {}
        train_neg_sampling = False  # negatives were already added to cv_data_unq
        for fold, fold_data_unq in cv_data_unq.items():
            logger.debug(f"starting fold {fold}")
            # 2.2 - predict on validation set (pos + random sampled neg)
            scores, predictions, training_history, _ = \
                cls.train_and_test(X_train=dummy_x_train, y_train=dummy_y_train, X_test=dummy_x_val, y_test=dummy_y_val,
                                   model_args=model_args, metadata_train=dummy_meta_train, metadata_test=dummy_meta_val,
                                   unq_train=fold_data_unq['unq_train'], unq_test=fold_data_unq['unq_val'],
                                   train_neg_sampling=train_neg_sampling, **kwargs)
            # 2.1 - fold's val scores
            cv_scores[fold] = {'val': scores['test']}
            # 2.2 - fold's training history
            cv_training_history[fold] = training_history
            # 2.3 - fold's predictions df
            y_val_pred = predictions['test_pred']
            y_val_original_score = predictions['test_original_score']
            unq_val = fold_data_unq['unq_val'][[cls.srna_nid_col, cls.mrna_nid_col]]
            y_val = fold_data_unq['unq_val'][cls.binary_intr_label_col]
            cv_pred_df = cls.get_predictions_df(unq_intr=unq_val, y_true=y_val, y_score=y_val_pred,
                                                y_original_score=y_val_original_score)
            cv_prediction_dfs[fold] = cv_pred_df

        return cv_scores, cv_prediction_dfs, cv_training_history, cv_data_unq
