# GraphRNA
This repository is the official implementation of the paper 
*"Enhancing sRNA-target prediction using graph neural network and local interaction features"*.

We introduce two machine-learning-based approaches for predicting sRNA-target interactions. The first method, ***GraphRNA***, is based on a graph neural network (GNN), and the second method uses ensemble models (XGBoost, RandomForest) and local interaction features computed over a predicted interaction duplex. Unlike other state-of-the-art prediction tools, our approaches enable the prediction of species-specific sRNAs interactions.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Data

All necessary data is provided for *Escherichia coli K12 MG1655* (NC_000913) under the `data/` directory.

For a detailed description per dataset, see the `load_data` function docstrings under `main.py`.

### Interactions data
Data source: [sInterBase](https://academic.oup.com/bioinformatics/article/39/4/btad172/7115836).

Train and test datasets, including the predicted duplex and computed features.

### RNA data
Data source: [EcoCyc](https://ecocyc.org/).

sRNA and mRNA data (used by ***GraphRNA***).



## Methods application

### Training and evaluation
```python
from models_handlers.graph_rna_model_handler import GraphRNAModelHandler
from models_handlers.xgboost_model_handler import XGBModelHandler
from models_handlers.rf_model_handler import RFModelHandler

# 1 - load data (see load_data implementation in main.py)  
data_path = "/sise/home/shanisa/GraphRNA/data"
train_fragments, test_complete, test_filtered, kwargs = load_data(data_path=data_path)
train, test = train_fragments, test_complete

# 2 - select model
model_h = GraphRNAModelHandler()  # XGBModelHandler()  # RFModelHandler()
model_args = model_h.get_model_args()

# 2 - run n-fold cross validation
cv_n_splits = 10
cv_predictions_dfs, cv_training_history = \
    model_h.run_cross_validation(X=train['X'], y=train['y'], metadata=train['metadata'], n_splits=cv_n_splits,
                                 model_args=model_args, **kwargs)
# 3 - train and test
predictions, training_history = \
    model_h.train_and_test(X_train=train['X'], y_train=train['y'], X_test=test['X'], y_test=test['y'], 
                           model_args=model_args, metadata_train=train['metadata'], metadata_test=test['metadata'], 
                           **kwargs)
test_predictions_df = predictions['out_test_pred']
```
### Reproducing results in the paper

Our results can be reproduced by directly running `main.py` with configured paths for data and outputs.