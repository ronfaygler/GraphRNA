import pandas as pd
from os.path import join
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, confusion_matrix
import numpy as np
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def create_rna_df(data_path, file_name, id_col, seq_col, output_file="", is_train_test=False):
    '''    create srna / mrna files    '''
    if is_train_test: # the full path is here
        train_df = pd.read_csv(file_name)
    else:
        train_df = pd.read_csv(join(data_path, file_name)) 
    # Create a new DataFrame with the specified columns and values
    rna_df = pd.DataFrame({
        "EcoCyc_accession_id": train_df[id_col],
        "EcoCyc_locus_tag": train_df[id_col],
        "EcoCyc_rna_name": train_df[id_col],
        "EcoCyc_rna_name_synonyms": train_df[id_col],
        "EcoCyc_start": 0,
        "EcoCyc_end": 0,
        "EcoCyc_strand": '+',
        "EcoCyc_sequence": train_df[seq_col],  # Assigning miRNA sequence to EcoCyc_sequence
        "EcoCyc_accession-2": train_df[id_col]
    })

    # Display the populated DataFrame
    rna_df = rna_df.drop_duplicates(subset='EcoCyc_accession_id', keep='first')
    
    if id_col.startswith("mRNA") or is_train_test:
        return rna_df

    rna_df.to_csv(join(data_path, output_file), index=False)
    print(f"created rna data file in {output_file}")

#rbp
# create_rna_df(data_path="/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp", 
#         file_name="ENCORI_hg38_RBPTarget.csv", 
#         id_col='RBP', seq_col='RBP', 
#         output_file="/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/DATA_rbp_eco.csv")

# # Create mRNA DataFrame from rbp interactions
# mrna_df_1 = create_rna_df(data_path="/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp", 
#                            file_name="ENCORI_hg38_RBPTarget.csv", 
#                            id_col='mRNA_ID_with_RBP', 
#                            seq_col='geneName')

# # Create mRNA DataFrame from mirna interactions
# mrna_df_2 = create_rna_df(data_path="/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp", 
#                            file_name="h3.csv", 
#                            id_col='mRNA_ID_with_sRNA',  # Assuming this is still relevant for the second file
#                            seq_col='sequence')

# # Combine both DataFrames and drop duplicates
# combined_df = pd.concat([mrna_df_1, mrna_df_2], ignore_index=True).drop_duplicates(subset='EcoCyc_accession_id', keep='first')

# # Save the combined DataFrame to a CSV file
# combined_df.to_csv("/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/DATA_mrna_eco.csv", index=False)
# print("Created combined RNA data file at DATA_mrna_eco.csv")

# --- mirna from rbp
# create_rna_df(data_path="/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp", 
#         file_name="h3.csv", 
#         id_col='miRNA ID', seq_col='miRNA sequence',
#         output_file="/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/DATA_mirna_eco.csv")


#---------- create calculate metrics
def calculate_metrics(y_true_list, y_score_list):
    auc_value = roc_auc_score(y_true_list, y_score_list)

    # Calculate FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true_list, y_score_list)
    youden_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden_index)]

    # Calculate pAUC (Partial AUC up to FPR of 0.1, for example)
    pauc_value = auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1])
    

    # Calculate precision, recall, and thresholds for Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true_list, y_score_list)

    # Calculate PR-AUC (Precision-Recall AUC)
    pr_auc_value = auc(recall, precision)

    # Calculate F1 Score
    y_pred = (y_score_list > optimal_threshold).astype(int)  # Convert probabilities to binary predictions
    f1_value = f1_score(y_true_list, y_pred)

    # Calculate Accuracy
    acc_value = accuracy_score(y_true_list, y_pred)

    # Calculate True Negative Rate (TNR)
    tn, fp, fn, tp = confusion_matrix(y_true_list, y_pred).ravel()
    tnr_value = tn / (tn + fp)

    return auc_value, pauc_value, pr_auc_value, f1_value, acc_value, tnr_value, fpr, tpr, thresholds

def create_metric_df(dfs):
    # Create lists to hold the results
    auc_list = []
    pauc_list = []
    pr_auc_list = []
    f1_list = []
    acc_list = []
    tnr_list = []
    fpr_list = []
    tpr_list = []
    thresholds_list = []

    all_metrics = []
    for i, df in enumerate(dfs):
        # Drop rows where 'y_true' or 'y_graph_score' is null
        df_clean = df[['y_true', 'y_graph_score']].dropna()

        # Extract the y_true and y_score columns after removing null values
        y_true = df_clean['y_true'].values
        y_score = df_clean['y_graph_score'].values
            
        # Calculate metrics for the entire DataFrame
        auc_value, pauc_value, pr_auc_value, f1_value, acc_value, tnr_value, fpr, tpr, thresholds = calculate_metrics(y_true, y_score)
        
        # Create a dictionary with the metrics
        metrics = {
            'Fold': i,
            'AUC': auc_value,
            'pAUC': pauc_value,
            'PR-AUC': pr_auc_value,
            'F1': f1_value,
            'Accuracy': acc_value,
            'TNR': tnr_value,
            'FPR': fpr.tolist(),  # Store as a single-element list to keep it in one cell
            'TPR': tpr.tolist(),  # Store as a single-element list to keep it in one cell
            'thresholds': thresholds.tolist()  # Store as a single-element list to keep it in one cell
        }

        # Convert the dictionary to a DataFrame with a single row
        metrics_df = pd.DataFrame([metrics])
        
        # Append the DataFrame to the list
        all_metrics.append(metrics_df)

    # Concatenate all DataFrames in the list into a single DataFrame
    final_metrics_df = pd.concat(all_metrics, ignore_index=True)

    # Save the final DataFrame to a CSV file
    # final_metrics_df.to_csv("/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/RF/RF_Mock_miRNA_metrics_summary.csv", index=False)

    # final_metrics_df.to_csv("/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/XGB/categorial_XGB_Mock_miRNA_metrics_summary.csv", index=False)
    # final_metrics_df.to_csv("/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir_rbp/GNN/metrics_summary.csv", index=False)
    final_metrics_df.to_csv("/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/GNN-Random_neg/10 folds/metrics_summary.csv", index=False)



# dfs = [pd.read_csv(f"/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/GNN/cv_fold{i}_predictions_GraphRNA.csv") for i in range(10)]
# dfs = [pd.read_csv(f"/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs/cv_fold{i}_predictions_GraphRNA.csv") for i in range(10)]
# dfs = [pd.read_csv(f"/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/RF/cv_fold{i}_predictions_RandomForest.csv") for i in range(10)]
# dfs = [pd.read_csv(f"/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/XGB/cv_fold{i}_predictions_XGBoost.csv") for i in range(10)]
# dfs = [pd.read_csv(f"/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir_rbp/GNN/cv_fold{i}_predictions_GraphRNA.csv") for i in range(10)]
# dfs = [pd.read_csv(f"/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/GNN-Random_neg/10 folds/cv_fold{i}_predictions_GraphRNA.csv") for i in range(10)]

# create_metric_df(dfs)


def get_features_cols(self):
    ''' create features from mirna + mrna interactions, from "S1 Table-features.xlsx" file,
     was a field: - self.features_file = "S1 Table-features.xlsx" in DataHandler_Mirna_Mrna class
     and this function also was in DataHandler_Mirna_Mrna class '''
    features_df = pd.read_excel(join(self.data_path, self.features_file), sheet_name=0)
    # Save the DataFrame as a CSV file
    features_df.to_csv(join(self.data_path, f"{self.features_file[:-5]}.csv"), index=False)
    features_df = pd.read_csv(join(self.data_path, f"{self.features_file[:-5]}.csv"))
    original_features = features_df['Feature name']
    features = features_df['h3']
    for i in range(36,54):
        if i != 40 and i != 46 and i != 52:
            features[i] = f'miRNAPairingCount_{original_features[i][1:]}'
    for i in range(54,67):
        features[i] = original_features[i]
    for i in range(68,128):
        features[i] = f'MRNA_{original_features[i]}'
    for i in range(129,136):
        features[i] = f'Energy_{original_features[i]}'
    for i in range(139,141):
        features[i] = f'MRNA_{original_features[i]}'
    features = features.dropna()

    #add all Acc_P%_#th to features_cols
    mir_df = pd.read_csv(join(self.data_path, self.train_fragments_file)) 
    acc_cols = [col for col in mir_df.columns if col.startswith("Acc_")]

    features_cols = features.tolist()
    features_cols.extend(acc_cols)
    with open(join(self.data_path,'features_cols.txt'), 'w') as file:
        for item in features_cols:
            file.write(f"{item}\n")
    return features_cols


# data_path = "/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir"
# train_fragments_file =  "h3.csv"
# mirna_data_file = "DATA_mirna_eco.csv"
# mrna_data_file = "DATA_mrna_eco.csv"



def is_same_cols(df1, df2):
    '''
    check if the columns in test srna mrna output are the same as in mirna mrna output
    '''
    # Get the columns from each DataFrame
    columns_df1 = set(df1.columns)
    columns_df2 = set(df2.columns)

    # Find differences
    columns_in_df1_not_in_df2 = columns_df1 - columns_df2
    columns_in_df2_not_in_df1 = columns_df2 - columns_df1

    # Print results
    if not columns_in_df1_not_in_df2 and not columns_in_df2_not_in_df1:
        print("Both DataFrames have the same columns.")
    else:
        if columns_in_df1_not_in_df2:
            print("Columns in df1 but not in df2:")
            print(columns_in_df1_not_in_df2)
        if columns_in_df2_not_in_df1:
            print("Columns in df2 but not in df1:")
            print(columns_in_df2_not_in_df1)


# # Load the CSV files into DataFrames
# df1 = pd.read_csv("/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs/test_predictions_GraphRNA.csv")
# df2 = pd.read_csv("/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/GNN/cv_fold0_predictions_GraphRNA.csv")
# df1=pd.read_csv("/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs/cv_fold0_predictions_GraphRNA.csv")
# df1 = pd.read_csv("/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir/NPS_CLASH_MFE.csv")
# df2 = pd.read_csv("/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir/h3.csv")
# is_same_cols(df1,df2)

# # check labels
# df3 = pd.read_csv("/home/ronfay/Data_bacteria/graphNN/GraphRNA/data/DATA_train_fragments.csv")
# print(df3['interaction_label'].value_counts())



def is_train_and_fold_lengths_equal(train_fragments_df):
    ''' check if length of train file is the sum of cv fold length files '''
    # List to hold the row counts from the cv_foldX_predictions_GraphRNA.csv files
    row_counts = []

    # Loop over the fold files
    for i in range(10):  # Assuming there are 10 folds (0 to 9)
        filename = f"/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/GNN/cv_fold{i}_predictions_GraphRNA.csv"
        fold_df = pd.read_csv(filename)
        row_counts.append(len(fold_df))

    # Calculate the sum of rows from all the cv_foldX_predictions_GraphRNA.csv files
    total_cv_rows = sum(row_counts)
    train_fragments_row_count = len(train_fragments_df)

    # Compare the row counts
    if train_fragments_row_count == total_cv_rows:
        print("The number of rows in self.train_fragments_file is equal to the sum of rows in the CV files.")
    else:
        print(f"The number of rows in self.train_fragments_file ({train_fragments_row_count}) does not match the sum of rows in the CV files ({total_cv_rows}).")

# train_fragments_df = pd.read_csv("/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir/h3.csv")
# is_train_and_fold_lengths_equal(train_fragments_df)


def combine_and_XGB():
    data_path = "/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir"

    with open(join(data_path,'features_cols.txt'), 'r') as file:
        features_cols = [line.strip() for line in file]

    neg_df = pd.read_csv("/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir/Mock_miRNA.csv")
    neg_df = neg_df.iloc[1:].reset_index(drop=True)
    pos_df = pd.read_csv("/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir/h3.csv")
    pos_df = pos_df.iloc[1:].reset_index(drop=True)

    # Assuming pos_df and neg_df have the same columns
    # Add label column (1 for positive, 0 for negative)
    pos_df['label'] = 1
    neg_df['label'] = 0

    for i in range(1, 21):
        name_col = "miRNAMatchPosition_" + str(i)
        pos_df[name_col] = pos_df[name_col].astype('category').cat.codes
        neg_df[name_col] = neg_df[name_col].astype('category').cat.codes

    neg_df=neg_df[features_cols+['label']]
    pos_df=pos_df[features_cols+['label']]
    # print(neg_df.dtypes)
    # print(pos_df.dtypes)
    
    # Convert object columns to numeric
    object_columns = neg_df.select_dtypes(include=['object']).columns
    # print("Object columns:", object_columns)
    neg_df[object_columns] = neg_df[object_columns].apply(pd.to_numeric, errors='coerce')
    
    object_columns = pos_df.select_dtypes(include=['object']).columns
    # print("Object columns:", object_columns)
    pos_df[object_columns] = pos_df[object_columns].apply(pd.to_numeric, errors='coerce')

    # # Number of loops
    n_loops = 10

    # Initialize lists to store results
    train_accuracies = []
    test_accuracies = []

    # XGBoost parameters (you can adjust them as needed)
    xgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    auc_list = []
    pauc_list = []
    pr_auc_list = []
    f1_list = []
    acc_list = []
    tnr_list = []
    fpr_list = []
    tpr_list = []
    thresholds_list = []

    all_metrics = []
    # Loop 10 times
    for i in range(n_loops):
    #   # Sample 80% of pos_df and neg_df for training
        pos_train, pos_test = train_test_split(pos_df, test_size=0.2, random_state=np.random.randint(0, 1000))
        neg_train, neg_test = train_test_split(neg_df, test_size=0.2, random_state=np.random.randint(0, 1000))
        print ("len(pos_train):", len(pos_train))
        print ("len(neg_train):", len(neg_train))
        print ("len(pos_test):", len(pos_test))
        print ("len(neg_test):", len(neg_test))
        # return
        # Combine pos and neg for training and testing
        train_df = pd.concat([pos_train, neg_train], axis=0)
        test_df = pd.concat([pos_test, neg_test], axis=0)
        
        # Shuffle the training and testing data
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)
        print("len(train_df): ", len(train_df))
        print("len(test_df) : ", len(test_df))
        # return
        # Check the columns of train and test data
        # print("Training Data Columns:", train_df.columns)
        # print("Test Data Columns:", test_df.columns)

        # Split features and labels
        X_train, y_train = train_df.drop('label', axis=1), train_df['label']
        X_test, y_test = test_df.drop('label', axis=1), test_df['label']

        for col in X_train.columns:
            if col not in features_cols:
                return "col not in features_cols:" + col
        for col in X_test.columns:
            if col not in features_cols:
                return "col not in features_cols:" + col
        for col in y_train:
            if col in features_cols:
                return "col in features_cols:" + col
        for col in y_test:
            if col in features_cols:
                return "col in features_cols:" + col
        # Check if X_train and X_test contain only feature columns
        # print("X_train Columns:", X_train.columns)
        # print("X_test Columns:", X_test.columns)
        
        # print("X_train:", X_train)
        # print("X_test:", X_test)

        # print("Y_train :", y_train)
        # print("Y_test :", y_test)

    # print("X_train null values:\n", X_train.isnull().sum())
    # print("X_test null values:\n", X_test.isnull().sum())

    # print("y_train null values:\n", y_train.isnull().sum())
    # print("y_test null values:\n", y_test.isnull().sum())

    #     # Function to print the indices of null values in pandas DataFrame or Series
    # def print_null_positions(df, name):
    #     print(f"Null values in {name}:")
    #     null_positions = df.isnull()
        
    #     # For DataFrames
    #     if null_positions.any().any():  # Check if there's any null value
    #         for column in df.columns:
    #             if null_positions[column].any():
    #                 print(f"Column '{column}' has null values at rows: {null_positions[null_positions[column]].index.tolist()}")
    #     # For Series
    #     elif null_positions.any():
    #         print(f"Series has null values at indices: {null_positions[null_positions].index.tolist()}")
    #     else:
    #         print(f"No null values found in {name}.")

    # # Checking for null values in X_train, X_test, y_train, y_test
    # print_null_positions(X_train, "X_train")
    # print_null_positions(X_test, "X_test")
    
    #     # ------------------ Initialize the XGBClassifier
    #     model = XGBClassifier(**xgb_params)
        
    #     # Train the model
    #     model.fit(X_train, y_train)
        
    #     # Predict on the training set
    #     y_train_pred = model.predict(X_train)
    #     train_acc = accuracy_score(y_train, y_train_pred)
    #     train_accuracies.append(train_acc)
        
    #     # Predict on the test set
    #     y_test_pred = model.predict(X_test)
    #     test_acc = accuracy_score(y_test, y_test_pred)
    #     test_accuracies.append(test_acc)
        
    #     print(f"Loop {i+1} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    #     # Calculate metrics for the entire DataFrame
    #     auc_value, pauc_value, pr_auc_value, f1_value, acc_value, tnr_value, fpr, tpr, thresholds = calculate_metrics(y_test, y_test_pred)
        
    #     # Create a dictionary with the metrics
    #     metrics = {
    #         'Fold': i,
    #         'AUC': auc_value,
    #         'pAUC': pauc_value,
    #         'PR-AUC': pr_auc_value,
    #         'F1': f1_value,
    #         'Accuracy': acc_value,
    #         'TNR': tnr_value,
    #         'FPR': fpr.tolist(),  # Store as a single-element list to keep it in one cell
    #         'TPR': tpr.tolist(),  # Store as a single-element list to keep it in one cell
    #         'thresholds': thresholds.tolist()  # Store as a single-element list to keep it in one cell
    #     }

    #     # Convert the dictionary to a DataFrame with a single row
    #     metrics_df = pd.DataFrame([metrics])
        
    #     # Append the DataFrame to the list
    #     all_metrics.append(metrics_df)

    # # Concatenate all DataFrames in the list into a single DataFrame
    # final_metrics_df = pd.concat(all_metrics, ignore_index=True)

    # # Save the final DataFrame to a CSV file
    # final_metrics_df.to_csv("/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/XGB/XGB_my_Mock_miRNA_metrics_summary.csv", index=False)

    # # Output the average train and test accuracies
    # print(f"\nAverage Train Accuracy: {np.mean(train_accuracies):.4f}")
    # print(f"Average Test Accuracy: {np.mean(test_accuracies):.4f}")

# ------------------------GridSearchCV
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees
        'max_depth': [3, 6, 9],  # Maximum depth of a tree
        'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
        'subsample': [0.8],  # Fraction of samples used for training
        'colsample_bytree': [0.8],  # Fraction of features used for each tree
        'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required to make a split
    }

    # Initialize the XGBClassifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, 
                            cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

    # Perform grid search and fit on your training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters found: {best_params}")

    # Train the model using the best parameters
    best_model = grid_search.best_estimator_

    # Predict on the test set
    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    # Define the file name
    output_file = "XGB_gs_efrat_test_acc_best_params.txt"

    # Write the results to the file
    with open(output_file, 'w') as file:
        file.write(f"Best parameters found: {best_params}\n")
        file.write(f"Test Accuracy: {test_acc:.4f}\n")


def update_mrna_mirna_intr_file(input_path, output_path, prev_col_name, new_col_name):
    df = pd.read_csv(input_path)
    #  remove the line with the types of columns
    df = df.iloc[1:]
    df.rename(columns={prev_col_name: new_col_name}, inplace=True)
    # Remove characters from '|' in the 'mRNA_ID_with_sRNA' column
    df[new_col_name] = df[new_col_name].str.split('|').str[0]
    # Save the modified DataFrame back to a CSV file
    df.to_csv(output_path, index=False)
    print(f"Modified mrna mirna intr file and saved to {output_path}.")

# update_mrna_mirna_intr_file(input_path="/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir/h3.csv", output_path="/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/h3.csv", prev_col_name='Gene_ID', new_col_name='mRNA_ID_with_sRNA')


def convert_txt_to_df(input_path, output_csv_path):
    # Read the text file into a DataFrame, skipping commented lines
    df = pd.read_csv(input_path, sep='\t', comment='#')
    
    df = df.iloc[:-1] # in case the data isnt full
    df.rename(columns={'geneID': 'mRNA_ID_with_RBP'}, inplace=True)

    df.to_csv(output_csv_path, index=False)
    print(f'converted txt to df and renamed geneID to mRNA_ID_with_RBP \nfrom {input_path} into {output_csv_path}')
# convert_txt_to_df(input_path = "/sise/home/ronfay/ENCORI_hg38_RBPTarget.txt", output_csv_path="/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/ENCORI_hg38_RBPTarget.csv")


def add_col(file_path):
    df = pd.read_csv(file_path)
    df['Seed_match_A'] = 'value'
    df.to_csv(file_path, index=False)
    print('added Seed_match_A col')
# add_col(file_path="/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/ENCORI_hg38_RBPTarget.csv")


def combine_rbp_mirna_interactions_csvs(df1, df2, output_path):
    # Concatenate the DataFrames along the columns
    combined_df = pd.concat([df1, df2], axis=1)
    
    # Optionally, you can reset the index if needed
    combined_df.reset_index(drop=True, inplace=True)

    # Display the first few rows of the combined DataFrame
    print("combined_df: \n",combined_df.head())
    
    # Save the combined DataFrame to a CSV file if needed
    combined_df.to_csv(output_path, index=False)
    print(f"combined rbp mirna interactions and saved to {output_path}")

# df1 = pd.read_csv('/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/ENCORI_hg38_RBPTarget.csv')
# df2 = pd.read_csv('/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/h3.csv')
# output_path='/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/combined_rbp_mirna_interactions.csv'
# combine_rbp_mirna_interactions_csvs(df1, df2, output_path)


# df1 = pd.read_csv("/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/h3.csv")
# df2 = pd.read_csv("/home/ronfay/Data_bacteria/graphNN/GraphRNA/data_mir_rbp/ENCORI_hg38_RBPTarget.csv")

# # print(df.head())
# print("len mirna: ", len(df1))
# print("len rbp: ", len(df2))