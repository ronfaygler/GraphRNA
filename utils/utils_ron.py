import pandas as pd
from os.path import join
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, confusion_matrix


def create_rna_df(output_file, id_col, seq_col):
    '''    create srna / mrna files    '''
    train_df = pd.read_csv(join(data_path, train_fragments_file)) 
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
    rna_df.to_csv(join(data_path, output_file), index=False)


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

#---------- create calculate metrics


def calculate_metrics(y_true_list, y_score_list):
    auc_value = roc_auc_score(y_true_list, y_score_list)

    # Calculate FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true_list, y_score_list)

    # Calculate pAUC (Partial AUC up to FPR of 0.1, for example)
    pauc_value = auc(fpr[fpr <= 0.1], tpr[fpr <= 0.1])

    # Calculate precision, recall, and thresholds for Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true_list, y_score_list)

    # Calculate PR-AUC (Precision-Recall AUC)
    pr_auc_value = auc(recall, precision)

    # Calculate F1 Score
    y_pred = (y_score_list > 0.5).astype(int)  # Convert probabilities to binary predictions
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
        # Extract the y_true and y_score columns
        y_true = df['y_true'].values
        y_score = df['y_score'].values
        
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
    # final_metrics_df.to_csv("/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/RF/metrics_summary.csv", index=False)

    final_metrics_df.to_csv("/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/XGB/XGB_metrics_summary.csv", index=False)


# dfs = [pd.read_csv(f"/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/GNN/cv_fold{i}_predictions_GraphRNA.csv") for i in range(10)]
# dfs = [pd.read_csv(f"/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs/cv_fold{i}_predictions_GraphRNA.csv") for i in range(10)]
# dfs = [pd.read_csv(f"/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/RF/cv_fold{i}_predictions_RandomForest.csv") for i in range(10)]
# dfs = [pd.read_csv(f"/sise/home/ronfay/Data_bacteria/graphNN/GraphRNA/outputs_mir/XGB/cv_fold{i}_predictions_XGBoost.csv") for i in range(10)]
# create_metric_df(dfs)



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


