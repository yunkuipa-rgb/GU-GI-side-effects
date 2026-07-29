import pandas as pd
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def compute_metrics(y_true, y_pred, threshold=0.5):
    """
    Args:
        y_pred: Predicted probabilities or logits (float tensor or numpy array) of shape [batch_size]
        y_true: Ground truth binary labels (int tensor or numpy array) of shape [batch_size]
        threshold: Threshold for converting predicted probabilities to binary predictions
        
    Returns:
        accuracy, sensitivity (recall), specificity, auc_score
    """
    # Convert predictions to binary labels based on the threshold
    y_pred_labels = (y_pred > threshold)
    # print(np.asarray(y_pred_labels, np.float64).reshape(1, -1))
    # print(y_true.reshape(1, -1))

    print("GT  ", y_true.transpose().astype(np.int8))
    print("Pred", y_pred_labels.transpose().astype(np.int8))

    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    accuracy = (y_pred_labels == y_true).sum().item() / len(y_true)

    # True positives (TP): predicted positive and true positive
    TP = ((y_pred_labels == 1) & (y_true == 1)).sum().item()

    # True negatives (TN): predicted negative and true negative
    TN = ((y_pred_labels == 0) & (y_true == 0)).sum().item()

    # False positives (FP): predicted positive but true negative
    FP = ((y_pred_labels == 1) & (y_true == 0)).sum().item()

    # False negatives (FN): predicted negative but true positive
    FN = ((y_pred_labels == 0) & (y_true == 1)).sum().item()

    # Sensitivity / Recall: TP / (TP + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Specificity: TN / (TN + FP)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # AUC score: Use sklearn's roc_auc_score (requires probabilities not binary predictions)
    auc_score = roc_auc_score(y_true, y_pred)

    return accuracy, sensitivity, specificity, auc_score


def fft_data():
    data = np.load('combined_dvh_curves.npz')

    # Select a specific group (for example, 'Gy_O_Bldr_SIM')
    group_name = 'Gy_O_Trigone_Fx5'  # Replace with the actual group you want to use
    group_data = data[group_name]

    # Get the number of patients in this group (i.e., the number of columns in group_data)
    num_patients = group_data.shape[1]

    # Set up the plot for DVH curves
    plt.figure(figsize=(10, 6))

    # Loop over all patients and plot each DVH curve
    for i in range(num_patients):
        patient_data = group_data[:, i]  # Extract DVH curve for the current patient
        plt.plot(patient_data, label=f'Patient {i+1}', alpha=0.6)  # Add lines with transparency

    plt.title(f'All Patients\' DVH Curves for {group_name}')
    plt.xlabel('Dose Points')
    plt.ylabel('DVH Value')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))  # Put legend outside to avoid overlapping
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Alternatively, you can plot all FFTs in one plot:
    plt.figure(figsize=(10, 6))

    N = len(patient_data)
    n = np.arange(N)
    T = N/N
    freq = n/T 

    # Loop over all patients and plot FFT result for each
    for i in range(num_patients):
        patient_data = group_data[:, i]
        fft_result = np.fft.fft(patient_data)
        fft_freqs = freq

        # Plot the magnitude of the FFT result
        plt.plot(fft_freqs, np.abs(fft_result), label=f'Patient {i+1}', alpha=0.6)

    plt.title(f'All Patients\' FFT of DVH Curves for {group_name}')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.xlim(0, np.max(fft_freqs))  # Show only positive frequencies
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def find_num(string):
    pattern = r"\d*\.\d+"  # Pattern for matching floats
    sub_str = string[:10]
    match = re.search(pattern, sub_str)
    if match:
        return string[len(match.group()):]
    else:
        pattern = r"\d+"  # Pattern for matching floats
        match = re.search(pattern, sub_str)
        if match:
            return string[len(match.group()):]
        else:
            return string


def group_data() -> None:
    df = pd.read_excel('/Users/a14038/Documents/workspace/research_medical/dvh/data/DVH.xlsx')
    outcome_data = pd.read_excel('/Users/a14038/Documents/workspace/research_medical/dvh/data/OUTCOME.xlsx')
        
    df['FullName'] = df['Unnamed: 0'].apply(lambda x: find_num(x))  # Full name without numbers

    # Group the data based on 'FullName' excluding leading numbers
    grouped_df = df.groupby('FullName')

    dvh_curves = {}
    for group_name, group_data in grouped_df:
        # Combine all patient data into a single matrix
        # Extract all patient data (ignoring 'Unnamed: 0', 'FullName', and 'Organ' columns)
        patient_data = group_data.iloc[:, 1:-1].values  # Extract as a numpy array
            
        # Store the combined patient data under the group name
        dvh_curves[group_name] = patient_data

    outcome_data_cleaned = outcome_data.dropna()
    gu = outcome_data_cleaned['GU'].values
    gi = outcome_data_cleaned['GI'].values
    gu = np.where(gu >= 1, 1, 0)
    dvh_curves["GU"] = gu[np.newaxis, ...]
    dvh_curves["GI"] = gi[np.newaxis, ...]
    # Save all grouped data into a single .npz file
    np.savez('combined_dvh_curves.npz', **dvh_curves)


def group_data2() -> None:
    df = pd.read_excel('/Users/a14038/Documents/workspace/research_medical/dvh/data/DVH.xlsx')
    outcome_data = pd.read_excel('/Users/a14038/Documents/workspace/research_medical/dvh/data/OUTCOME.xlsx')

    # 10 features
    selected_f = ['41.2Gy_O_Urethra', '41.3Gy_O_Urethra', '41.4Gy_O_Urethra', '41.1Gy_O_Trigone', '41.2Gy_O_Trigone', \
                  '41.3Gy_O_Trigone', '41.4Gy_O_Trigone', '41.5Gy_O_Trigone', '0.4Gy(abs)_O_Rctm', '8.1Gy(abs)_O_Rctm']
    # selected_f = ['9.2Gy(abs)_O_Rctm', '9.3Gy(abs)_O_Rctm', '9.4Gy(abs)_O_Rctm', '9.5Gy(abs)_O_Rctm', \
    #               '9.6Gy(abs)_O_Rctm', '9.7Gy(abs)_O_Rctm', '9.8Gy(abs)_O_Rctm', '9.9Gy(abs)_O_Rctm', \
    #               '10Gy(abs)_O_Rctm', '10.1Gy(abs)_O_Rctm', '11.1Gy(abs)_O_Rctm', '11.2Gy(abs)_O_Rctm', \
    #               '11.3Gy(abs)_O_Rctm', '11.4Gy(abs)_O_Rctm', '11.5Gy(abs)_O_Rctm', '11.6Gy(abs)_O_Rctm', \
    #               '11.7Gy(abs)_O_Rctm', '11.8Gy(abs)_O_Rctm', '7.8Gy(abs)_O_Rctm_Wall', '7.9Gy(abs)_O_Rctm_Wall', \
    #               '8Gy(abs)_O_Rctm_Wall', '8.1Gy(abs)_O_Rctm_Wall', '8.2Gy(abs)_O_Rctm_Wall', '8.3Gy(abs)_O_Rctm_Wall', \
    #               '8.4Gy(abs)_O_Rctm_Wall', '8.5Gy(abs)_O_Rctm_Wall', '8.6Gy(abs)_O_Rctm_Wall', '8.7Gy(abs)_O_Rctm_Wall', \
    #               '8.8Gy(abs)_O_Rctm_Wall', '8.9Gy(abs)_O_Rctm_Wall']

    # selected_f = ['4.7Gy_O_Rctm_Wall', '41.5Gy_O_Urethra', '39.8Gy_O_Trigone',
    #                 '39.9Gy_O_Trigone', '40Gy_O_Trigone', '40.1Gy_O_Trigone',
    #                 '40.2Gy_O_Trigone', '40.3Gy_O_Trigone', '40.4Gy_O_Trigone',
    #                 '40.6Gy_O_Trigone', '40.8Gy_O_Trigone', '41Gy_O_Trigone',
    #                 '41.1Gy_O_Trigone', '41.2Gy_O_Trigone', '41.3Gy_O_Trigone',
    #                 '40.7Gy_O_Trigone', '1.3Gy(abs)_O_Rctm', '1.4Gy(abs)_O_Rctm',
    #                 '1.5Gy(abs)_O_Rctm', '1.6Gy(abs)_O_Rctm', '1.7Gy(abs)_O_Rctm',
    #                 '1.9Gy(abs)_O_Rctm', '0Gy(abs)_O_Rctm', '0.2Gy(abs)_O_Rctm',
    #                 '0.3Gy(abs)_O_Rctm', '0.4Gy(abs)_O_Rctm', '0.5Gy(abs)_O_Rctm',
    #                 '0.6Gy(abs)_O_Rctm', 'Volume(ml)_O_Rctm',
    #                 'Min Dose(Gy)_O_Urethra']

    dvh_curves = {}
    # Group the data based on 'FullName' excluding leading numbers
    for s_f in selected_f:
        d = df[df.iloc[:, 0].str.startswith(s_f)]
        idx = []
        for i, ds in enumerate(d.iloc[:, 0]):
            if "Wall" not in s_f and "Wall" in str(ds):
                print(s_f, str(ds))
                continue
            # print(d.iloc[:, 0])
            idx.append(i)
        d = d.iloc[idx, 1:]
        dvh_curves[s_f] = d

    outcome_data_cleaned = outcome_data.dropna()
    gu = outcome_data_cleaned['GU'].values
    gi = outcome_data_cleaned['GI'].values
    gu = np.where(gu >= 2, 1, 0)
    dvh_curves["GU"] = gu[np.newaxis, ...]
    dvh_curves["GI"] = gi[np.newaxis, ...]
    # Save all grouped data into a single .npz file
    np.savez('10_gu_2.npz', **dvh_curves)


def group_data3() -> None:
    df = pd.read_excel('/Users/a14038/Documents/workspace/research_medical/dvh/data/DVH.xlsx')
    outcome_data = pd.read_excel('/Users/a14038/Documents/workspace/research_medical/dvh/data/OUTCOME.xlsx')

    selected_f = ['22.7Gy_O_Rctm_Wall', '23.2Gy_O_Rctm_Wall', '41.1Gy_O_Urethra', '41.2Gy_O_Urethra', \
                  '41.3Gy_O_Urethra', '39.7Gy_O_Trigone', '9.8Gy_O_Trigone', '39.9Gy_O_Trigone', \
                  'Min Dose(Gy)_O_Urethra', 'Min Dose(Gy)_O_Trigone']
    # selected_f = ['38.3Gy_O_Bldr', '38.4Gy_O_Bldr', '38.5Gy_O_Bldr', '38.6Gy_O_Bldr', \
    #               '38.7Gy_O_Bldr', '38.8Gy_O_Bldr', '38.9Gy_O_Bldr', '39Gy_O_Bldr', '39.1Gy_O_Bldr', \
    #               '38.2Gy(abs)_O_Bldr_Wall', '38.3Gy(abs)_O_Bldr_Wall', '38.4Gy(abs)_O_Bldr_Wall', \
    #               '38.5Gy(abs)_O_Bldr_Wall', '38.6Gy(abs)_O_Bldr_Wall', '38.7Gy(abs)_O_Bldr_Wall', \
    #               '38.8Gy(abs)_O_Bldr_Wall',  '38.9Gy(abs)_O_Bldr_Wall', '39Gy(abs)_O_Bldr_Wall', \
    #               '39.1Gy(abs)_O_Bldr_Wall', '39.2Gy(abs)_O_Bldr_Wall', '39.3Gy(abs)_O_Bldr_Wall', \
    #               '37.5Gy(abs)_O_Bldr', '37.6Gy(abs)_O_Bldr', '37.7Gy(abs)_O_Bldr', '37.8Gy(abs)_O_Bldr', \
    #               '37.9Gy(abs)_O_Bldr', '38Gy(abs)_O_Bldr', '38.1Gy(abs)_O_Bldr', '38.2Gy(abs)_O_Bldr', \
    #               '38.3Gy(abs)_O_Bldr']

    dvh_curves = {}
    # Group the data based on 'FullName' excluding leading numbers
    for s_f in selected_f:
        d = df[df.iloc[:, 0].str.startswith(s_f)]
        if "Wall" not in s_f and "Wall" in d.iloc[0, 0]:
            continue
        print(d.iloc[0, 0])
        d = d.iloc[:, 1:]
        dvh_curves[s_f] = d

    outcome_data_cleaned = outcome_data.dropna()
    gu = outcome_data_cleaned['GU'].values
    gi = outcome_data_cleaned['GI'].values
    gu = np.where(gu >= 1, 1, 0)
    dvh_curves["GU"] = gu[np.newaxis, ...]
    dvh_curves["GI"] = gi[np.newaxis, ...]
    # Save all grouped data into a single .npz file
    np.savez('10_gi.npz', **dvh_curves)


# fft_data()
# group_data2()

grouped_dvh = np.load('/Users/a14038/Documents/workspace/research_medical/dvh/selected_dvh_curves_gi.npz')
print(len(grouped_dvh.keys()))
for key in grouped_dvh.keys():
    print(f"Key: {key}, value: {grouped_dvh[key].shape}")
# print(grouped_dvh["GU"])