import pandas as pd
import numpy as np
import re

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
        

# for GU >= 1
def group_data1() -> None:
    df = pd.read_excel(r'./data/DVH.xlsx')
    outcome_data = pd.read_excel(r'./data/OUTCOME.xlsx')

    # 30 features GU >= 1
    selected_f = ['9.2Gy(abs)_O_Rctm', '9.3Gy(abs)_O_Rctm', '9.4Gy(abs)_O_Rctm', '9.5Gy(abs)_O_Rctm', \
                  '9.6Gy(abs)_O_Rctm', '9.7Gy(abs)_O_Rctm', '9.8Gy(abs)_O_Rctm', '9.9Gy(abs)_O_Rctm', \
                  '10Gy(abs)_O_Rctm', '10.1Gy(abs)_O_Rctm', '11.1Gy(abs)_O_Rctm', '11.2Gy(abs)_O_Rctm', \
                  '11.3Gy(abs)_O_Rctm', '11.4Gy(abs)_O_Rctm', '11.5Gy(abs)_O_Rctm', '11.6Gy(abs)_O_Rctm', \
                  '11.7Gy(abs)_O_Rctm', '11.8Gy(abs)_O_Rctm', '7.8Gy(abs)_O_Rctm_Wall', '7.9Gy(abs)_O_Rctm_Wall', \
                  '8Gy(abs)_O_Rctm_Wall', '8.1Gy(abs)_O_Rctm_Wall', '8.2Gy(abs)_O_Rctm_Wall', '8.3Gy(abs)_O_Rctm_Wall', \
                  '8.4Gy(abs)_O_Rctm_Wall', '8.5Gy(abs)_O_Rctm_Wall', '8.6Gy(abs)_O_Rctm_Wall', '8.7Gy(abs)_O_Rctm_Wall', \
                  '8.8Gy(abs)_O_Rctm_Wall', '8.9Gy(abs)_O_Rctm_Wall']

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
    gu = np.where(gu >= 1, 1, 0)
    dvh_curves["GU"] = gu[np.newaxis, ...]
    dvh_curves["GI"] = gi[np.newaxis, ...]
    # Save all grouped data into a single .npz file
    np.savez('selected_dvh_curves_gu_1.npz', **dvh_curves)

# for GU >= 2
def group_data2() -> None:
    df = pd.read_excel(r'./data/DVH.xlsx')
    outcome_data = pd.read_excel(r'./data/OUTCOME.xlsx')

    # 30 features GU >= 2
    selected_f = ['4.7Gy_O_Rctm_Wall', '41.5Gy_O_Urethra', '39.8Gy_O_Trigone',
                    '39.9Gy_O_Trigone', '40Gy_O_Trigone', '40.1Gy_O_Trigone',
                    '40.2Gy_O_Trigone', '40.3Gy_O_Trigone', '40.4Gy_O_Trigone',
                    '40.6Gy_O_Trigone', '40.8Gy_O_Trigone', '41Gy_O_Trigone',
                    '41.1Gy_O_Trigone', '41.2Gy_O_Trigone', '41.3Gy_O_Trigone',
                    '40.7Gy_O_Trigone', '1.3Gy(abs)_O_Rctm', '1.4Gy(abs)_O_Rctm',
                    '1.5Gy(abs)_O_Rctm', '1.6Gy(abs)_O_Rctm', '1.7Gy(abs)_O_Rctm',
                    '1.9Gy(abs)_O_Rctm', '0Gy(abs)_O_Rctm', '0.2Gy(abs)_O_Rctm',
                    '0.3Gy(abs)_O_Rctm', '0.4Gy(abs)_O_Rctm', '0.5Gy(abs)_O_Rctm',
                    '0.6Gy(abs)_O_Rctm', 'Volume(ml)_O_Rctm',
                    'Min Dose(Gy)_O_Urethra']

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
    np.savez('selected_dvh_curves_gu_2.npz', **dvh_curves)

# For GI
def group_data3() -> None:
    df = pd.read_excel(r'./data/DVH.xlsx')
    outcome_data = pd.read_excel(r'./data/OUTCOME.xlsx')

    selected_f = ['38.3Gy_O_Bldr', '38.4Gy_O_Bldr', '38.5Gy_O_Bldr', '38.6Gy_O_Bldr', \
                  '38.7Gy_O_Bldr', '38.8Gy_O_Bldr', '38.9Gy_O_Bldr', '39Gy_O_Bldr', '39.1Gy_O_Bldr', \
                  '38.2Gy(abs)_O_Bldr_Wall', '38.3Gy(abs)_O_Bldr_Wall', '38.4Gy(abs)_O_Bldr_Wall', \
                  '38.5Gy(abs)_O_Bldr_Wall', '38.6Gy(abs)_O_Bldr_Wall', '38.7Gy(abs)_O_Bldr_Wall', \
                  '38.8Gy(abs)_O_Bldr_Wall',  '38.9Gy(abs)_O_Bldr_Wall', '39Gy(abs)_O_Bldr_Wall', \
                  '39.1Gy(abs)_O_Bldr_Wall', '39.2Gy(abs)_O_Bldr_Wall', '39.3Gy(abs)_O_Bldr_Wall', \
                  '37.5Gy(abs)_O_Bldr', '37.6Gy(abs)_O_Bldr', '37.7Gy(abs)_O_Bldr', '37.8Gy(abs)_O_Bldr', \
                  '37.9Gy(abs)_O_Bldr', '38Gy(abs)_O_Bldr', '38.1Gy(abs)_O_Bldr', '38.2Gy(abs)_O_Bldr', \
                  '38.3Gy(abs)_O_Bldr']

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
    gu = np.where(gu >= 2, 1, 0)
    dvh_curves["GU"] = gu[np.newaxis, ...]
    dvh_curves["GI"] = gi[np.newaxis, ...]
    # Save all grouped data into a single .npz file
    np.savez('selected_dvh_curves_gi.npz', **dvh_curves)


group_data1()
group_data2()
group_data3()

grouped_dvh = np.load('selected_dvh_curves_gu_2.npz')
for key in grouped_dvh.keys():
    print(f"Key: {key}, value: {grouped_dvh[key].shape}")
print(grouped_dvh["GU"])