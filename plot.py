import pandas as pd
import matplotlib.pyplot as plt

# Load the two data files without headers
file1_path = '/Users/a14038/Documents/workspace/research_medical/dvh/dvh_data/GU Grade 0-ANON69329__DVH_NormalizedVolume_AbsoluteDose.csv'
file1_path = '/Users/a14038/Documents/workspace/research_medical/dvh/dvh_data/GU Grade 2-ANON63638__DVH_NormalizedVolume_AbsoluteDose.csv'
# file1_path = '/Users/a14038/Documents/workspace/research_medical/dvh/dvh_data/GU Grade 0-ANON69329_i1Prst_Fx1Delivered, I1PRST_FX1DELIVE (scaled)_DVH_NormalizedVolume_AbsoluteDose.csv'
# file1_path = '/Users/a14038/Documents/workspace/research_medical/dvh/dvh_data/GU Grade 0-ANON69329_i1Prst_Fx2Delivered, I1PRST_FX2DELIVE (scaled)_DVH_NormalizedVolume_AbsoluteDose.csv'
# file1_path = 'dvh_data/GU Grade 0-ANON69329_i1Prst_Fx3Delivered, I1PRST_FX3DELIVE (scaled)_DVH_NormalizedVolume_AbsoluteDose.csv'
# file1_path = 'dvh_data/GU Grade 0-ANON69329_i1Prst_Fx4Delivered, I1PRST_FX4DELIVE (scaled)_DVH_NormalizedVolume_AbsoluteDose.csv'
# file1_path = 'dvh_data/GU Grade 0-ANON69329_i1Prst_Fx5Delivered, I1PRST_FX5DELIVE (scaled)_DVH_NormalizedVolume_AbsoluteDose.csv'
# file1_path = 'dvh_data/GU Grade 2-ANON63638_i1Prst_Fx1Delivered, I1PRST_FX1DELIVE (scaled)_DVH_NormalizedVolume_AbsoluteDose.csv'
# file1_path = 'dvh_data/GU Grade 2-ANON63638_i1Prst_Fx2Delivered, I1PRST_FX2DELIVE (scaled)_DVH_NormalizedVolume_AbsoluteDose.csv'
# file1_path = 'dvh_data/GU Grade 2-ANON63638_i1Prst_Fx3Delivered, I1PRST_FX3DELIVE (scaled)_DVH_NormalizedVolume_AbsoluteDose.csv'
# file1_path = 'dvh_data/GU Grade 2-ANON63638_i1Prst_Fx4Delivered, I1PRST_FX4DELIVE (scaled)_DVH_NormalizedVolume_AbsoluteDose.csv'
# file1_path = 'dvh_data/GU Grade 2-ANON63638_i1Prst_Fx5Delivered, I1PRST_FX5DELIVE (scaled)_DVH_NormalizedVolume_AbsoluteDose.csv'

data1 = pd.read_csv(file1_path, header=None)
# data2 = pd.read_csv(file2_path, header=None)

# Extract organ names from the second row
organ_names1 = data1.iloc[1, 1:].values  # Skip the first column (dose)
# organ_names2 = data2.iloc[1, 1:].values  # Skip the first column (dose)

# Create a dictionary of column indices for the organs of interest
visited = []
def get_organ_columns(organ_names, organs_of_interest):
    organ_columns = {}
    for i, name in enumerate(organ_names):
        if isinstance(name, str):
            for organ in organs_of_interest:
                if organ in name and organ not in visited:
                    visited.append(organ)
                    organ_columns[organ] = i + 1  # Adjust index to match column position
    return organ_columns

# Define organs of interest
organs_of_interest = ["O_Bldr", "O_Rctm", "C_Prst"]
organ_dict = {"O_Bldr": "Bladder", "O_Rctm": "Rectum", "C_Prst": "Prostate", "Urethra": "Urethra", "OUrethra": "Urethra"}
#organs_of_interest = ["O_Bldr", "O_Rctm", "O_Trigone"]
#organ_dict = {"O_Bldr": "Bladder", "O_Rctm": "Rectum", "O_Bldr_Wall": "Bladder Wall", "O_Rctm_Wall": "Rectum Wall", "O_Trigone": "Trigone"}


# Extract column indices for the organs of interest
organ_columns1 = get_organ_columns(organ_names1, organs_of_interest)
# organ_columns2 = get_organ_columns(organ_names2, organs_of_interest)

print("Data1 columns:", data1.shape[1])
# print("Data2 columns:", data2.shape[1])

# Function to plot DVH for a patient
def plot_dvh_for_patient(data, organ_columns, linestyle, label_prefix, colors):
    dose_column = 0  # Dose is in the first column
    for i, (organ, col) in enumerate(organ_columns.items()):
        print(organ, col)
        plt.plot(
            data.iloc[2:, dose_column].astype(float),  # Dose values
            data.iloc[2:, col].astype(float),  # Volume values
            linestyle=linestyle,
            color=colors[organ],
            label=f"{organ_dict[organ]}"
        )

# Define colors for the organs
colors = {"O_Bldr": "blue", "O_Rctm": "orange", "C_Prst": "red"}

# Plot the DVH curves
plt.figure(figsize=(10, 6))
# plot_dvh_for_patient(data2, organ_columns1, linestyle='-', label_prefix="GU=0", colors=colors)
plot_dvh_for_patient(data1, organ_columns1, linestyle='-', label_prefix="GU=0", colors=colors)

# Customize the plot
plt.title("DVH Curves for Selected Organs")
plt.xlabel("Dose (Gy)")
plt.ylabel("Volume (%)")
plt.legend()
plt.grid(True)
plt.savefig('dvh_curves.png', dpi=300)  # Save with high resolution (300 DPI)
plt.show()