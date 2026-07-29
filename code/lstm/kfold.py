import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold

def norm(x):
    return (x - x.mean()) / x.std()
    

#"file_name", tx_idx: list type, 0: SIM, 1: Fx1, 2: Fx2, 3:Fx3, 4: Fx4, 5: Fx5
class SelectedDVHDataset(Dataset):
    def __init__(self, patient_ids=None, clip_rate=400, file_name=None, tx_idx=None):
        
        self.data = np.load(file_name)
        pred_gu = False
        if "gu" in file_name:
            pred_gu = True
            
        self.data_collection = []
        if patient_ids is None:
            patient_ids = np.arange(self.data["GU"].shape[1])
        for key in self.data.keys():
            if "GI" in key or "GU" in key:
                continue
            d = self.data[key]
            data_s = d[:, patient_ids] / 100
            time_steps = data_s.shape[0]
            if time_steps == 1:
                self.data_collection.append(np.repeat(data_s, 6, axis=0))
            elif time_steps == 6:
                self.data_collection.append(data_s)
            else:
                self.data_collection.append(data_s[0::2, :])
                self.data_collection.append(data_s[1::2, :])
        self.data_collection = np.stack(self.data_collection, axis=1) # [timepoints, num features, patients]
        # reverse
        # self.data_collection = self.data_collection[:3, :, :]
        idx = tx_idx
        self.data_collection = self.data_collection[idx]
        # print(self.data_collection.shape)
        #np.random.shuffle(self.data_collection)
        self.data_collection = np.transpose(self.data_collection, (2, 0, 1))
        self.y = np.asarray(self.data["GU"][:, patient_ids], np.bool_) if pred_gu else np.asarray(self.data["GI"][:, patient_ids], np.bool_)
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        # Return the DVH data for a specific patient and the corresponding patient ID
        x = torch.FloatTensor(self.data_collection[idx, ...])
        y1 = torch.FloatTensor(self.y[:, idx])
        return x, y1


# Function for setting up k-fold cross-validation splits
def get_k_fold_loaders(dataset, k=5, batch_size=4, shuffle=True, pred_gu=True):
    """
    Args:
        dataset: The dataset object containing all patient data
        k: Number of folds
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the dataset before splitting into folds

    Returns:
        A generator that yields (train_loader, test_loader) pairs for each fold
    """
    # Create a KFold object
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    folds = list()
    data_tensor = dataset.data_collection
    labels_tensor = dataset.y.transpose(1, 0)

    folds = list(skf.split(data_tensor, labels_tensor))

    for fold, (train_idx, test_idx) in enumerate(folds):
        # print("Fold {} / {}".format(fold + 1, k))
        # Subset the dataset for training and testing
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        # DataLoader for training and testing
        train_targets = np.asarray([train_subset[i][1].numpy() for i in range(len(train_subset))], np.bool_)[:, 0]
        class_sample_count = np.bincount(train_targets)
        class_weights = 1. / class_sample_count
        sample_weights = np.where(train_targets == 0, class_weights[0], class_weights[1])

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights) * 2, replacement=True)
        
        # DataLoader with sampler for balanced training
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        yield train_loader, test_loader


