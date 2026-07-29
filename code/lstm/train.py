from lstm import LSTMClassifier2

import torch
import numpy as np
from kfold import SelectedDVHDataset, get_k_fold_loaders
from utils import compute_metrics

device = "cpu"

num_epochs = 300 # 500 for gu, 300 for gi
batch_size = 512 # 32 for gu, 512 for gi; SIM+FX 32, FX only 64
k = 5
N = 69
n_out = 1
# 5e-4
lr = 5e-4 # 1e-3 for gu, 5e-4 for gi
weight_decay = 1e-8
criterion = torch.nn.BCELoss()


def train_model(dataset, batch_size, max_epochs, k, pred_gu, fx_only):
    accuracy, sensitivity, specificity, auc_score = 0, 0, 0, 0
    tps = dataset.data_collection.shape[-1]
    pred_list = []
    gt_list = []
    for train_loader, test_loader in get_k_fold_loaders(dataset, k=k, batch_size=batch_size):
        # gu model
        if pred_gu:
            model = LSTMClassifier2(tps, 128, 128, n_out).to(device)
        # gi model
        else:
            model = LSTMClassifier2(tps, 192, 192, n_out).to(device)

        if fx_only:
            model = LSTMClassifier2(tps, 192, 192, n_out).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Iterate through the folds
        for epoch in range(max_epochs):
            total_loss = 0
            for dvh_data, y in train_loader:
                # Get DVH data and patient IDs from the batch
                dvh_data, y = dvh_data.to(device), y.to(device)
                model.zero_grad()
                print(dvh_data.shape)
                pred = model(dvh_data)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss
            if (epoch + 1) % 10 == 0:
                print("Train loss: {}".format(total_loss.data.float() / len(train_loader)))
        acc, sen, spe, auc, gt, pred = evaluate_validation_set(model, test_loader)
        accuracy += acc
        sensitivity += sen
        specificity += spe
        auc_score += auc
        pred_list.append(pred)
        gt_list.append(gt)
    
    # print(f"Accuracy: {accuracy / k}")
    # print(f"Sensitivity: {sensitivity / k}")
    # print(f"Specificity: {specificity / k}")
    print(f"AUC score: {auc_score / k}")
    return np.concatenate(pred_list), np.concatenate(gt_list)


def evaluate_validation_set(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    for dvh_data, y in test_loader:
        # Get DVH data and patient IDs from the batch
        dvh_data, y = dvh_data.to(device), y.to(device)
        pred = model(dvh_data)
        y_true += list(y.detach().cpu().numpy())
        y_pred += list(pred.detach().cpu().numpy())
    accuracy, sensitivity, specificity, auc_score = compute_metrics(np.asarray(y_true), np.asarray(y_pred))
    model.train()
    return accuracy, sensitivity, specificity, auc_score, np.asarray(y_true), np.asarray(y_pred)


dataset_gu = SelectedDVHDataset(file_name='/Users/a14038/Documents/workspace/research_medical/dvh/selected_dvh_curves_gi.npz', tx_idx=[0, 1, 2])
dataset_gi = SelectedDVHDataset(file_name='/Users/a14038/Documents/workspace/research_medical/dvh/selected_dvh_curves_gi.npz', tx_idx=[0, 1, 2])
# print("GU")
# train_model(dataset_gu, batch_size, num_epochs, k, pred_gu=True, fx_only=False)
print("GI")
train_model(dataset_gi, batch_size, num_epochs, k, pred_gu=False, fx_only=False)