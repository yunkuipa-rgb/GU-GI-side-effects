# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report, make_scorer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils import data, predict


def combine_test(model, data, file_name):
    print("Using " + file_name + " data, GU")
    model.train(data.X_GU, d_all.y_gu)
    model.evaluate(data.X_GU, data.y_gu, file_name + "_GU.JPG")
    print()
    print("Using " + file_name + " data, GI")
    # model.n_estimators = model.n_estimators * 2
    model.train(data.X_GI, data.y_gi)
    model.evaluate(data.X_GI, data.y_gi, file_name + "_GI.JPG")

    print("================")

k = 30
d_all = data(None, 1, k=k)
d_sim = data('SIM', 1, k=k)
d_fx = data('Fx', 1, k=k)

n_estimators=200
p_all = predict(n_estimators)
p_sim = predict(n_estimators)
p_fx = predict(n_estimators)

combine_test(p_all, d_all, "all")
combine_test(p_sim, d_sim, "sim")
combine_test(p_fx, d_fx, "fx")





