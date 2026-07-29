import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report, make_scorer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold

import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

def specificity_scorer(estimator, X, y_true):
    y_pred = estimator.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract TN and FP
    tn = cm[0, 0]
    fp = cm[0, 1]
    
    # Compute specificity
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    return specificity

# special name: leave it as "None", ignore "downsample rate"
# type "L1" (lasso regression), "ANOVA", "rfe"
# k: number of features to be selected
class data:
    def __init__(self, special_name, downsample_rate=1, type="L1", k=10) -> None:
        dvh_data = pd.read_excel(r'./data/DVH.xlsx')
        outcome_data = pd.read_excel(r'./data/OUTCOME.xlsx')

        if special_name is not None:
            dvh_data = dvh_data[dvh_data.iloc[:, 0].str.contains(special_name, case=False, na=False)]

        # Clean data and align the patient IDs (assuming columns in dvh_data correspond to patient IDs)
        dvh_data_cleaned = dvh_data.iloc[:, 1:]  # Selecting the columns for patients 1-58
        outcome_data_cleaned = outcome_data.dropna()  # Remove rows with missing outcomes
        feature_names = np.asarray(dvh_data.iloc[:, 0].tolist())

        # Step 2: Data Combining
        # Combine DVH data with GU and GI outcomes for patients 1-58
        self.X = dvh_data_cleaned.T  # Transpose to have patients as rows and features as columns
        selector = VarianceThreshold(threshold=0)
        self.X = selector.fit_transform(self.X)
        features_idx = selector.get_support(indices=True)
        dvh_head = feature_names[features_idx]
        self.X = self.X[:, ::downsample_rate]
        dvh_head = dvh_head[::downsample_rate]

        # Feature selection
        self.y_gu = outcome_data_cleaned['GU'].values
        self.y_gi = outcome_data_cleaned['GI'].values
        ### note: you need to change here ####
        self.y_gu = np.where(self.y_gu >= 2, 1, 0)
        gu_idx, gi_idx = self.feature_selection(type, k)
        self.X_GU = self.X[:, gu_idx]
        self.X_GI = self.X[:, gi_idx]
        print("feature selected for GU")
        print(dvh_head[gu_idx])
        print("feature selected for GI")
        print(dvh_head[gi_idx])

    def get_feature(self, type, selector):
        if type == "GU":
            X_train, y_train = self.X, self.y_gu
        else:
            X_train, y_train = self.X, self.y_gi
        _ = selector.fit_transform(X_train, y_train)
        # X_test_selected = selector.transform(X_test)
        feature_idx = selector.get_support(indices=True)
        if hasattr(selector.estimator_, 'feature_importances_'):
            # For tree-based models (RandomForest, XGBoost, etc.)
            all_scores = selector.estimator_.feature_importances_
            selected_scores = all_scores[feature_idx]
        elif hasattr(selector.estimator_, 'coef_'):
            # For linear models (Lasso, Ridge, LogisticRegression, etc.)
            coef = selector.estimator_.coef_
            if coef.ndim > 1:  # Multi-class case
                all_scores = np.abs(coef).mean(axis=0)  # Average across classes
            else:
                all_scores = np.abs(coef)
            selected_scores = all_scores[feature_idx]
        else:
            print("Estimator doesn't have feature_importances_ or coef_ attribute")
            selected_scores = None
        print(selected_scores)
        return feature_idx

    def feature_selection(self, type, k):
        if type == "ANOVA":
            selector = SelectKBest(score_func=f_classif, k=k)
        elif type == "rfe":
            model = RandomForestClassifier(random_state=42)
            selector = RFE(estimator=model, n_features_to_select=k)
        elif type == "L1":
            model = LogisticRegression(C=1.0, penalty='l1', solver='liblinear', random_state=42)
            selector = SelectFromModel(model, prefit=False, max_features=k)

        gu_idx = self.get_feature("GU", selector)
        gi_idx = self.get_feature("GI", selector)

        return gu_idx, gi_idx


class predict:
    def __init__(self, n_estimators) -> None:
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.n_estimators = n_estimators

    def train(self, X, y):
        model = AdaBoostClassifier(n_estimators=self.n_estimators, random_state=42)

        acc = cross_val_score(model, X, y, cv=self.kf, scoring='accuracy')
        recall = cross_val_score(model, X, y, cv=self.kf, scoring='recall')
        specificity = cross_val_score(model, X, y, cv=self.kf, scoring=specificity_scorer)
        auc_scores = cross_val_score(model, X, y, cv=self.kf, scoring='roc_auc')
        print(f"Accuracy: {acc.mean()}")
        print(f"Sensitivity: {recall.mean()}")
        print(f"Specificity: {specificity.mean()}")
        print(f"AUC score: {auc_scores.mean()}")

    def evaluate(self, X, y, save_path):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.14, random_state=42)
        model = AdaBoostClassifier(n_estimators=250, random_state=42)

        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]

        # Compute ROC curve and AUC score
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='forestgreen', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve (Binary Classification)')
        # plt.legend(loc="lower right")
        
        # Save the figure if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)  # Save with high resolution (300 dpi)
            print(f"ROC curve saved to {save_path}")
            print()
        
        # Show the plot
        # plt.show()
        
        return roc_auc
        
