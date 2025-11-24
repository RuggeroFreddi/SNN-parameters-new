import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def cross_validation_rf(dataset: pd.DataFrame, n_splits: int = 10) -> Tuple[float, float, float, float, float, float]:
    features = dataset.drop(columns=["label"]).to_numpy()
    labels = dataset["label"].to_numpy()

    stratified_kfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    random_forest_params = {
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_split": 12,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "class_weight": None,
        "oob_score": False,
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    }

    fold_accuracies = []
    fold_f1s = []
    fold_mccs = []

    for fold_index, (train_indices, test_indices) in enumerate(
        stratified_kfold.split(features, labels), start=1
    ):
        x_train = features[train_indices]
        x_test = features[test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]
        
        model = make_pipeline(
            #SelectKBest(f_classif, k=200),
            RandomForestClassifier(**random_forest_params)
        )

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        mcc = matthews_corrcoef(y_test, y_pred)  # supporta anche multi-classe

        fold_accuracies.append(accuracy)
        fold_f1s.append(f1_macro)
        fold_mccs.append(mcc)

        print(f"[CV] Fold {fold_index:2d} – acc: {accuracy:.4f} – macro-F1: {f1_macro:.4f} – MCC: {mcc:.4f}")

    mean_accuracy = float(np.mean(fold_accuracies))
    std_accuracy = float(np.std(fold_accuracies))

    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s))

    mean_mcc = float(np.mean(fold_mccs))
    std_mcc = float(np.std(fold_mccs))

    print("\n=== Cross-validation Random Forest summary ===")
    print(f"Mean accuracy:   {mean_accuracy:.4f}  ± {std_accuracy:.4f}")
    print(f"Mean macro-F1:   {mean_f1:.4f}      ± {std_f1:.4f}")
    print(f"Mean MCC:        {mean_mcc:.4f}      ± {std_mcc:.4f}")
    print(f"Fold accuracies: {[round(a, 4) for a in fold_accuracies]}")
    print(f"Fold macro-F1:   {[round(f, 4) for f in fold_f1s]}")
    print(f"Fold MCC:        {[round(m, 4) for m in fold_mccs]}")

    return mean_accuracy, std_accuracy, mean_f1, std_f1, mean_mcc, std_mcc

def cross_validation_lr(dataset: pd.DataFrame, n_splits: int = 10) -> Tuple[float, float, float, float, float, float]:
    features = dataset.drop(columns=["label"]).to_numpy()
    labels = dataset["label"].to_numpy()

    stratified_kfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    logistic_params = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "saga",
        "multi_class": "multinomial",
        "max_iter": 3000,
        "n_jobs": -1,
        "random_state": 42,
    }

    fold_accuracies = []
    fold_f1s = []
    fold_mccs = []

    for fold_index, (train_indices, test_indices) in enumerate(
        stratified_kfold.split(features, labels), start=1
    ):
        x_train = features[train_indices]
        x_test = features[test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]

        model = make_pipeline(
            #SelectKBest(f_classif, k=200),
            StandardScaler(),
            LogisticRegression(**logistic_params),
        )

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        mcc = matthews_corrcoef(y_test, y_pred)

        fold_accuracies.append(accuracy)
        fold_f1s.append(f1_macro)
        fold_mccs.append(mcc)

        print(f"[CV] Fold {fold_index:2d} – acc: {accuracy:.4f} – macro-F1: {f1_macro:.4f} – MCC: {mcc:.4f}")

    mean_accuracy = float(np.mean(fold_accuracies))
    std_accuracy = float(np.std(fold_accuracies))

    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s))

    mean_mcc = float(np.mean(fold_mccs))
    std_mcc = float(np.std(fold_mccs))

    print("\n=== Cross-validation Logistic Regressor summary ===")
    print(f"Mean accuracy:   {mean_accuracy:.4f}  ± {std_accuracy:.4f}")
    print(f"Mean macro-F1:   {mean_f1:.4f}      ± {std_f1:.4f}")
    print(f"Mean MCC:        {mean_mcc:.4f}      ± {std_mcc:.4f}")
    print(f"Fold accuracies: {[round(a, 4) for a in fold_accuracies]}")
    print(f"Fold macro-F1:   {[round(f, 4) for f in fold_f1s]}")
    print(f"Fold MCC:        {[round(m, 4) for m in fold_mccs]}")

    return mean_accuracy, std_accuracy, mean_f1, std_f1, mean_mcc, std_mcc

def cross_validation_slp(dataset: pd.DataFrame, n_splits: int = 10) -> Tuple[float, float, float, float, float, float]:
    # separo feature e label come nelle altre funzioni
    features = dataset.drop(columns=["label"]).to_numpy()
    labels = dataset["label"].to_numpy()

    stratified_kfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    # iperparametri di base per il perceptron
    perceptron_params = {
        "penalty": None,          # perceptron "classico"
        "alpha": 0.0001,
        "max_iter": 3000,
        "tol": 1e-3,
        "shuffle": True,
        "random_state": 42,
        "n_jobs": -1,
    }

    fold_accuracies = []
    fold_f1s = []
    fold_mccs = []

    for fold_index, (train_indices, test_indices) in enumerate(
        stratified_kfold.split(features, labels), start=1
    ):
        x_train = features[train_indices]
        x_test = features[test_indices]
        y_train = labels[train_indices]
        y_test = labels[test_indices]

        model = make_pipeline(
            #SelectKBest(f_classif, k=200),
            StandardScaler(),
            Perceptron(**perceptron_params),
        )

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        mcc = matthews_corrcoef(y_test, y_pred)

        fold_accuracies.append(accuracy)
        fold_f1s.append(f1_macro)
        fold_mccs.append(mcc)

        print(f"[CV] Fold {fold_index:2d} – acc: {accuracy:.4f} – macro-F1: {f1_macro:.4f} – MCC: {mcc:.4f}")

    mean_accuracy = float(np.mean(fold_accuracies))
    std_accuracy = float(np.std(fold_accuracies))

    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s))

    mean_mcc = float(np.mean(fold_mccs))
    std_mcc = float(np.std(fold_mccs))

    print("\n=== Cross-validation summary Single Layer Perceptron ===")
    print(f"Mean accuracy:   {mean_accuracy:.4f}  ± {std_accuracy:.4f}")
    print(f"Mean macro-F1:   {mean_f1:.4f}      ± {std_f1:.4f}")
    print(f"Mean MCC:        {mean_mcc:.4f}      ± {std_mcc:.4f}")
    print(f"Fold accuracies: {[round(a, 4) for a in fold_accuracies]}")
    print(f"Fold macro-F1:   {[round(f, 4) for f in fold_f1s]}")
    print(f"Fold MCC:        {[round(m, 4) for m in fold_mccs]}")

    return mean_accuracy, std_accuracy, mean_f1, std_f1, mean_mcc, std_mcc
