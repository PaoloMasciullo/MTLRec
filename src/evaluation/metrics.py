import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from src.evaluation.utils import mrr_score, ndcg_score
from torch.nn import functional as F


class Metric:
    name: str
    type = str

    def calculate(self, y_true, y_pred) -> float: ...

    def __call__(self, y_true, y_pred) -> float:
        return self.calculate(y_true, y_pred)


class MSE(Metric):
    def __init__(self):
        self.name = "Mean Squared Error"
        self.type = "metric"

    def calculate(self, y_true, y_pred) -> float:
        mse = np.mean((y_true - y_pred) ** 2)
        return float(mse)


class RMSE(Metric):
    def __init__(self):
        self.name = "Root Mean Squared Error"
        self.type = "metric"

    def calculate(self, y_true, y_pred) -> float:
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return float(rmse)


class R2Score(Metric):
    def __init__(self):
        self.name = "R2"  # Coefficient of determination
        self.type = "metric"

    def calculate(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Calculates the R2 score (coefficient of determination).
        Formula: 1 - (SS_res / SS_tot)
        Where:
        - SS_res is the sum of squared residuals
        - SS_tot is the total sum of squares
        """
        # Compute residual sum of squares (SS_res)
        ss_res = np.sum((y_true - y_pred) ** 2)

        # Compute total sum of squares (SS_tot)
        mean_y_true = np.mean(y_true)
        ss_tot = np.sum((y_true - mean_y_true) ** 2)

        # Compute R2 score
        r2 = 1 - (ss_res / ss_tot)
        return float(r2)


class AccuracyScore(Metric):
    def __init__(self):
        self.name = "Accuracy"
        self.type = "metric"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)  # multiclass-classification
        else:
            y_pred = np.round(y_pred)
        res = accuracy_score(y_true, y_pred)
        return float(res)


class F1Score(Metric):
    def __init__(self):
        self.name = "F1 Score"
        self.type = "metric"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)  # multiclass-classification
            res = f1_score(y_true, y_pred, average="weighted")  # multiclass classification
        else:
            y_pred = np.round(y_pred)
            res = f1_score(y_true, y_pred)
        return float(res)


class ConfusionMatrix(Metric):
    def __init__(self):
        self.name = "Confusion Matrix"
        self.type = "metric"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = np.round(y_pred)
        res = confusion_matrix(y_true, y_pred)
        return res


class AucScore(Metric):
    def __init__(self):
        self.name = "AUC"
        self.type = "metric"

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        res = roc_auc_score(y_true, y_pred)
        return float(res)


class AvgAucScore(Metric):
    def __init__(self):
        self.name = "AvgAUC"
        self.type = "group_metric"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                roc_auc_score(each_labels, each_preds)
                for each_labels, each_preds in zip(y_true, y_pred)
                if 0 < np.sum(each_labels) < len(each_labels)  # in case all negatives or all positives for a group
            ]
        )
        return float(res)


class MrrScore(Metric):
    def __init__(self):
        self.name = "MRR"
        self.type = "group_metric"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        mean_mrr = np.mean(
            [
                mrr_score(each_labels, each_preds)
                for each_labels, each_preds in zip(y_true, y_pred)
                if 0 < np.sum(each_labels) < len(each_labels)  # in case all negatives or all positives for a group
            ]
        )
        return float(mean_mrr)


class NdcgScore(Metric):
    def __init__(self, k: int):
        self.k = k
        self.name = f"NDCG@{k}"
        self.type = "group_metric"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                ndcg_score(each_labels, each_preds, self.k)
                for each_labels, each_preds in zip(y_true, y_pred)
                if 0 < np.sum(each_labels) < len(each_labels)  # in case all negatives or all positives for a group
            ]
        )
        return float(res)
