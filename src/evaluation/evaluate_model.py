import numpy as np

from src.evaluation.metrics import Metric
from src.evaluation.utils import group_by_id


class ModelEvaluator:
    """
        >>> y_true = [[1, 0, 0], [1, 1, 0], [1, 0, 0, 0]]
        >>> y_pred = [[0.2, 0.3, 0.5], [0.18, 0.7, 0.1], [0.18, 0.2, 0.1, 0.1]]

        >>> met_eval = ModelEvaluator(
                labels=y_true,
                predictions=y_pred,
                metric_functions=[
                    AucScore(),
                    MrrScore(),
                    NdcgScore(k=5),
                    NdcgScore(k=10),
                    LogLossScore(),
                    RootMeanSquaredError(),
                    AccuracyScore(threshold=0.5),
                    F1Score(threshold=0.5),
                ],
            )
        >>> met_eval.evaluate()
        {
            "auc": 0.5555555555555556,
            "mrr": 0.5277777777777778,
            "ndcg@5": 0.7103099178571526,
            "ndcg@10": 0.7103099178571526,
            "logloss": 0.716399020295845,
            "rmse": 0.5022870658128165
            "accuracy": 0.5833333333333334,
            "f1": 0.2222222222222222
        }
        """

    def __init__(
            self,
            metric_functions: list[Metric],
    ):
        self.metric_functions = metric_functions
        self.evaluations = dict()

    def evaluate(self,
                 labels: list[np.ndarray],
                 predictions: list[np.ndarray],
                 group_ids: list[np.ndarray] = None
                 ) -> dict:

        for metric_function in self.metric_functions:
            if metric_function.type == "group_metric":
                assert group_ids is not None, "group_index is required."
                # Group outputs and labels
                grouped_labels, grouped_predictions = group_by_id(group_ids, predictions, labels)
                self.evaluations.update({metric_function.name: metric_function(grouped_labels, grouped_predictions)})
            else:
                self.evaluations.update({metric_function.name: metric_function(labels, predictions)})
        return self.evaluations
