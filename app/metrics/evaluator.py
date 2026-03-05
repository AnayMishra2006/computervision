# app/metrics/evaluator.py
"""
Online metrics evaluator for the waste detection pipeline.

Since we don't have a labelled test dataset at demo time, we derive
an *approximate* ground-truth automatically:

    ground_truth = 1  when (room is empty) AND (appliance is ON)
    ground_truth = 0  otherwise

This gives us live Precision / Recall / F1 numbers on the dashboard
so judges can see the system isn't just always-ON or always-OFF.

Note: this is a *running* (online) evaluator – it accumulates counts
since startup.  Restart the process to reset the metrics.
"""


class Evaluator:
    """
    Accumulates confusion-matrix counts and computes detection metrics.

    Attributes
    ----------
    tp, fp, tn, fn : int  running confusion matrix counts
    """

    def __init__(self):
        self.tp = 0  # correctly raised waste alert
        self.fp = 0  # false alarm (alert raised when room was occupied)
        self.tn = 0  # correctly silent (room occupied, no alert)
        self.fn = 0  # missed waste (room empty+lights on but no alert yet)

    def update(self, ground_truth: int, prediction: int) -> None:
        """
        Record one frame's ground_truth / prediction pair.

        Parameters
        ----------
        ground_truth : int  1 = waste condition, 0 = secure
        prediction   : int  1 = waste alert active, 0 = no alert
        """
        if ground_truth == 1 and prediction == 1:
            self.tp += 1
        elif ground_truth == 0 and prediction == 1:
            self.fp += 1
        elif ground_truth == 0 and prediction == 0:
            self.tn += 1
        elif ground_truth == 1 and prediction == 0:
            self.fn += 1

    def compute(self) -> dict:
        """
        Compute and return current metrics.

        Returns a dict with keys:
            Precision, Recall, F1 Score, False Trigger Rate,
            TP, FP, TN, FN
        """
        precision = (
            self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        )
        recall = (
            self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        )
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) else 0.0
        )
        false_trigger_rate = (
            self.fp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        )

        return {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
            "False Trigger Rate": round(false_trigger_rate, 4),
            "TP": self.tp,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
        }