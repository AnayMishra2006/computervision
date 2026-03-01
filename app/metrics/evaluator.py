# app/metrics/evaluator.py

class Evaluator:
    def __init__(self):
        self.tp = 0  # true positives
        self.fp = 0  # false positives
        self.tn = 0  # true negatives
        self.fn = 0  # false negatives

    def update(self, ground_truth, prediction):
        """
        ground_truth: 1 (waste), 0 (secure)
        prediction: 1 (waste), 0 (secure)
        """

        if ground_truth == 1 and prediction == 1:
            self.tp += 1
        elif ground_truth == 0 and prediction == 1:
            self.fp += 1
        elif ground_truth == 0 and prediction == 0:
            self.tn += 1
        elif ground_truth == 1 and prediction == 0:
            self.fn += 1

    def compute(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
        false_trigger_rate = self.fp / (self.tp + self.fp) if (self.tp + self.fp) else 0

        return {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "False Trigger Rate": false_trigger_rate,
            "TP": self.tp,
            "FP": self.fp,
            "TN": self.tn,
            "FN": self.fn,
        }