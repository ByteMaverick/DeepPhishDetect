import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, precision_recall_curve
)
import numpy as np

class MetricsCalculator:
    def __init__(self):
        pass

    def compute_metrics(self, y_true, y_pred, y_scores=None):
        """Compute and return a dictionary of metrics."""
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)

        if y_scores is not None:
            metrics['average_precision'] = average_precision_score(y_true, y_scores)

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """Plot a confusion matrix using seaborn."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=labels, yticklabels=labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_precision_recall_curve(self, y_true, y_scores):
        """Plot a precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.show()

    def evaluate(self, y_true, y_pred, y_scores=None):
        """Compute metrics and display visualizations."""
        metrics = self.compute_metrics(y_true, y_pred, y_scores)
        print("Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        # Plot the confusion matrix
        self.plot_confusion_matrix(y_true, y_pred, labels=["Benign", "Phishing"])

        # Plot the precision-recall curve if scores are provided
        if y_scores is not None:
            self.plot_precision_recall_curve(y_true, y_scores)
