"""
Model evaluation module with comprehensive metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, Optional
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not installed. Plotting disabled.")

sys.path.append(str(Path(__file__).parent.parent.parent))


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self):
        """Initialize evaluator"""
        pass

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                y_pred_proba: Optional[np.ndarray] = None,
                model_name: str = "Model") -> Dict:
        """
        Comprehensive model evaluation

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of model for display

        Returns:
            Dictionary with all metrics
        """
        # Filter out invalid predictions (from LSTM padding)
        valid_mask = y_pred != -1
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        if y_pred_proba is not None:
            y_pred_proba = y_pred_proba[valid_mask]

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value

        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'npv': npv,
            'confusion_matrix': cm,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'n_samples': len(y_true),
            'class_distribution': {
                'class_0': (y_true == 0).sum(),
                'class_1': (y_true == 1).sum()
            }
        }

        # ROC AUC if probabilities provided
        if y_pred_proba is not None:
            if y_pred_proba.ndim == 2:
                # Extract probability of positive class
                y_pred_proba = y_pred_proba[:, 1]

            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
                metrics['roc_auc'] = roc_auc
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC: {e}")

        return metrics

    def print_metrics(self, metrics: Dict):
        """
        Print metrics in formatted way

        Args:
            metrics: Dictionary of metrics from evaluate()
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION: {metrics['model_name']}")
        print(f"{'='*60}")

        print(f"\nSamples: {metrics['n_samples']}")
        print(f"Class distribution: {metrics['class_distribution']}")

        print(f"\n{'Metric':<20} {'Value':<10}")
        print(f"{'-'*30}")
        print(f"{'Accuracy':<20} {metrics['accuracy']:.4f}")
        print(f"{'Precision':<20} {metrics['precision']:.4f}")
        print(f"{'Recall (Sensitivity)':<20} {metrics['recall']:.4f}")
        print(f"{'Specificity':<20} {metrics['specificity']:.4f}")
        print(f"{'F1 Score':<20} {metrics['f1_score']:.4f}")

        if 'roc_auc' in metrics:
            print(f"{'ROC AUC':<20} {metrics['roc_auc']:.4f}")

        print(f"\n{'Confusion Matrix':<20}")
        print(f"{'-'*30}")
        cm = metrics['confusion_matrix']
        print(f"                 Predicted")
        print(f"               0         1")
        print(f"Actual  0   {cm[0,0]:4d}     {cm[0,1]:4d}")
        print(f"        1   {cm[1,0]:4d}     {cm[1,1]:4d}")

        print(f"\nTP: {metrics['true_positives']}, "
              f"TN: {metrics['true_negatives']}, "
              f"FP: {metrics['false_positives']}, "
              f"FN: {metrics['false_negatives']}")

    def compare_models(self, metrics_list: list):
        """
        Compare multiple models

        Args:
            metrics_list: List of metric dictionaries
        """
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")

        # Create comparison DataFrame
        comparison = pd.DataFrame([
            {
                'Model': m['model_name'],
                'Accuracy': m['accuracy'],
                'Precision': m['precision'],
                'Recall': m['recall'],
                'F1': m['f1_score'],
                'ROC AUC': m.get('roc_auc', np.nan)
            }
            for m in metrics_list
        ])

        print(f"\n{comparison.to_string(index=False)}")

        # Highlight best model
        print(f"\n{'Best Models:'}")
        print(f"  Accuracy:  {comparison.loc[comparison['Accuracy'].idxmax(), 'Model']}")
        print(f"  Precision: {comparison.loc[comparison['Precision'].idxmax(), 'Model']}")
        print(f"  Recall:    {comparison.loc[comparison['Recall'].idxmax(), 'Model']}")
        print(f"  F1 Score:  {comparison.loc[comparison['F1'].idxmax(), 'Model']}")

        if not comparison['ROC AUC'].isna().all():
            print(f"  ROC AUC:   {comparison.loc[comparison['ROC AUC'].idxmax(), 'Model']}")

    def plot_confusion_matrix(self, metrics: Dict, save_path: Optional[Path] = None):
        """
        Plot confusion matrix

        Args:
            metrics: Metrics dictionary with confusion matrix
            save_path: Optional path to save figure
        """
        if not PLOTTING_AVAILABLE:
            print("Warning: Plotting not available (matplotlib/seaborn not installed)")
            return

        cm = metrics['confusion_matrix']

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Down', 'Up'],
                   yticklabels=['Down', 'Up'])
        plt.title(f"Confusion Matrix - {metrics['model_name']}")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Confusion matrix saved to {save_path}")

        plt.close()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = "Model", save_path: Optional[Path] = None):
        """
        Plot ROC curve

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Model name
            save_path: Optional path to save figure
        """
        if not PLOTTING_AVAILABLE:
            print("Warning: Plotting not available (matplotlib/seaborn not installed)")
            return

        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ ROC curve saved to {save_path}")

        plt.close()

    def evaluate_ensemble(self, y_true: np.ndarray,
                         ensemble_pred: np.ndarray,
                         ensemble_proba: np.ndarray,
                         individual_preds: Dict[str, np.ndarray]) -> Dict:
        """
        Evaluate ensemble and individual models

        Args:
            y_true: True labels
            ensemble_pred: Ensemble predictions
            ensemble_proba: Ensemble probabilities
            individual_preds: Dictionary of individual model predictions

        Returns:
            Dictionary with all evaluations
        """
        results = {}

        # Evaluate ensemble
        results['ensemble'] = self.evaluate(
            y_true, ensemble_pred, ensemble_proba, "Ensemble"
        )

        # Evaluate individual models
        for name, pred in individual_preds.items():
            if pred is not None:
                results[name] = self.evaluate(y_true, pred, None, name.capitalize())

        return results


if __name__ == "__main__":
    # Test evaluation
    print("Testing model evaluator...")

    # Generate dummy predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    y_proba = np.random.rand(1000, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, y_proba, "Test Model")
    evaluator.print_metrics(metrics)
