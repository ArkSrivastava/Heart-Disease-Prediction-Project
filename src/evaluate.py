from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(y_true, y_pred, y_proba=None):
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_proba is not None:
        results['roc_auc'] = roc_auc_score(y_true, y_proba)
        results['y_proba'] = y_proba
    
    return results

def plot_roc_curves(models_results, output_path='roc_curves.png'):
    """
    Plot ROC curves for multiple models
    models_results: dict with model names as keys and results dict as values
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, results in models_results.items():
        if 'y_proba' in results and 'y_true' in results:
            fpr, tpr, _ = roc_curve(results['y_true'], results['y_proba'])
            auc = results['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"ROC curves saved to {output_path}")
