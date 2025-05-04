import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix
def evaluate_classification(y_true, y_pred):
    """
    Evaluate the performance of a classification model.

    Parameters:
    y_true (array-like): True values of the target variable.
    y_pred (array-like): Predicted values of the target variable.

    Returns:
    dict: Dictionary containing Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = None  # In case of multiclass without probability prediction

    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC Score': roc_auc,
        'Confusion Matrix': cm
    }
    
    return metrics

# Example usage
if __name__ == "__main__":
    # For using evaluate_classification() in other file use import as:
    # from evaluate_classification import evaluate_classification
    y_true = [3, 2, 2, 2, 3, 3, 1, 1, 2, 3]
    y_pred = [3, 2, 2, 2, 1, 3, 1, 2, 2, 3]
    
    metrics = evaluate_classification(y_true, y_pred)
    for metric, value in metrics.items():
        if metric != 'Confusion Matrix':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}:\n{value}")