import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def compare_models(models_dict, X_test, y_true):
    """
    Compare the performance of multiple models.

    Parameters:
        models_dict: dictionary containing models to compare {'model name': model}
        X_test: test data
        y_true: actual labels
    """
    # Initialize dictionary to store metrics
    results = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'ROC AUC': [],
        'Precision-class0': [],
        'Precision-class1': [],
        'Precision-class2': [],
        'Recall-class0': [],
        'Recall-class1': [],
        'Recall-class2': [],
        'F1-class0': [],
        'F1-class1': [],
        'F1-class2': []
    }

    # Calculate metrics for each model
    for name, model in models_dict.items():
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Average metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=True)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=True)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=True)

        # Metrics per class
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=True)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=True)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=True)

        # Mean ROC AUC across classes
        y_true_bin = pd.get_dummies(y_true)
        roc_auc = np.mean([roc_auc_score(y_true_bin.iloc[:, i], y_pred_proba[:, i])
                           for i in range(3)])

        # Append results
        results['Model'].append(name)
        results['Accuracy'].append(accuracy)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1-Score'].append(f1)
        results['ROC AUC'].append(roc_auc)

        # Add class-specific metrics
        for i in range(3):
            results[f'Precision-class{i}'].append(precision_per_class[i])
            results[f'Recall-class{i}'].append(recall_per_class[i])
            results[f'F1-class{i}'].append(f1_per_class[i])

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Format float columns to 4 decimal places
    float_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC',
                  'Precision-class0', 'Precision-class1', 'Precision-class2',
                  'Recall-class0', 'Recall-class1', 'Recall-class2',
                  'F1-class0', 'F1-class1', 'F1-class2']
    results_df[float_cols] = results_df[float_cols].round(4)

    return results_df
