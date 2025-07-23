import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def plot_roc_auc(y_true, y_pred_proba, model_name="Model"):
    """
    Plot the ROC curve and calculate AUC for a multi-class model.

    Parameters:
        y_true: actual labels
        y_pred_proba: predicted probabilities for each class
        model_name: name of the model (default is "Model")
    """
    # Convert labels to one-hot encoding
    y_true_bin = pd.get_dummies(y_true)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Colors for ROC curves
    colors = ['blue', 'red', 'yellow']
    labels = ['Non-diabetic', 'Pre-diabetic', 'Diabetic']

    # Plot ROC curve for each class
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_true_bin.iloc[:, i], y_pred_proba[:, i])
        auc = roc_auc_score(y_true_bin.iloc[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, color=colors[i],
                 label=f'{labels[i]} (AUC = {auc:.3f})')

    # Add diagonal line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    # Chart settings
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Show plot
    plt.show()


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
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # Metrics per class
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)

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

    # Plot performance comparison
    plt.figure(figsize=(15, 6))

    # Bar positions
    x = np.arange(len(models_dict))
    width = 0.15

    # Plot bars
    plt.bar(x - width*2, results_df['Accuracy'], width, label='Accuracy', color='skyblue')
    plt.bar(x - width, results_df['Precision'], width, label='Precision', color='lightgreen')
    plt.bar(x, results_df['Recall'], width, label='Recall', color='salmon')
    plt.bar(x + width, results_df['F1-Score'], width, label='F1-Score', color='purple')
    plt.bar(x + width*2, results_df['ROC AUC'], width, label='ROC AUC', color='orange')

    # Customize plot
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, results_df['Model'])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Annotate bar values
    for i in range(len(models_dict)):
        plt.text(i - width*2, results_df['Accuracy'][i], f"{results_df['Accuracy'][i]:.3f}",
                 ha='center', va='bottom', rotation=0)
        plt.text(i - width, results_df['Precision'][i], f"{results_df['Precision'][i]:.3f}",
                 ha='center', va='bottom', rotation=0)
        plt.text(i, results_df['Recall'][i], f"{results_df['Recall'][i]:.3f}",
                 ha='center', va='bottom', rotation=0)
        plt.text(i + width, results_df['F1-Score'][i], f"{results_df['F1-Score'][i]:.3f}",
                 ha='center', va='bottom', rotation=0)
        plt.text(i + width*2, results_df['ROC AUC'][i], f"{results_df['ROC AUC'][i]:.3f}",
                 ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.show()

    return results_df
