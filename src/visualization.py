import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score

# %%
def plot_class_distribution(y, title="Class Distribution"):
    """
    Plot the distribution of classes in a dataset.

    Parameters:
        y: list or numpy array or Pandas series or pandas dataframe
            The target variable
        title: str
            The title of the plot (default is "Class Distribution")
    """
    y.value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    plt.show()

# %%
def plot_roc_auc(y_true, y_pred_proba, model_name="Model"):
    """
    Plot the ROC curve and calculate AUC for a multi-class model.

    Parameters:
        y_true: list or numpy array
            The actual labels
        y_pred_proba: list or numpy array
            The predicted probabilities for each class
        model_name: name of the model (default is "Model")
    """
    # Convert labels to one-hot encoding
    y_true_bin = pd.get_dummies(y_true)

    # Create a figure
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

    # Add a diagonal line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    # Chart settings
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Show plot
    plt.show()

# %%
def plot_correlation_matrix(df, title="Correlation Matrix", figsize=(16, 12)):
    """
    Plot the correlation matrix of a dataset.

    Parameters:
        df: pandas.DataFrame
            The dataset to plot the correlation matrix for.
        title: str
            of the plot (default is "Correlation Matrix")
        figsize: tuple
            Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm_r", fmt=".2")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.title(title)

# %%
def plot_compare_models(results_df):
    """
    Plot the performance comparison of multiple models.
    Parameters:
        results_df: pandas.DataFrame
            The DataFrame containing model performance metrics
    """
    # Plot performance comparison
    plt.figure(figsize=(15, 6))

    # Bar positions
    x = np.arange(len(results_df["Model"]))
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
    for i in range(len(results_df["Model"])):
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

# %%
def plot_diabetes_distribution_by_indicators(df, indicators, year=None, title="Distribution of Diabetes by Indicators", figsize=(16, 12)):
    """
    Graph the distribution of diabetes according to indicators

    Parameters:
    df : pandas.DataFrame
        The DataFrame containing the data
    indicators : list
        List of indicators to plot chart
    year : int, optional
        Year to display in title
    title : str, optional
        Title of the plot
    figsize : tuple
        Figure size (width, height)
    """

    # Mapping for labels
    diabetes_mapping = {
        0: "No",
        1: "Pre-diabetes",
        2: "Diabetes"
    }

    binary_mapping = {
        0: "No",
        1: "Yes"
    }

    general_health_mapping = {
        1: 'Excellent',
        2: 'Very good',
        3: 'Good',
        4: 'Fair',
        5: 'Poor'
    }

    gender_mapping = {
        0: "Female",
        1: "Male"
    }

    # Create a copy so as not to affect the original DataFrame
    df_plot = df.copy()

    # Apply mapping for Diabetes if not already present
    if "Diabetes_label" not in df_plot.columns:
        df_plot["Diabetes_label"] = df_plot["Diabetes"].map(diabetes_mapping)

    # Calculate layout
    n_indicators = len(indicators)

    if n_indicators <= 3:
        # 1 line for 1-3 indicators
        nrows, ncols = 1, n_indicators
    else:
        # 2 columns for 4+ indicators
        ncols = 2
        nrows = (n_indicators + 1) // 2

    # Create the figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Make sure axes is always a 2D array
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Dictionary contains configurations for each indicator
    indicator_configs = {
        'BMI': {
            'plot_type': 'hist',
            'title': 'Distribution of diabetes by BMI',
            'xlabel': 'BMI',
            'mapping': None
        },
        'HighBP': {
            'plot_type': 'count',
            'title': 'Distribution of diabetes by High Blood Pressure',
            'xlabel': 'High Blood Pressure',
            'mapping': binary_mapping,
            'label_col': 'HighBP_label'
        },
        'HighChol': {
            'plot_type': 'count',
            'title': 'Distribution of diabetes by High Cholesterol',
            'xlabel': 'High Cholesterol',
            'mapping': binary_mapping,
            'label_col': 'HighChol_label'
        },
        'DiffWalk': {
            'plot_type': 'count',
            'title': 'Distribution of diabetes by Difficulty Walking',
            'xlabel': 'Difficulty Walking',
            'mapping': binary_mapping,
            'label_col': 'DiffWalk_label'
        },
        'Age': {
            'plot_type': 'count',
            'title': 'Distribution of diabetes by Age Group',
            'xlabel': 'Age Group',
            'mapping': None
        },
        'GenHlth': {
            'plot_type': 'count',
            'title': 'Distribution of diabetes by General Health',
            'xlabel': 'General Health',
            'mapping': general_health_mapping,
            'label_col': 'GenHlth_label'
        },
        'Sex': {
            'plot_type': 'count',
            'title': 'Distribution of diabetes by Gender',
            'xlabel': 'Gender',
            'mapping': gender_mapping,
            'label_col': 'Sex_label'
        },
        'HvyAlcoholConsump': {
            'plot_type': 'count',
            'title': 'Distribution of diabetes by Heavy Alcohol Consumption',
            'xlabel': 'Heavy Alcohol Consumption',
            'mapping': binary_mapping,
            'label_col': 'HvyAlcoholConsump_label'
        },
        'MentHlth': {
            'plot_type': 'hist',
            'title': 'Distribution of diabetes by Mental Health',
            'xlabel': 'Mental Health (days)',
            'mapping': None
        },
        'PhysHlth': {
            'plot_type': 'hist',
            'title': 'Distribution of diabetes by Physical Health',
            'xlabel': 'Physical Health (days)',
            'mapping': None
        }
    }

    # Draw a chart for each indicator
    for i, indicator in enumerate(indicators):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        # Get configuration for current indicator
        config = indicator_configs.get(indicator, {
            'plot_type': 'count',
            'title': f'Distribution of diabetes by {indicator}',
            'xlabel': indicator,
            'mapping': None
        })

        # Apply mapping if any
        if config.get('mapping') and config.get('label_col'):
            if config['label_col'] not in df_plot.columns:
                df_plot[config['label_col']] = df_plot[indicator].map(config['mapping'])
            x_col = config['label_col']
        else:
            x_col = indicator

        # Draw chart based on type
        if config['plot_type'] == 'hist':
            sns.histplot(
                data=df_plot,
                x=indicator,
                hue='Diabetes_label',
                bins=30,
                ax=ax
            )
        else:  # count plot
            sns.countplot(
                data=df_plot,
                x=x_col,
                hue='Diabetes_label',
                dodge=True,
                ax=ax
            )

        # Set title and label
        ax.set_title(config['title'])
        ax.set_xlabel(config['xlabel'])

        # Rotate the x-axis label if needed
        if config['plot_type'] == 'count' and len(df_plot[x_col].unique()) > 3:
            ax.tick_params(axis='x', rotation=45)

    # Hide unused subplots if any
    total_subplots = nrows * ncols
    if n_indicators < total_subplots:
        for i in range(n_indicators, total_subplots):
            row = i // ncols
            col = i % ncols
            axes[row, col].set_visible(False)

    # Set the main title
    if year:
        title += f" in {year}"

    fig.suptitle(title, fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# %%
def plot_feature_distributions_comparison(df_list, year_list, feature, figsize=(16, 8)):
    """
    Compare the distribution of a feature over many years

    Parameters:
    -----------
    df_list : list
        List of DataFrames for each year
    year_list : list
        List of corresponding years
    feature : str
        Feature name to compare
    figsize : tuple
        Figure size
    """
    n_years = len(df_list)
    fig, axes = plt.subplots(1, n_years, figsize=figsize)

    if n_years == 1:
        axes = [axes]

    # Mapping for features
    mappings = {
        'Smoker': {0: 'No', 1: 'Yes'},
        'Stroke': {0: 'No', 1: 'Yes'},
        'HeartDiseaseorAttack': {0: 'No', 1: 'Yes'},
        'PhysActivity': {0: 'No', 1: 'Yes'},
        'AnyHealthcare': {0: 'No', 1: 'Yes'},
        'NoDocbcCost': {0: 'No', 1: 'Yes'},
        'CholCheck': {0: 'No', 1: 'Yes'},
        'Education': {
            1: 'Never attended school',
            2: 'Elementary',
            3: 'Some high school',
            4: 'High school graduate',
            5: 'Some college',
            6: 'College graduate'
        },
        'Income': {
            1: '<$10,000',
            2: '$10,000-$15,000',
            3: '$15,000-$20,000',
            4: '$20,000-$25,000',
            5: '$25,000-$35,000',
            6: '$35,000-$50,000',
            7: '$50,000-$75,000',
            8: '≥$75,000'
        }
    }

    for i, (df, year) in enumerate(zip(df_list, year_list)):
        df_plot = df.copy()

        # Create Diabetes_label if it doesn't exist
        if "Diabetes_label" not in df_plot.columns:
            diabetes_mapping = {0: "No", 1: "Pre-diabetes", 2: "Diabetes"}
            df_plot["Diabetes_label"] = df_plot["Diabetes"].map(diabetes_mapping)

        # Apply mapping if available
        if feature in mappings:
            label_col = f"{feature}_label"
            df_plot[label_col] = df_plot[feature].map(mappings[feature])
            x_col = label_col
        else:
            x_col = feature

        # Draw a chart
        if feature in ['BMI', 'MentHlth', 'PhysHlth']:
            sns.histplot(data=df_plot, x=feature, hue='Diabetes_label',
                         bins=30, ax=axes[i], alpha=0.7)
        elif feature == "Age":
            sns.countplot(data=df_plot, x=x_col, hue='Diabetes_label',
                          dodge=False, ax=axes[i], 
                          hue_order=df_plot["Diabetes_label"].value_counts(ascending=False).index.tolist())
        else:
            sns.countplot(data=df_plot, x=x_col, hue='Diabetes_label',
                          dodge=True, ax=axes[i])

        axes[i].set_title(f'{feature} Distribution in {year}')
        axes[i].set_xlabel(feature)

        # Rotate the label if necessary
        if feature in ['Education', 'Income']:
            axes[i].tick_params(axis='x', rotation=45)

    plt.suptitle(f'Distribution of {feature} by Diabetes Status Over Years', fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# %%
def plot_healthcare_access_analysis(df, year=None, figsize=(16, 10)):
    """
    Analysis of access to health services

    Parameters:
        df: pandas.DataFrame
            The DataFrame containing the data
        year: int, optional
            The year to display in the title
        figsize: tuple
            Figure size (width, height)
    """
    healthcare_indicators = ['AnyHealthcare', 'NoDocbcCost', 'CholCheck', 'GenHlth']

    plot_diabetes_distribution_by_indicators(
        df=df,
        indicators=healthcare_indicators,
        year=year,
        figsize=figsize
    )

# %%
def plot_socioeconomic_analysis(df, year=None, figsize=(16, 10)):
    """
    Analysis of socio-economic factors

    Parameters:
        df: pandas.DataFrame
            The DataFrame containing the data
        year: int, optional
            The year to display in the title
        figsize: tuple
            Figure size (width, height)
    """
    socioeconomic_indicators = ['Education', 'Income', 'Sex', 'Age']

    plot_diabetes_distribution_by_indicators(
        df=df,
        indicators=socioeconomic_indicators,
        year=year,
        figsize=figsize
    )


# %%
def plot_comorbidity_analysis(df, year=None, figsize=(16, 10)):
    """
    Analysis of diseases associated with diabetes

    Parameters:
        df: pandas.DataFrame
            The DataFrame containing the data
        year: int, optional
            The year to display in the title
        figsize: tuple
            Figure size (width, height)
    """
    comorbidity_indicators = ['HighBP', 'HighChol', 'Stroke', 'HeartDiseaseorAttack']

    plot_diabetes_distribution_by_indicators(
        df=df,
        indicators=comorbidity_indicators,
        year=year,
        figsize=figsize
    )


# %%
def plot_comprehensive_risk_profile(df, year=None, figsize=(20, 16)):
    """
    Comprehensive overview of diabetes risk profile

    Parameters:
        df: pandas.DataFrame
            The DataFrame containing the data
        year: int, optional
            The year to display in the title
        figsize: tuple
            Figure size (width, height)
    """
    # Chia thành 4 nhóm risk factors
    risk_groups = {
        'Physical Health': ['BMI', 'PhysActivity', 'DiffWalk'],
        'Chronic Conditions': ['HighBP', 'HighChol', 'HeartDiseaseorAttack'],
        'Lifestyle': ['Smoker', 'HvyAlcoholConsump', 'Fruits'],
        'Demographics': ['Age', 'Sex', 'Income']
    }

    df_plot = df.copy()

    # Create diabetes label
    if "Diabetes_label" not in df_plot.columns:
        diabetes_mapping = {0: "No", 1: "Pre-diabetes", 2: "Diabetes"}
        df_plot["Diabetes_label"] = df_plot["Diabetes"].map(diabetes_mapping)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, (group_name, indicators) in enumerate(risk_groups.items()):
        # Calculate the diabetes ratio for each indicator in the group
        diabetes_rates = {}

        for indicator in indicators:
            if indicator in df_plot.columns:
                # Calculate the diabetes ratio according to each indicator value
                rates = df_plot.groupby(indicator)['Diabetes'].apply(
                    lambda x: (x > 0).sum() / len(x) * 100
                ).to_dict()
                diabetes_rates[indicator] = rates

        # Draw a heatmap for this group
        if diabetes_rates:
            # Prepare data for heatmap
            heatmap_data = []
            labels = []

            for indicator, rates in diabetes_rates.items():
                for value, rate in rates.items():
                    heatmap_data.append([rate])
                    labels.append(f"{indicator}_{value}")

            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data, index=labels, columns=['Diabetes Rate (%)'])
                sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='Reds',
                            ax=axes[idx], cbar_kws={'label': 'Diabetes Rate (%)'})
                axes[idx].set_title(f'{group_name} - Diabetes Risk Profile')
                axes[idx].set_ylabel('')

    title = f"Comprehensive Diabetes Risk Profile"
    if year:
        title += f" - {year}"

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# %%
def plot_diabetes_progression_heatmap(df, features_list, figsize=(14, 10)):
    """
    Draw a heatmap showing the degree of correlation between features and diabetes progression

    Parameters:
        df: pandas.DataFrame
            The DataFrame containing the data
        features_list: list
            The list of features to include in the heatmap
        figsize: tuple
            Figure size (width, height)
    """
    df_plot = df.copy()

    # Tính correlation matrix giữa features và diabetes
    correlation_data = []

    for feature in features_list:
        if feature in df_plot.columns:
            # Tính correlation với diabetes
            corr_with_diabetes = df_plot[feature].corr(df_plot['Diabetes'])

            # Tính mean values cho mỗi diabetes level
            mean_no_diabetes = df_plot[df_plot['Diabetes'] == 0][feature].mean()
            mean_prediabetes = df_plot[df_plot['Diabetes'] == 1][feature].mean()
            mean_diabetes = df_plot[df_plot['Diabetes'] == 2][feature].mean()

            correlation_data.append({
                'Feature': feature,
                'Correlation_with_Diabetes': corr_with_diabetes,
                'No_Diabetes_Mean': mean_no_diabetes,
                'Pre_Diabetes_Mean': mean_prediabetes,
                'Diabetes_Mean': mean_diabetes
            })

    corr_df = pd.DataFrame(correlation_data)
    corr_df = corr_df.set_index('Feature')

    plt.figure(figsize=figsize)
    sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
                cbar_kws={'label': 'Value'})
    plt.title('Feature Analysis: Diabetes Relationship Heatmap', fontsize=14)
    plt.xlabel('Analysis Metrics')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
