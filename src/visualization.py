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
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create mapping for better labels
    diabetes_mapping = {0: "No Diabetes", 1: "Pre-diabetes", 2: "Diabetes"}
    y_labeled = y.map(diabetes_mapping)
    
    counts = y_labeled.value_counts()
    bars = ax.bar(counts.index, counts.values, 
                  color=['lightgreen', 'orange', 'red'], alpha=0.7)
    
    # Add percentage labels on bars
    total = len(y)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = (height / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                f'{height:,}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel("Diabetes Status", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
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
        fpr, tpr, _ = roc_curve(y_true=y_true_bin.iloc[:, i], y_score=y_pred_proba[:, i])
        auc = roc_auc_score(y_true=y_true_bin.iloc[:, i], y_score=y_pred_proba[:, i])
        plt.plot(fpr, tpr, color=colors[i],
                 label=f'{labels[i]} (AUC = {auc:.3f})')

    # Add a diagonal line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    # Chart settings
    plt.xlabel(xlabel='False Positive Rate')
    plt.ylabel(ylabel='True Positive Rate')
    plt.title(label=f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(visible=True)

    # Show plot
    plt.show()

# %%
def plot_correlation_matrix(df, title="Correlation Matrix", figsize=(20, 18)):
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
    
    # Calculate correlation matrix
    corr = df.select_dtypes(include=[np.number]).corr()
    
    # Generate heatmap
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, square=True, fmt='.2f')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

# %%
def plot_compare_models(comparison_df, title="Model Performance Comparison"):
    """
    Plot the performance comparison of multiple models.
    Parameters:
        comparison_df: pandas.DataFrame
            The DataFrame containing model performance metrics
    """
    # Plot performance comparison
    plt.figure(figsize=(15, 6))

    # Bar positions
    x = np.arange(len(comparison_df["Model"]))
    width = 0.15

    # Plot bars
    plt.bar(x=x - width*2, height=comparison_df['Accuracy'], width=width, label='Accuracy', color='skyblue')
    plt.bar(x=x - width, height=comparison_df['Precision'], width=width, label='Precision', color='lightgreen')
    plt.bar(x=x, height=comparison_df['Recall'], width=width, label='Recall', color='salmon')
    plt.bar(x=x + width, height=comparison_df['F1-Score'], width=width, label='F1-Score', color='purple')
    plt.bar(x=x + width*2, height=comparison_df['ROC AUC'], width=width, label='ROC AUC', color='orange')

    # Customize plot
    plt.xlabel(xlabel='Model')
    plt.ylabel(ylabel='Score')
    plt.title(label=title)
    plt.xticks(ticks=x, labels=comparison_df['Model'])
    plt.legend()
    plt.grid(visible=True, alpha=0.3)

    # Annotate bar values
    for i in range(len(comparison_df["Model"])):
        plt.text(x=i - width*2, y=comparison_df['Accuracy'][i],s=f"{comparison_df['Accuracy'][i]:.3f}",
                 ha='center', va='bottom', rotation=0)
        plt.text(x=i - width, y=comparison_df['Precision'][i],s=f"{comparison_df['Precision'][i]:.3f}",
                 ha='center', va='bottom', rotation=0)
        plt.text(x=i, y=comparison_df['Recall'][i],s=f"{comparison_df['Recall'][i]:.3f}",
                 ha='center', va='bottom', rotation=0)
        plt.text(x=i + width, y=comparison_df['F1-Score'][i],s=f"{comparison_df['F1-Score'][i]:.3f}",
                 ha='center', va='bottom', rotation=0)
        plt.text(x=i + width*2, y=comparison_df['ROC AUC'][i], s=f"{comparison_df['ROC AUC'][i]:.3f}",
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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

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

    fig.suptitle(t=title, fontsize=15)
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
    fig, axes = plt.subplots(nrows=1, ncols=n_years, figsize=figsize)

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
def plot_comprehensive_diabetes_analysis(df, year, figsize=(20, 18)):
    """
    Comprehensive analysis of all key features with diabetes
    """
    # Define feature groups and their configurations
    feature_configs = {
        # Demographics
        'Age': {'type': 'categorical', 'title': 'Age Distribution by Diabetes Status'},
        'Sex': {'type': 'binary', 'mapping': {0: 'Male', 1: 'Female'}, 
                'title': 'Gender Distribution by Diabetes Status'},
        'Education': {'type': 'categorical', 'mapping': {
            1: 'Never attended', 2: 'Elementary', 3: 'Some HS', 
            4: 'HS graduate', 5: 'Some college', 6: 'College graduate'
        }, 'title': 'Education Level by Diabetes Status'},
        'Income': {'type': 'categorical', 'mapping': {
            1: '<$10K', 2: '$10-15K', 3: '$15-20K', 4: '$20-25K',
            5: '$25-35K', 6: '$35-50K', 7: '$50-75K', 8: '$75-100K',
            9: '$100-150K', 10: '$150-200K', 11: '$200K+'
        }, 'title': 'Income Level by Diabetes Status'},
        
        # Health Conditions
        'HighBP': {'type': 'binary', 'mapping': {0: 'No', 1: 'Yes'}, 
                   'title': 'High Blood Pressure by Diabetes Status'},
        'HighChol': {'type': 'binary', 'mapping': {0: 'No', 1: 'Yes'}, 
                     'title': 'High Cholesterol by Diabetes Status'},
        'Stroke': {'type': 'binary', 'mapping': {0: 'No', 1: 'Yes'}, 
                   'title': 'Stroke History by Diabetes Status'},
        'HeartDiseaseorAttack': {'type': 'binary', 'mapping': {0: 'No', 1: 'Yes'}, 
                                 'title': 'Heart Disease/Attack by Diabetes Status'},
        'KidneyDisease': {'type': 'binary', 'mapping': {0: 'No', 1: 'Yes'}, 
                          'title': 'Kidney Disease by Diabetes Status'},
        'COPD': {'type': 'binary', 'mapping': {0: 'No', 1: 'Yes'}, 
                 'title': 'COPD by Diabetes Status'},
        'Depression': {'type': 'binary', 'mapping': {0: 'No', 1: 'Yes'}, 
                       'title': 'Depression by Diabetes Status'},
        
        # Lifestyle Factors
        'Smoker': {'type': 'binary', 'mapping': {0: 'No', 1: 'Yes'}, 
                   'title': 'Smoking History by Diabetes Status'},
        'PhysActivity': {'type': 'binary', 'mapping': {0: 'No', 1: 'Yes'}, 
                         'title': 'Physical Activity by Diabetes Status'},
        'HvyAlcoholConsump': {'type': 'binary', 'mapping': {0: 'No', 1: 'Yes'}, 
                              'title': 'Heavy Alcohol Consumption by Diabetes Status'},
        
        # Health Status & Access
        'GenHlth': {'type': 'categorical', 'mapping': {
            1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'
        }, 'title': 'General Health by Diabetes Status'},
        'AnyHealthcare': {'type': 'binary', 'mapping': {0: 'No', 1: 'Yes'}, 
                          'title': 'Healthcare Coverage by Diabetes Status'},
        'NoDocbcCost': {'type': 'binary', 'mapping': {0: 'No', 1: 'Yes'}, 
                        'title': 'Cannot Afford Doctor by Diabetes Status'},
        
        # Continuous Variables
        'BMI': {'type': 'continuous', 'title': 'BMI Distribution by Diabetes Status'},
        'MentHlth': {'type': 'continuous', 'title': 'Mental Health Days by Diabetes Status'},
        'PhysHlth': {'type': 'continuous', 'title': 'Physical Health Days by Diabetes Status'},
    }
    
    # Select features that exist in the dataframe
    available_features = [f for f in feature_configs.keys() if f in df.columns]
    
    # Calculate grid layout
    n_features = len(available_features)
    ncols = 4
    nrows = (n_features + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 else [axes] if nrows == 1 else []
    
    # Create diabetes mapping
    diabetes_mapping = {0: "No Diabetes", 1: "Pre-diabetes", 2: "Diabetes"}
    df_plot = df.copy()
    df_plot['Diabetes_label'] = df_plot['Diabetes'].map(diabetes_mapping)
    
    for i, feature in enumerate(available_features):
        ax = axes[i]
        config = feature_configs[feature]
        
        if config['type'] == 'continuous':
            # Box plot for continuous variables
            sns.boxplot(data=df_plot, x='Diabetes_label', y=feature, ax=ax)
            ax.tick_params(axis='x', rotation=45)
        else:
            # Apply mapping if available
            if 'mapping' in config:
                feature_mapped = f"{feature}_mapped"
                df_plot[feature_mapped] = df_plot[feature].map(config['mapping'])
                x_col = feature_mapped
            else:
                x_col = feature
            
            # Count plot for categorical variables
            sns.countplot(data=df_plot, x=x_col, hue='Diabetes_label', ax=ax)
            
            # Rotate labels if needed
            if len(df_plot[x_col].unique()) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        ax.set_title(config['title'], fontsize=10, fontweight='bold')
        ax.set_xlabel(feature, fontsize=9)
        
        # Adjust legend
        if ax.get_legend():
            ax.legend(fontsize=8, loc='upper right')
    
    # Hide unused subplots
    for i in range(len(available_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Comprehensive Diabetes Analysis by All Features in {year}', 
                 fontsize=16, fontweight='bold', y=0.94)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# %%
def plot_additional_features_analysis(df, year, figsize=(20, 18)):
    """
    Analysis of additional health conditions and lifestyle factors
    """
    # Group features by categories based on actual data structure
    health_conditions = ['DiagnosedHeartAttack', 'CoronaryHeartDisease', 'KidneyDisease', 
                        'COPD', 'Depression', 'CognitiveIssues']
    
    healthcare_access = ['LastCheckup', 'HasPersonalDoctor', 'CholesterolMeds', 
                        'CannotAffordDoctor']
    
    socioeconomic_factors = ['MaritalStatus', 'EmploymentStatus']
    
    lifestyle_factors = ['AlcoholDays']
    
    # Combine all features and filter what exists in dataframe
    all_additional_features = health_conditions + healthcare_access + socioeconomic_factors + lifestyle_factors
    available_features = [f for f in all_additional_features if f in df.columns]
    
    if not available_features:
        print("No additional features found in the dataset")
        return
    
    # Calculate grid layout
    n_features = len(available_features)
    ncols = 3
    nrows = (n_features + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Create diabetes status labels
    diabetes_mapping = {0: "No Diabetes", 1: "Pre-diabetes", 2: "Diabetes"}
    df_plot = df.copy()
    df_plot['Diabetes_label'] = df_plot['Diabetes'].map(diabetes_mapping)
    
    # Feature mappings for better visualization
    binary_mapping = {0: 'No', 1: 'Yes'}
    
    marital_mapping = {1: 'Married', 2: 'Divorced', 3: 'Widowed', 
                      4: 'Separated', 5: 'Never married', 6: 'Unmarried couple'}
    
    employment_mapping = {1: 'Employed wages', 2: 'Self-employed', 3: 'Out work >1yr',
                         4: 'Out work <1yr', 5: 'Homemaker', 6: 'Student', 
                         7: 'Retired', 8: 'Unable to work'}
    
    checkup_mapping = {1: '<1 year', 2: '1-2 years', 3: '2-5 years', 
                      0: '>5 years/Never'}
    
    for i, feature in enumerate(available_features):
        ax = axes[i]
        
        # Determine feature category and apply appropriate visualization
        if feature in health_conditions:
            # Binary health conditions
            df_plot[f'{feature}_mapped'] = df_plot[feature].map(binary_mapping)
            sns.countplot(data=df_plot, x=f'{feature}_mapped', hue='Diabetes_label', ax=ax)
            title_prefix = "Health Condition:"
            
        elif feature in healthcare_access:
            if feature == 'LastCheckup':
                df_plot[f'{feature}_mapped'] = df_plot[feature].map(checkup_mapping)
                sns.countplot(data=df_plot, x=f'{feature}_mapped', hue='Diabetes_label', ax=ax)
                title_prefix = "Healthcare Access:"
            elif feature in ['HasPersonalDoctor', 'CholesterolMeds', 'CannotAffordDoctor']:
                df_plot[f'{feature}_mapped'] = df_plot[feature].map(binary_mapping)
                sns.countplot(data=df_plot, x=f'{feature}_mapped', hue='Diabetes_label', ax=ax)
                title_prefix = "Healthcare Access:"
            else:
                sns.countplot(data=df_plot, x=feature, hue='Diabetes_label', ax=ax)
                title_prefix = "Healthcare Access:"
                
        elif feature == 'MaritalStatus':
            df_plot[f'{feature}_mapped'] = df_plot[feature].map(marital_mapping)
            sns.countplot(data=df_plot, x=f'{feature}_mapped', hue='Diabetes_label', ax=ax)
            title_prefix = "Demographics:"
            
        elif feature == 'EmploymentStatus':
            df_plot[f'{feature}_mapped'] = df_plot[feature].map(employment_mapping)
            sns.countplot(data=df_plot, x=f'{feature}_mapped', hue='Diabetes_label', ax=ax)
            title_prefix = "Demographics:"
            
        elif feature == 'AlcoholDays':
            # Histogram for continuous variable
            sns.histplot(data=df_plot, x=feature, hue='Diabetes_label', 
                        bins=20, ax=ax, alpha=0.7, multiple="stack")
            title_prefix = "Lifestyle:"
        else:
            # Default handling
            sns.countplot(data=df_plot, x=feature, hue='Diabetes_label', ax=ax)
            title_prefix = "Other:"
        
        # Set title and formatting
        clean_feature_name = feature.replace('Diagnosed', '').replace('Cannot', 'Cannot ')
        ax.set_title(f'{title_prefix} {clean_feature_name}', fontweight='bold', fontsize=10)
        
        # Rotate labels if many categories
        if feature in ['MaritalStatus', 'EmploymentStatus', 'LastCheckup']:
            ax.tick_params(axis='x', rotation=45)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        
        # Clean up axis labels
        ax.set_xlabel('')
    
    # Hide unused subplots
    for i in range(len(available_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Comprehensive Analysis of Health Conditions, Healthcare Access & Demographics in {year}', 
                 fontsize=14, fontweight='bold', y=0.94)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# %%
def plot_diabetes_prevalence_heatmap(df, figsize=(20, 18)):
    """
    Create heatmap showing diabetes prevalence across different demographic groups
    """
    # Create age groups for better visualization
    df_plot = df.copy()
    age_mapping = {1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44',
                  6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69',
                  11: '70-74', 12: '75-79', 13: '80+'}
    
    income_mapping = {1: '<$10K', 2: '$10-15K', 3: '$15-20K', 4: '$20-25K',
                     5: '$25-35K', 6: '$35-50K', 7: '$50-75K', 8: '$75-100K',
                     9: '$100-150K', 10: '$150-200K', 11: '$200K+'}
    
    df_plot['Age_group'] = df_plot['Age'].map(age_mapping)
    df_plot['Income_group'] = df_plot['Income'].map(income_mapping)
    
    # Calculate diabetes prevalence (combining pre-diabetes and diabetes)
    df_plot['Has_Diabetes'] = (df_plot['Diabetes'] > 0).astype(int)
    
    # Create pivot table for heatmap
    heatmap_data = df_plot.groupby(['Age_group', 'Income_group'])['Has_Diabetes'].agg(['mean', 'count']).reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Age_group', columns='Income_group', values='mean')
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(heatmap_pivot, annot=True, fmt='.3f', cmap='Reds', 
                cbar_kws={'label': 'Diabetes Prevalence Rate'})
    
    plt.title('Diabetes Prevalence Rate by Age Group and Income Level', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Income Level', fontsize=12)
    plt.ylabel('Age Group', fontsize=12)
    plt.tight_layout()
    plt.show()

# %5
def plot_risk_factor_analysis(df, figsize=(20, 18)):
    """
    Analyze multiple risk factors together
    """
    risk_factors = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'PhysActivity']
    available_factors = [f for f in risk_factors if f in df.columns]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    df_plot = df.copy()
    diabetes_mapping = {0: "No Diabetes", 1: "Pre-diabetes", 2: "Diabetes"}
    df_plot['Diabetes_label'] = df_plot['Diabetes'].map(diabetes_mapping)
    
    # 1. Risk factor prevalence by diabetes status
    risk_prevalence = []
    for factor in available_factors:
        if factor == 'BMI':
            # For BMI, calculate mean
            prev = df_plot.groupby('Diabetes')[factor].mean()
        elif factor == 'PhysActivity':
            # Physical activity is protective (inverse relationship)
            prev = df_plot.groupby('Diabetes')[factor].mean()
        else:
            # For binary factors, calculate prevalence
            prev = df_plot.groupby('Diabetes')[factor].mean()
        risk_prevalence.append(prev)
    
    # Plot 1: Risk factor comparison
    ax = axes[0]
    x = np.arange(len(available_factors))
    width = 0.25
    
    for i, diabetes_status in enumerate([0, 1, 2]):
        values = [risk_prevalence[j][diabetes_status] for j in range(len(available_factors))]
        ax.bar(x + i*width, values, width, 
               label=diabetes_mapping[diabetes_status], alpha=0.8)
    
    ax.set_xlabel('Risk Factors')
    ax.set_ylabel('Prevalence/Mean Value')
    ax.set_title('Risk Factor Profiles by Diabetes Status')
    ax.set_xticks(x + width)
    ax.set_xticklabels(available_factors)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    # Plot 2-5: Individual risk factor distributions
    for i, factor in enumerate(available_factors[:4]):
        ax = axes[i+1]
        if factor == 'BMI':
            sns.boxplot(data=df_plot, x='Diabetes_label', y=factor, ax=ax)
        else:
            factor_mapping = {0: 'No', 1: 'Yes'} if factor != 'PhysActivity' else {0: 'No', 1: 'Yes'}
            df_plot[f'{factor}_mapped'] = df_plot[factor].map(factor_mapping)
            sns.countplot(data=df_plot, x=f'{factor}_mapped', hue='Diabetes_label', ax=ax)
        
        ax.set_title(f'{factor} by Diabetes Status')
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplot
    if len(available_factors) < 5:
        axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.show()