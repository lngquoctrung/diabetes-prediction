import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_prediction_probability_chart(probabilities, prediction_class):
    """
    Create bar chart showing prediction probabilities
    
    Args:
        probabilities: Array of class probabilities
        prediction_class: Predicted class label
    
    Returns:
        Plotly figure object
    """
    classes = ['No Diabetes', 'Pre-diabetes', 'Diabetes']
    colors = ['#00CC96', '#FFA15A', '#EF553B']
    
    # Highlight predicted class
    bar_colors = [
        colors[i] if classes[i] == prediction_class else '#E5E5E5'
        for i in range(len(classes))
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probabilities * 100,
            marker_color=bar_colors,
            text=[f'{p:.1f}%' for p in probabilities * 100],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Class',
        yaxis_title='Probability (%)',
        showlegend=False,
        height=400
    )
    
    return fig


def create_feature_importance_chart(feature_importance_dict):
    """
    Create horizontal bar chart for feature importance
    
    Args:
        feature_importance_dict: Dictionary of feature names and importance scores
    
    Returns:
        Plotly figure object
    """
    # Convert to DataFrame and sort
    df = pd.DataFrame(
        list(feature_importance_dict.items()),
        columns=['Feature', 'Importance']
    )
    df = df.sort_values('Importance', ascending=True)
    
    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance (Top Features)',
        labels={'Importance': 'Importance Score', 'Feature': 'Features'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=600,
        showlegend=False
    )
    
    return fig


def create_model_comparison_chart(df_models, metric='F1-Score'):
    """
    Create bar chart comparing model performance
    
    Args:
        df_models: DataFrame with model comparison data
        metric: Metric to compare (Accuracy, Precision, Recall, F1-Score)
    
    Returns:
        Plotly figure object
    """
    # Create model labels
    df_models['Model_Label'] = df_models['Model'] + ' + ' + df_models['Method']
    
    # Convert percentage strings to float if needed
    if isinstance(df_models[metric].iloc[0], str):
        df_models[metric] = df_models[metric].str.rstrip('%').astype(float)
    
    fig = px.bar(
        df_models,
        x='Model_Label',
        y=metric,
        title=f'Model Comparison - {metric}',
        labels={'Model_Label': 'Model + Balancing Method', metric: f'{metric} (%)'},
        color=metric,
        color_continuous_scale='RdYlGn',
        text=metric
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False
    )
    
    return fig


def create_confusion_matrix_heatmap(cm, class_names):
    """
    Create confusion matrix heatmap
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=500,
        width=500
    )
    
    return fig


def create_pie_chart(data_dict, title="Distribution"):
    """
    Create pie chart for data distribution
    
    Args:
        data_dict: Dictionary with category names and values
        title: Chart title
    
    Returns:
        Plotly figure object
    """
    fig = px.pie(
        values=list(data_dict.values()),
        names=list(data_dict.keys()),
        title=title,
        color_discrete_sequence=['#00CC96', '#FFA15A', '#EF553B']
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    return fig


def create_health_metrics_radar(user_inputs):
    """
    Create radar chart showing user's health metrics
    
    Args:
        user_inputs: Dictionary with user health data
    
    Returns:
        Plotly figure object
    """
    # Normalize metrics to 0-100 scale
    metrics = {
        'Physical Activity': user_inputs.get('phys_activity', 0) * 100,
        'General Health': (6 - user_inputs.get('gen_hlth', 3)) * 20,
        'Mental Health': max(0, 100 - user_inputs.get('ment_hlth', 0) * 3.33),
        'Physical Health': max(0, 100 - user_inputs.get('phys_hlth', 0) * 3.33),
        'BMI Health': max(0, 100 - abs(user_inputs.get('bmi', 25) - 22) * 5),
        'Lifestyle': (1 - user_inputs.get('smoker', 0)) * 50 + 
                     (1 - user_inputs.get('alcohol', 0)) * 50
    }
    
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='#636EFA',
        fillcolor='rgba(99, 110, 250, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title='Health Metrics Overview',
        height=500
    )
    
    return fig
