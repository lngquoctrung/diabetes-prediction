import plotly.express as px
import plotly.graph_objects as go

def create_pie_chart(data, title="Distribution"):
    """Create pie chart for data distribution"""
    fig = px.pie(
        values=list(data.values()),
        names=list(data.keys()),
        title=title,
        color_discrete_sequence=['#00CC96', '#FFA15A', '#EF553B']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_feature_importance_chart(feature_importance):
    """Create horizontal bar chart for feature importance"""
    fig = px.bar(
        x=list(feature_importance.values()),
        y=list(feature_importance.keys()),
        orientation='h',
        title="Feature importance (Correlation with target)",
        labels={'x': 'Correlation Score', 'y': 'Features'},
        color=list(feature_importance.values()),
        color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def create_prediction_chart(predictions):
    """Create bar chart for prediction probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(predictions.keys()),
            y=[pred * 100 for pred in predictions.values()],
            marker_color=['#00CC96', '#FFA15A', '#EF553B'],
            text=[f"{pred:.1%}" for pred in predictions.values()],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities (%)",
        xaxis_title="Status",
        yaxis_title="Probability (%)",
        showlegend=False
    )
    return fig

def create_model_comparison_chart(df_models, metric):
    """Create model comparison bar chart"""
    fig = px.bar(
        df_models, 
        x='Model', 
        y=metric,
        color='Method',
        title=f'{metric} Comparison Between Models',
        text=metric,
        hover_data=['Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(xaxis_title="Model", yaxis_title=metric)
    return fig

def create_confusion_matrix(conf_matrix):
    """Create confusion matrix heatmap"""
    fig = px.imshow(
        conf_matrix,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['No Diabetes', 'Pre-diabetes', 'Diabetes'],
        y=['No Diabetes', 'Pre-diabetes', 'Diabetes'],
        color_continuous_scale='Blues'
    )
    return fig
