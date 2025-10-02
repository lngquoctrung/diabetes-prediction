import streamlit as st
import sys
import os

from pathlib import Path

root_dir = str(Path(__file__).parent.parent.absolute())
dashboard_dir = os.path.join(root_dir, "app")
if not dashboard_dir in sys.path:
    sys.path.insert(0, dashboard_dir)
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

from utils.ui_components import setup_page_config, add_custom_css, create_metric_card, add_sidebar_info
from data.sample_data import load_sample_data, get_feature_importance_data
from utils.visualization import create_pie_chart, create_feature_importance_chart

# Page configuration
setup_page_config()
add_custom_css()

# Main title
st.title("Diabetes Prediction Dashboard")
st.markdown("**Advanced diabetes prediction system using BRFSS 2017-2023 data with machine learning**")
st.markdown("---")

# Load sample data
df = load_sample_data()

# Overview section
st.header("Project Overview")

# Key metrics - Updated with actual results
col1, col2, col3, col4 = st.columns(4)

with col1:
    create_metric_card(
        label="Total Samples",
        value="450,445",
        help="Total samples after feature engineering"
    )

with col2:
    create_metric_card(
        label="Best Accuracy",
        value="90.11%",
        delta="LightGBM + ADASYN Tomek Links",
        help="Highest accuracy model"
    )

with col3:
    create_metric_card(
        label="Best F1-Score",
        value="69.92%",
        delta="XGBoost + Random Oversampling",
        help="Best balanced precision and recall model"
    )

with col4:
    create_metric_card(
        label="Features",
        value="34",
        delta="From 43 original features",
        help="Number of features after feature selection"
    )

st.markdown("---")

# Layout in 2 columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Diabetes Distribution in Dataset")
    # Create pie chart
    diabetes_counts = df['Diabetes'].value_counts()
    diabetes_labels = {0: 'No Diabetes', 1: 'Pre-diabetes', 2: 'Diabetes'}
    diabetes_data = {diabetes_labels[i]: count for i, count in diabetes_counts.items()}
    fig_pie = create_pie_chart(diabetes_data, "Distribution of diabetes status")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Data Balancing Techniques")
    balancing_methods = [
        "Random Oversampling",
        "SMOTE",
        "ADASYN",
        "Random Undersampling",
        "Tomek Links",
        "Edited Nearest Neighbors",
        "SMOTE + Tomek Links",
        "SMOTE + ENN",
        "ADASYN + Tomek Links"
    ]
    
    st.markdown("**Applied methods:**")
    for method in balancing_methods:
        st.markdown(f"{method}")

# Feature importance visualization
st.subheader("Top 10 Most Important Features")
feature_importance = get_feature_importance_data()
fig_importance = create_feature_importance_chart(feature_importance)
st.plotly_chart(fig_importance, use_container_width=True)

# Navigation guide
st.markdown("---")
st.markdown("## Getting Started")
st.markdown("**Use the sidebar to navigate between different pages:**")
st.markdown("""
- **Prediction**: Enter your health information to get diabetes risk prediction
- **Model Analysis**: Compare different machine learning models performance  
- **Project Details**: Technical details about the project methodology
""")

# Add sidebar info
add_sidebar_info()
