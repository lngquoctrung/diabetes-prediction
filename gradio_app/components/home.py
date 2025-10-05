import gradio as gr
import pandas as pd
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.sample_data import get_feature_importance_data


def create_home_tab():
    """Create Home tab content"""
    
    with gr.Column():
        gr.Markdown(
            """
            ## Project Overview
            
            Welcome to the Diabetes Prediction Dashboard! This application uses advanced machine learning 
            models trained on BRFSS (Behavioral Risk Factor Surveillance System) data from 2017-2023 to 
            predict diabetes risk.
            """
        )
        
        # Key metrics
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                    <div class='metric-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
                        <div class='metric-label'>Dataset Size</div>
                        <div class='metric-value'>450,445</div>
                        <div style='font-size: 0.9em; opacity: 0.9;'>Total Samples</div>
                    </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML("""
                    <div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                        <div class='metric-label'>Best Accuracy</div>
                        <div class='metric-value'>90.11%</div>
                        <div style='font-size: 0.9em; opacity: 0.9;'>LightGBM Model</div>
                    </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML("""
                    <div class='metric-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
                        <div class='metric-label'>Best F1-Score</div>
                        <div class='metric-value'>69.92%</div>
                        <div style='font-size: 0.9em; opacity: 0.9;'>XGBoost Model</div>
                    </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML("""
                    <div class='metric-card' style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);'>
                        <div class='metric-label'>Total Features</div>
                        <div class='metric-value'>34</div>
                        <div style='font-size: 0.9em; opacity: 0.9;'>Engineered Features</div>
                    </div>
                """)
        
        gr.Markdown("---")
        
        # Diabetes distribution
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Diabetes Distribution")
                
                # Create pie chart data
                distribution_data = {
                    "Class": ["No Diabetes", "Pre-diabetes", "Diabetes"],
                    "Percentage": [82.0, 2.4, 15.6],
                    "Count": [355421, 11985, 83039]
                }
                df_dist = pd.DataFrame(distribution_data)
                
                gr.DataFrame(
                    value=df_dist,
                    label="Class Distribution",
                    interactive=False
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Data Balancing Methods")
                
                balancing_data = {
                    "Method": [
                        "Random Oversampling",
                        "SMOTE",
                        "ADASYN",
                        "SMOTE + Tomek Links",
                        "ADASYN + Tomek Links"
                    ],
                    "Type": [
                        "Oversampling",
                        "Synthetic",
                        "Adaptive Synthetic",
                        "Hybrid",
                        "Hybrid"
                    ]
                }
                df_balance = pd.DataFrame(balancing_data)
                
                gr.DataFrame(
                    value=df_balance,
                    label="Balancing Techniques Used",
                    interactive=False
                )
        
        gr.Markdown("---")
        
        # Top features
        gr.Markdown("### Top 10 Most Important Features")
        
        feature_importance = get_feature_importance_data()
        top_features = list(feature_importance.items())[:10]
        
        df_features = pd.DataFrame(
            top_features,
            columns=["Feature", "Importance Score"]
        )
        df_features["Importance Score"] = df_features["Importance Score"].round(4)
        
        gr.DataFrame(
            value=df_features,
            label="Feature Importance (Correlation with Diabetes)",
            interactive=False
        )
        
        gr.Markdown(
            """
            ---
            ### Getting Started
            
            1. **Prediction Tab**: Enter health information to get diabetes risk prediction
            2. **Model Analysis**: Compare different ML models and their performance
            3. **Project Details**: Learn about data pipeline, feature engineering, and methodology
            4. **Data Dashboard**: Explore the BRFSS dataset with interactive visualizations
            """
        )
