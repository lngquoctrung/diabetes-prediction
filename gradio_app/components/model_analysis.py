import gradio as gr
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

root_dir = str(Path(__file__).parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.sample_data import get_model_comparison_data


def create_model_analysis_tab():
    """Create Model Analysis tab content"""
    
    with gr.Column():
        gr.Markdown(
            """
            ## Model Performance Analysis
            
            Compare the performance of different machine learning models trained with 
            various data balancing techniques.
            """
        )
        
        # Best performing models
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                    <div class='metric-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
                        <div class='metric-label'>Best Accuracy</div>
                        <div class='metric-value'>90.11%</div>
                        <div style='font-size: 0.85em; opacity: 0.9; margin-top: 0.5rem;'>
                            LightGBM + ADASYN Tomek Links
                        </div>
                    </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML("""
                    <div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                        <div class='metric-label'>Best Precision</div>
                        <div class='metric-value'>91.67%</div>
                        <div style='font-size: 0.85em; opacity: 0.9; margin-top: 0.5rem;'>
                            LightGBM + ADASYN Tomek Links
                        </div>
                    </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML("""
                    <div class='metric-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
                        <div class='metric-label'>Best Recall</div>
                        <div class='metric-value'>62.47%</div>
                        <div style='font-size: 0.85em; opacity: 0.9; margin-top: 0.5rem;'>
                            Multiple Models
                        </div>
                    </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML("""
                    <div class='metric-card' style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);'>
                        <div class='metric-label'>Best F1-Score</div>
                        <div class='metric-value'>69.92%</div>
                        <div style='font-size: 0.85em; opacity: 0.9; margin-top: 0.5rem;'>
                            XGBoost + Random Oversampling
                        </div>
                    </div>
                """)
        
        gr.Markdown("---")
        
        # Model comparison table
        gr.Markdown("### Top 5 Model Performance Comparison")
        
        df_models = get_model_comparison_data()
        
        # Format percentages
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
            df_models[col] = df_models[col].apply(lambda x: f"{x:.2%}")
        
        gr.DataFrame(
            value=df_models,
            label="Model Comparison Results",
            interactive=False
        )
        
        gr.Markdown("---")
        
        # Key insights
        gr.Markdown(
            """
            ### Key Insights
            
            **Model Performance Analysis:**
            
            1. **Best Overall Performance**: LightGBM with ADASYN Tomek Links achieves 90.11% accuracy
            2. **Balanced Performance**: XGBoost with Random Oversampling provides best F1-Score (69.92%)
            3. **Class Imbalance Impact**: Pre-diabetes class remains challenging due to limited samples (2.4%)
            4. **Data Balancing Success**: Synthetic sampling methods significantly improve minority class detection
            
            **Recommendations:**
            
            - Use **LightGBM + ADASYN Tomek Links** for highest accuracy in production
            - Use **XGBoost + Random Oversampling** for balanced precision-recall performance
            - Consider ensemble methods for further improvement
            - Focus on collecting more pre-diabetes samples to improve model performance
            """
        )
