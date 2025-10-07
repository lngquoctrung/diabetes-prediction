import gradio as gr
import sys
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
        
        # Best performing models cards
        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                gr.HTML("""
                    <div class='metric-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
                        <div class='metric-label'>Best Accuracy</div>
                        <div class='metric-value'>90.11%</div>
                        <div style='font-size: 0.85em; opacity: 0.9; margin-top: 0.5rem;'>
                            LightGBM + ADASYN Tomek Links
                        </div>
                    </div>
                """)
            
            with gr.Column(scale=1, min_width=200):
                gr.HTML("""
                    <div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                        <div class='metric-label'>Best Precision</div>
                        <div class='metric-value'>91.67%</div>
                        <div style='font-size: 0.85em; opacity: 0.9; margin-top: 0.5rem;'>
                            LightGBM + ADASYN Tomek Links
                        </div>
                    </div>
                """)
            
            with gr.Column(scale=1, min_width=200):
                gr.HTML("""
                    <div class='metric-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
                        <div class='metric-label'>Best Recall</div>
                        <div class='metric-value'>65.42%</div>
                        <div style='font-size: 0.85em; opacity: 0.9; margin-top: 0.5rem;'>
                            XGBoost + Random Oversampling
                        </div>
                    </div>
                """)
            
            with gr.Column(scale=1, min_width=200):
                gr.HTML("""
                    <div class='metric-card' style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);'>
                        <div class='metric-label'>Best F1-Score</div>
                        <div class='metric-value'>69.92%</div>
                        <div style='font-size: 0.85em; opacity: 0.9; margin-top: 0.5rem;'>
                            XGBoost + Random Oversampling
                        </div>
                    </div>
                """)
        
        # Model comparison table
        gr.Markdown("### Top 5 Model Performance Comparison")
        
        # Get data and display as DataFrame
        df_models = get_model_comparison_data()
        
        gr.DataFrame(
            value=df_models,
            interactive=False,
            wrap=False
        )
        
        # Key insights
        gr.Markdown(
            """
            ### Key Insights
            
            - **LightGBM with ADASYN + Tomek Links** achieved the highest accuracy (90.11%) and precision (91.67%)
            - **XGBoost with Random Oversampling** shows the best balance with highest F1-Score (69.92%)
            - **Data balancing techniques** significantly improve model performance
            - **Hybrid methods** (SMOTE/ADASYN + Tomek Links) generally outperform single methods
            
            ### Recommendations
            
            - For **high-precision requirements**: Use LightGBM + ADASYN Tomek Links
            - For **balanced performance**: Use XGBoost + Random Oversampling
            - For **recall-focused tasks**: Consider XGBoost with oversampling methods
            """
        )
