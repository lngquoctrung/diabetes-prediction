import gradio as gr
import sys
from pathlib import Path

# Setup paths
root_dir = str(Path(__file__).parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from gradio_app.components.home import create_home_tab
from gradio_app.components.prediction import create_prediction_tab
from gradio_app.components.model_analysis import create_model_analysis_tab
from gradio_app.components.project_details import create_project_details_tab
from gradio_app.components.data_dashboard import create_data_dashboard_tab
from gradio_app.config.settings import get_theme, load_custom_css


def main():
    """Main Gradio app with tabs"""
    
    # Load custom CSS
    custom_css = load_custom_css()
    
    # Create Gradio app with custom theme
    with gr.Blocks(theme=get_theme(), css=custom_css, title="Diabetes Prediction Dashboard") as app:
        
        gr.Markdown(
            """
            # Diabetes Prediction Dashboard
            ### Advanced diabetes prediction system using BRFSS 2017-2023 data with machine learning
            """
        )
        
        # Create tabs
        with gr.Tabs():
            with gr.Tab("Home"):
                create_home_tab()
            
            with gr.Tab("Prediction"):
                create_prediction_tab()
            
            with gr.Tab("Model Analysis"):
                create_model_analysis_tab()
            
            with gr.Tab("Project Details"):
                create_project_details_tab()
            
            with gr.Tab("Data Dashboard"):
                create_data_dashboard_tab()
        
        gr.Markdown(
            """
            ---
            <div style='text-align: center; color: #666; font-size: 0.9em;'>
            Diabetes Prediction Dashboard | BRFSS 2017-2023 Dataset | Built with Gradio & XGBoost
            </div>
            """
        )
    
    return app


if __name__ == "__main__":
    app = main()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )