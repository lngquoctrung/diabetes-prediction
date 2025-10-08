import gradio as gr
import sys
from pathlib import Path

# Setup paths
root_dir = str(Path(__file__).parent.parent.absolute())
gradio_app_dir = Path(__file__).parent.absolute()
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if gradio_app_dir not in sys.path:
    sys.path.insert(0, gradio_app_dir)

from components.home import create_home_tab
from components.prediction import create_prediction_tab
from components.model_analysis import create_model_analysis_tab
from components.project_details import create_project_details_tab
from components.data_dashboard import create_data_dashboard_tab
from config.settings import get_theme, load_custom_css, load_fonts_html

def main():
    """Main Gradio app with tabs"""
    gr.set_static_paths(paths=[gradio_app_dir])
    
    # Load custom CSS
    custom_css = load_custom_css()
    
    # Create Gradio app with custom theme
    with gr.Blocks(theme=get_theme(), css=custom_css, title="Diabetes Prediction Dashboard") as app:
        # Suppress console errors with JavaScript
        gr.HTML("""
        <script>
            // Suppress manifest and CSS errors in console
            const originalError = console.error;
            console.error = function(...args) {
                const msg = args[0]?.toString() || '';
                if (msg.includes('manifest') || msg.includes('@import') || msg.includes('construct-stylesheets')) {
                    return;
                }
                originalError.apply(console, args);
            };
            </script>
        """)
        
        # Load Google Fonts via HTML
        gr.HTML(load_fonts_html())
        
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
            
            with gr.Tab("Data Dashboard"):
                create_data_dashboard_tab()
            
            with gr.Tab("Project Details"):
                create_project_details_tab()
    
    return app

if __name__ == "__main__":
    app = main()
    app.launch(
        server_name="0.0.0.0",
        server_port=7000,
        allowed_paths=[gradio_app_dir],
        pwa=True
    )
