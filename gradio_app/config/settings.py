import gradio as gr
from pathlib import Path


def get_theme():
    """Get custom Gradio theme with better fonts"""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=[
            "Inter",
            "system-ui",
            "-apple-system",
            "BlinkMacSystemFont",
            "Segoe UI",
            "Roboto",
            "sans-serif"
        ],
        font_mono=[
            "IBM Plex Mono",
            "ui-monospace",
            "Consolas",
            "monospace"
        ],
    ).set(
        body_text_size="*text_md",
        body_text_weight="400",
    )
    return theme


def load_custom_css():
    """Load custom CSS for better styling"""
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    /* Global font improvements */
    * {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
        letter-spacing: -0.02em;
        line-height: 1.3;
    }
    
    h1 { font-size: 2.5em !important; }
    h2 { font-size: 2em !important; }
    h3 { font-size: 1.5em !important; }
    
    /* Paragraphs and text */
    p, span, label {
        line-height: 1.6;
        font-size: 1em;
    }
    
    /* Code blocks */
    code, pre {
        font-family: 'IBM Plex Mono', 'Consolas', monospace !important;
    }
    
    /* Buttons */
    button {
        font-weight: 500 !important;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
    }
    
    button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Input fields */
    input, select, textarea {
        font-size: 1em !important;
        border-radius: 8px !important;
    }
    
    /* Tables */
    table {
        font-size: 0.95em;
    }
    
    table th {
        font-weight: 600 !important;
        background-color: #f8f9fa;
    }
    
    /* Tabs */
    .tabs button {
        font-size: 1.1em !important;
        font-weight: 500 !important;
    }
    
    /* Plots */
    .plot-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Success/Warning/Error messages */
    .gr-message {
        border-radius: 8px;
        padding: 1rem;
        font-weight: 500;
    }
    """
    return css


# App configuration
APP_CONFIG = {
    "title": "Diabetes Prediction Dashboard",
    "version": "2.0.0",
    "description": "ML-powered diabetes risk assessment using BRFSS data",
    "port": 7860,
    "host": "0.0.0.0"
}
