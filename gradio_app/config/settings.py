import gradio as gr

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
        
        /* Paragraphs */
        p, span, label {
            line-height: 1.6;
            font-size: 1em;
        }
        
        /* Buttons */
        button {
            font-weight: 500 !important;
            transition: all 0.2s ease;
        }
        
        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Input fields */
        input, select, textarea {
            font-size: 1em !important;
            border-radius: 8px !important;
        }
        
        /* Tabs */
        .tabs button {
            font-size: 1.1em !important;
            font-weight: 500 !important;
        }
        
        /* Custom metric cards */
        .metric-card {
            color: white !important;
            padding: 24px !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 8px !important;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2.5em !important;
            font-weight: 700 !important;
            margin: 12px 0 !important;
            line-height: 1.2;
        }
        
        .metric-label {
            font-size: 1em !important;
            opacity: 0.95;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 500 !important;
        }
        
        /* Gradient boxes styling */
        div[style*="linear-gradient"] h3 {
            font-size: 1em !important;
            margin: 0 !important;
            font-weight: 500 !important;
        }
        
        div[style*="linear-gradient"] p {
            font-size: 2rem !important;
            margin: 10px 0 0 0 !important;
            font-weight: bold !important;
        }
        
        /* Main container */
            .dataframe {
            width: 100% !important;
            overflow-x: auto !important;
            border: 1px solid #ddd !important;
            border-radius: 4px !important;
        }
        
        .dataframe table {
            width: 100% !important;
            border-collapse: collapse !important;
            font-size: 14px !important;
            font-family: Arial, sans-serif !important;
        }
        
        .dataframe th {
            background-color: #f8f9fa !important;
            border: 1px solid #ddd !important;
            padding: 10px !important;
            text-align: center !important;
            font-weight: 600 !important;
            white-space: nowrap !important;
        }
        
        .dataframe td {
            border: 1px solid #ddd !important;
            padding: 8px 10px !important;
            text-align: center !important;
            white-space: nowrap !important;
        }
        
        .dataframe tbody tr:hover {
            background-color: #f5f5f5 !important;
        }
        
        /* Mobile responsive */
        @media (max-width: 768px) {
            .dataframe {
                font-size: 12px !important;
            }
        }
    """
    
    return css

def load_fonts_html():
    """Load Google Fonts via HTML link tag"""
    return """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    """
