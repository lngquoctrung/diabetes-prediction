import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
import gc

root_dir = str(Path(__file__).parent.parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import BRFSS_CLEANED_FILE_PATH

_cached_df = None
_cached_stats = None

def load_and_optimize_brfss_data():
    global _cached_df
    if _cached_df is not None:
        return _cached_df
    
    try:
        df = pd.read_csv(BRFSS_CLEANED_FILE_PATH, low_memory=False)
        
        # Optimize memory usage
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == 'float64':
                if df[col].isnull().any():
                    continue
                if (df[col] == df[col].astype(int)).all():
                    if df[col].max() <= 127 and df[col].min() >= -128:
                        df[col] = df[col].astype('int8')
                    elif df[col].max() <= 32767 and df[col].min() >= -32768:
                        df[col] = df[col].astype('int16')
                    else:
                        df[col] = df[col].astype('int32')
                else:
                    df[col] = df[col].astype('float32')
            elif col_type == 'int64':
                if df[col].max() <= 127 and df[col].min() >= -128:
                    df[col] = df[col].astype('int8')
                elif df[col].max() <= 32767 and df[col].min() >= -32768:
                    df[col] = df[col].astype('int16')
                else:
                    df[col] = df[col].astype('int32')
        
        _cached_df = df
        gc.collect()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_summary_statistics(df):
    global _cached_stats
    if _cached_stats is not None:
        return _cached_stats
    
    stats = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'diabetes_rate': (len(df[df['Diabetes'] == 2]) / len(df)) * 100 if 'Diabetes' in df.columns else 0,
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
    }
    _cached_stats = stats
    return stats

def create_diabetes_distribution_pie():
    df = load_and_optimize_brfss_data()
    if df is None or 'Diabetes' not in df.columns:
        return None
    
    diabetes_counts = df['Diabetes'].value_counts().sort_index()
    labels = ['No Diabetes', 'Pre-diabetes', 'Diabetes']
    colors = ['#00CC96', '#FFA15A', '#EF553B']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=diabetes_counts.values,
        hole=0.4,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title="Diabetes Distribution",
        height=350,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_age_distribution_by_diabetes():
    df = load_and_optimize_brfss_data()
    if df is None or 'Age' not in df.columns or 'Diabetes' not in df.columns:
        return None
    
    sample_df = df.sample(n=min(10000, len(df)), random_state=42)
    
    fig = px.histogram(
        sample_df,
        x='Age',
        color='Diabetes',
        nbins=13,
        title="Age Distribution by Diabetes Status",
        color_discrete_map={0: '#00CC96', 1: '#FFA15A', 2: '#EF553B'},
        barmode='overlay'
    )
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_traces(opacity=0.7)
    return fig

def create_risk_factors_analysis():
    df = load_and_optimize_brfss_data()
    if df is None or 'Diabetes' not in df.columns:
        return None
    
    risk_factors = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack']
    available_risks = [col for col in risk_factors if col in df.columns]
    
    if not available_risks:
        return None
    
    risk_data = []
    for factor in available_risks:
        no_diabetes_rate = df[df['Diabetes'] == 0][factor].mean() * 100
        diabetes_rate = df[df['Diabetes'] == 2][factor].mean() * 100
        risk_data.append({
            'Factor': factor,
            'No Diabetes': no_diabetes_rate,
            'Diabetes': diabetes_rate
        })
    
    risk_df = pd.DataFrame(risk_data)
    
    fig = px.bar(
        risk_df,
        x='Factor',
        y=['No Diabetes', 'Diabetes'],
        title="Risk Factor Prevalence (%)",
        barmode='group',
        color_discrete_map={'No Diabetes': '#00CC96', 'Diabetes': '#EF553B'}
    )
    fig.update_layout(
        height=350,
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_correlation_analysis():
    df = load_and_optimize_brfss_data()
    if df is None or 'Diabetes' not in df.columns:
        return None
    
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)
    correlations = sample_df.corr()['Diabetes'].abs().sort_values(ascending=False)[1:11]
    
    fig = px.bar(
        x=correlations.values,
        y=correlations.index,
        orientation='h',
        title="Top 10 Features Correlated with Diabetes",
        color=correlations.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def get_paginated_data(page_num, page_size):
    df = load_and_optimize_brfss_data()
    if df is None:
        return None, 0, 0
    
    total_rows = len(df)
    total_pages = (total_rows - 1) // page_size + 1
    
    start_idx = page_num * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    page_data = df.iloc[start_idx:end_idx]
    return page_data, total_rows, total_pages

def create_bmi_distribution_by_diabetes():
    df = load_and_optimize_brfss_data()
    if df is None or 'BMI' not in df.columns or 'Diabetes' not in df.columns:
        return None
    
    sample_df = df.sample(n=min(10000, len(df)), random_state=42)
    
    fig = px.histogram(
        sample_df,
        x='BMI',
        color='Diabetes',
        nbins=30,
        title="BMI Distribution by Diabetes Status",
        color_discrete_map={0: '#00CC96', 1: '#FFA15A', 2: '#EF553B'},
        barmode='overlay'
    )
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_traces(opacity=0.7)
    return fig

def create_data_dashboard_tab():
    # CSS responsive
    gr.HTML("""
        <style>
        /* Responsive layout cho mobile */
        @media (max-width: 768px) {
            /* Force tất cả rows thành columns trên mobile */
            .gradio-row {
                flex-direction: column !important;
            }
            
            .gradio-column {
                width: 100% !important;
                max-width: 100% !important;
                margin-bottom: 15px !important;
            }
            
            /* Tối ưu card statistics */
            .stat-card {
                margin-bottom: 15px !important;
            }
            
            /* Plot containers */
            .plot-container {
                width: 100% !important;
                margin-bottom: 20px !important;
            }
            
            /* Tabs responsive */
            .tabs {
                width: 100% !important;
            }
            
            /* Button groups */
            .gradio-button {
                font-size: 14px !important;
                padding: 8px 12px !important;
            }
        }
        
        /* Desktop layout - giữ nguyên grid 2x2 */
        @media (min-width: 769px) {
            .quick-analysis-grid {
                display: grid !important;
                grid-template-columns: repeat(2, 1fr) !important;
                gap: 20px !important;
            }
        }
        </style>
    """)
    
    with gr.Column():
        gr.Markdown("## BRFSS Dataset Dashboard")
        
        df = load_and_optimize_brfss_data()
        if df is None:
            gr.Markdown("**Error**: Unable to load dataset.")
            return
        
        stats = calculate_summary_statistics(df)
        
        # Summary Statistics Cards
        gr.Markdown("### Dataset Overview")
        
        # Card 1 & 2
        with gr.Row(elem_classes="stats-row"):
            with gr.Column(scale=1, min_width=200, elem_classes="stat-card"):
                gr.HTML(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 20px; border-radius: 10px; color: white; text-align: center;">
                        <h3 style="margin: 0; font-size: 16px; font-weight: 500;">Total Samples</h3>
                        <p style="margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{stats['total_samples']:,}</p>
                    </div>
                """)
            
            with gr.Column(scale=1, min_width=200, elem_classes="stat-card"):
                gr.HTML(f"""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 20px; border-radius: 10px; color: white; text-align: center;">
                        <h3 style="margin: 0; font-size: 16px; font-weight: 500;">Features</h3>
                        <p style="margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{stats['total_features']}</p>
                    </div>
                """)
        
        # Card 3 & 4
        with gr.Row(elem_classes="stats-row"):
            with gr.Column(scale=1, min_width=200, elem_classes="stat-card"):
                gr.HTML(f"""
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                padding: 20px; border-radius: 10px; color: white; text-align: center;">
                        <h3 style="margin: 0; font-size: 16px; font-weight: 500;">Diabetes Rate</h3>
                        <p style="margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{stats['diabetes_rate']:.1f}%</p>
                    </div>
                """)
            
            with gr.Column(scale=1, min_width=200, elem_classes="stat-card"):
                gr.HTML(f"""
                    <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                                padding: 20px; border-radius: 10px; color: white; text-align: center;">
                        <h3 style="margin: 0; font-size: 16px; font-weight: 500;">Missing Data</h3>
                        <p style="margin: 10px 0 0 0; font-size: 32px; font-weight: bold;">{stats['missing_percentage']:.2f}%</p>
                    </div>
                """)
        
        # Analysis Tabs
        with gr.Tabs(elem_classes="tabs"):
            with gr.Tab("Quick Analysis"):
                # Desktop: 2x2 grid, Mobile: stacked
                with gr.Row(elem_classes="quick-analysis-grid"):
                    with gr.Column(elem_classes="plot-container"):
                        diabetes_pie = gr.Plot(value=create_diabetes_distribution_pie())
                    
                    with gr.Column(elem_classes="plot-container"):
                        age_dist = gr.Plot(value=create_age_distribution_by_diabetes())
                
                with gr.Row(elem_classes="quick-analysis-grid"):
                    with gr.Column(elem_classes="plot-container"):
                        bmi_dist = gr.Plot(value=create_bmi_distribution_by_diabetes())
                    
                    with gr.Column(elem_classes="plot-container"):
                        risk_factors = gr.Plot(value=create_risk_factors_analysis())
            
            with gr.Tab("Detailed Analysis"):
                # Responsive correlation plot
                with gr.Column(elem_classes="plot-container"):
                    correlation_plot = gr.Plot(value=create_correlation_analysis())
            
            with gr.Tab("Data Viewer"):
                page_size_state = gr.State(value=50)
                current_page_state = gr.State(value=0)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        page_size_dropdown = gr.Dropdown(
                            choices=[50, 100], 
                            value=50, 
                            label="Rows per page"
                        )
                    with gr.Column(scale=1):
                        page_info = gr.Markdown(value="Page 1")
                
                data_table = gr.DataFrame(value=df.head(50), interactive=False)
                
                with gr.Row():
                    first_btn = gr.Button("First", size="sm")
                    prev_btn = gr.Button("Previous", size="sm")
                    next_btn = gr.Button("Next", size="sm")
                    last_btn = gr.Button("Last", size="sm")
                
                def update_page_data(page_num, page_size):
                    page_data, total_rows, total_pages = get_paginated_data(page_num, page_size)
                    page_text = f"Page {page_num + 1} of {total_pages}"
                    return page_data, page_text, page_num
                
                def go_first(page_size):
                    return update_page_data(0, page_size)
                
                def go_prev(current_page, page_size):
                    new_page = max(0, current_page - 1)
                    return update_page_data(new_page, page_size)
                
                def go_next(current_page, page_size):
                    _, total_rows, total_pages = get_paginated_data(current_page, page_size)
                    new_page = min(total_pages - 1, current_page + 1)
                    return update_page_data(new_page, page_size)
                
                def go_last(page_size):
                    _, total_rows, total_pages = get_paginated_data(0, page_size)
                    return update_page_data(total_pages - 1, page_size)
                
                def change_page_size(page_size):
                    return update_page_data(0, page_size)
                
                # Event handlers
                first_btn.click(
                    fn=go_first,
                    inputs=[page_size_state],
                    outputs=[data_table, page_info, current_page_state]
                )
                
                prev_btn.click(
                    fn=go_prev,
                    inputs=[current_page_state, page_size_state],
                    outputs=[data_table, page_info, current_page_state]
                )
                
                next_btn.click(
                    fn=go_next,
                    inputs=[current_page_state, page_size_state],
                    outputs=[data_table, page_info, current_page_state]
                )
                
                last_btn.click(
                    fn=go_last,
                    inputs=[page_size_state],
                    outputs=[data_table, page_info, current_page_state]
                )
                
                page_size_dropdown.change(
                    fn=change_page_size,
                    inputs=[page_size_dropdown],
                    outputs=[data_table, page_info, current_page_state]
                ).then(
                    fn=lambda x: x,
                    inputs=[page_size_dropdown],
                    outputs=[page_size_state]
                )
