import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import BRFSS_CLEANED_FILE_PATH


# Global variable to cache loaded data
_cached_df = None
_cached_stats = None


def load_and_optimize_brfss_data():
    """Load BRFSS dataset with memory optimization"""
    global _cached_df
    
    if _cached_df is not None:
        return _cached_df
    
    try:
        df = pd.read_csv(BRFSS_CLEANED_FILE_PATH)
        
        # Optimize memory usage
        for col in df.select_dtypes(include=['int64']).columns:
            if not df[col].isnull().any():
                if df[col].max() <= 32767 and df[col].min() >= -32768:
                    df[col] = df[col].astype('int32')
                elif df[col].max() <= 2147483647 and df[col].min() >= -2147483648:
                    df[col] = df[col].astype('int32')
        
        _cached_df = df
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def calculate_summary_statistics(df):
    """Pre-calculate summary statistics"""
    global _cached_stats
    
    if _cached_stats is not None and _cached_df is df:
        return _cached_stats
    
    stats = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'diabetes_rate': (len(df.loc[df['Diabetes'] == 2.0]) / len(df)) * 100 if 'Diabetes' in df.columns else 0,
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'diabetes_distribution': df['Diabetes'].value_counts() if 'Diabetes' in df.columns else None,
        'missing_data': df.isnull().sum().sort_values(ascending=False)
    }
    
    _cached_stats = stats
    return stats


def get_sample_data(df, sample_size=10000):
    """Get sample data for quick preview"""
    if len(df) > sample_size:
        return df.sample(n=sample_size, random_state=42)
    return df


def create_diabetes_distribution_pie():
    """Create diabetes distribution pie chart with real data"""
    df = load_and_optimize_brfss_data()
    
    if df is None or 'Diabetes' not in df.columns:
        return None
    
    diabetes_counts = df['Diabetes'].value_counts(ascending=True)
    labels = ['No Diabetes', 'Pre-diabetes', 'Diabetes']
    colors = ['#00CC96', '#FFA15A', '#EF553B']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=diabetes_counts.values,
        hole=0.4,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title="Diabetes Distribution (Full Dataset)",
        height=500,
        showlegend=True
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig


def create_age_distribution_by_diabetes():
    """Create age distribution histogram by diabetes status"""
    df = load_and_optimize_brfss_data()
    
    if df is None or 'Age' not in df.columns or 'Diabetes' not in df.columns:
        return None
    
    sample_df = get_sample_data(df, 20000)
    
    fig = px.histogram(
        sample_df, 
        x='Age', 
        color='Diabetes',
        nbins=30, 
        title="Age Distribution by Diabetes Status (Sample: 20K)",
        labels={'Diabetes': 'Diabetes Status', 'Age': 'Age'},
        color_discrete_map={0: '#00CC96', 1: '#FFA15A', 2: '#EF553B'},
        barmode='overlay'
    )
    
    fig.update_layout(height=500)
    fig.update_traces(opacity=0.7)
    
    return fig


def create_health_indicator_analysis(indicator):
    """Create box and violin plots for selected health indicator"""
    df = load_and_optimize_brfss_data()
    
    if df is None or indicator not in df.columns or 'Diabetes' not in df.columns:
        return None, None
    
    # Box plot
    fig_box = px.box(
        df, 
        x='Diabetes', 
        y=indicator,
        title=f"{indicator} by Diabetes Status",
        labels={'Diabetes': 'Diabetes Status'},
        color='Diabetes',
        color_discrete_map={0: '#00CC96', 1: '#FFA15A', 2: '#EF553B'}
    )
    fig_box.update_layout(height=400)
    
    # Violin plot
    fig_violin = px.violin(
        df, 
        x='Diabetes', 
        y=indicator,
        title=f"{indicator} Distribution",
        labels={'Diabetes': 'Diabetes Status'},
        color='Diabetes',
        color_discrete_map={0: '#00CC96', 1: '#FFA15A', 2: '#EF553B'},
        box=True
    )
    fig_violin.update_layout(height=400)
    
    return fig_box, fig_violin


def create_risk_factors_analysis():
    """Create risk factors prevalence comparison"""
    df = load_and_optimize_brfss_data()
    
    if df is None or 'Diabetes' not in df.columns:
        return None
    
    risk_factors = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                    'PhysActivity', 'Depression', 'KidneyDisease', 'COPD']
    available_risks = [col for col in risk_factors if col in df.columns]
    
    if not available_risks:
        return None
    
    # Calculate risk factor prevalence
    risk_data = []
    for factor in available_risks:
        no_diabetes_rate = df[df['Diabetes'] == 0][factor].mean() * 100
        pre_diabetes_rate = df[df['Diabetes'] == 1][factor].mean() * 100
        diabetes_rate = df[df['Diabetes'] == 2][factor].mean() * 100
        
        risk_data.append({
            'Factor': factor,
            'No Diabetes': no_diabetes_rate,
            'Pre-diabetes': pre_diabetes_rate,
            'Diabetes': diabetes_rate
        })
    
    risk_df = pd.DataFrame(risk_data)
    
    # Create grouped bar chart
    fig = px.bar(
        risk_df,
        x='Factor',
        y=['No Diabetes', 'Pre-diabetes', 'Diabetes'],
        title="Risk Factor Prevalence by Diabetes Status (%)",
        barmode='group',
        color_discrete_map={
            'No Diabetes': '#00CC96',
            'Pre-diabetes': '#FFA15A',
            'Diabetes': '#EF553B'
        },
        labels={'value': 'Prevalence (%)', 'variable': 'Diabetes Status'}
    )
    
    fig.update_layout(height=600, xaxis_tickangle=-45)
    
    return fig


def create_correlation_analysis():
    """Create correlation analysis with diabetes"""
    df = load_and_optimize_brfss_data()
    
    if df is None or 'Diabetes' not in df.columns:
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Diabetes' not in numeric_cols:
        return None
    
    # Calculate correlations with diabetes
    correlations = df[numeric_cols].corr()['Diabetes'].abs().sort_values(ascending=False)[1:16]
    
    fig = px.bar(
        x=correlations.values,
        y=correlations.index,
        orientation='h',
        title="Top 15 Features Correlated with Diabetes",
        labels={'x': 'Absolute Correlation', 'y': 'Feature'},
        color=correlations.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=600)
    
    return fig


def get_paginated_data(page_num, page_size):
    """Get paginated data for data viewer"""
    df = load_and_optimize_brfss_data()
    
    if df is None:
        return None, 0, 0
    
    total_rows = len(df)
    total_pages = (total_rows - 1) // page_size + 1
    
    start_idx = page_num * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    page_data = df.iloc[start_idx:end_idx]
    
    return page_data, total_rows, total_pages


def create_missing_data_chart():
    """Create missing data visualization"""
    df = load_and_optimize_brfss_data()
    
    if df is None:
        return None
    
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) == 0:
        return None
    
    fig = px.bar(
        x=missing_data.values,
        y=missing_data.index,
        orientation='h',
        title="Missing Data Count by Column",
        labels={'x': 'Missing Count', 'y': 'Column'},
        color=missing_data.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=500)
    
    return fig


def create_bmi_distribution_by_diabetes():
    """Create BMI distribution by diabetes status"""
    df = load_and_optimize_brfss_data()
    
    if df is None or 'BMI' not in df.columns or 'Diabetes' not in df.columns:
        return None
    
    sample_df = get_sample_data(df, 20000)
    
    fig = px.histogram(
        sample_df,
        x='BMI',
        color='Diabetes',
        nbins=50,
        title="BMI Distribution by Diabetes Status (Sample: 20K)",
        labels={'Diabetes': 'Diabetes Status', 'BMI': 'BMI'},
        color_discrete_map={0: '#00CC96', 1: '#FFA15A', 2: '#EF553B'},
        barmode='overlay'
    )
    
    fig.update_layout(height=500)
    fig.update_traces(opacity=0.7)
    
    return fig


def create_data_dashboard_tab():
    """Create Data Dashboard tab with full functionality"""
    
    with gr.Column():
        gr.Markdown(
            """
            ## BRFSS Dataset Dashboard
            
            Comprehensive analysis and visualization of BRFSS diabetes dataset with real data.
            """
        )
        
        # Load data
        df = load_and_optimize_brfss_data()
        
        if df is None:
            gr.Markdown("**Error**: Unable to load dataset. Please check the file path in config.py")
            return
        
        # Calculate statistics
        stats = calculate_summary_statistics(df)
        
        # Dataset overview metrics
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(f"""
                    <div class='metric-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
                        <div class='metric-label'>Total Samples</div>
                        <div class='metric-value'>{stats['total_samples']:,}</div>
                        <div style='font-size: 0.9em; opacity: 0.9;'>Training Dataset</div>
                    </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML(f"""
                    <div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                        <div class='metric-label'>Features</div>
                        <div class='metric-value'>{stats['total_features']}</div>
                        <div style='font-size: 0.9em; opacity: 0.9;'>Total Columns</div>
                    </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML(f"""
                    <div class='metric-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
                        <div class='metric-label'>Diabetes Rate</div>
                        <div class='metric-value'>{stats['diabetes_rate']:.1f}%</div>
                        <div style='font-size: 0.9em; opacity: 0.9;'>Positive Cases</div>
                    </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML(f"""
                    <div class='metric-card' style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);'>
                        <div class='metric-label'>Data Quality</div>
                        <div class='metric-value'>{100 - stats['missing_percentage']:.1f}%</div>
                        <div style='font-size: 0.9em; opacity: 0.9;'>Complete</div>
                    </div>
                """)
        
        gr.Markdown("---")
        
        # Tabs for different analyses
        with gr.Tabs():
            # Tab 1: Quick Analysis
            with gr.Tab("Quick Analysis"):
                gr.Markdown("### Diabetes Distribution and Key Demographics")
                
                with gr.Row():
                    with gr.Column():
                        diabetes_pie = gr.Plot(
                            value=create_diabetes_distribution_pie(),
                            label="Diabetes Class Distribution"
                        )
                    
                    with gr.Column():
                        age_dist = gr.Plot(
                            value=create_age_distribution_by_diabetes(),
                            label="Age Distribution by Diabetes Status"
                        )
                
                with gr.Row():
                    with gr.Column():
                        bmi_dist = gr.Plot(
                            value=create_bmi_distribution_by_diabetes(),
                            label="BMI Distribution by Diabetes Status"
                        )
                    
                    with gr.Column():
                        risk_factors = gr.Plot(
                            value=create_risk_factors_analysis(),
                            label="Risk Factors Prevalence"
                        )
            
            # Tab 2: Detailed Analysis
            with gr.Tab("Detailed Analysis"):
                gr.Markdown("---")
                gr.Markdown("### Feature Correlation with Diabetes")
                
                correlation_plot = gr.Plot(
                    value=create_correlation_analysis(),
                    label="Top Features Correlated with Diabetes"
                )
            
            # Tab 3: Data Viewer with Pagination
            with gr.Tab("Data Viewer"):
                gr.Markdown("### Browse Dataset with Pagination")
                
                page_size_state = gr.State(value=100)
                current_page_state = gr.State(value=0)
                
                with gr.Row():
                    page_size_dropdown = gr.Dropdown(
                        choices=[50, 100, 500, 1000],
                        value=100,
                        label="Rows per page"
                    )
                    
                    page_info = gr.Markdown(value="Page 1")
                
                data_table = gr.DataFrame(
                    value=df.head(100),
                    label="Dataset View",
                    interactive=False,
                )
                
                with gr.Row():
                    first_btn = gr.Button("First")
                    prev_btn = gr.Button("Previous")
                    next_btn = gr.Button("Next")
                    last_btn = gr.Button("Last")
                
                def update_page_data(page_num, page_size):
                    page_data, total_rows, total_pages = get_paginated_data(page_num, page_size)
                    page_text = f"Page {page_num + 1} of {total_pages} | Showing rows {page_num * page_size + 1}-{min((page_num + 1) * page_size, total_rows)} of {total_rows:,}"
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
                )
                
                page_size_dropdown.change(
                    fn=lambda x: x,
                    inputs=[page_size_dropdown],
                    outputs=[page_size_state]
                )
                
                gr.Markdown("---")
                
                # Download button
                gr.Markdown("### Download Dataset")
                gr.Markdown("Export the full BRFSS dataset to CSV format.")
                
                download_btn = gr.Button("Download Full Dataset", variant="primary")
                download_file = gr.File(label="Download will appear here", visible=False)
                
                def prepare_download():
                    df = load_and_optimize_brfss_data()
                    if df is not None:
                        output_path = "/tmp/brfss_dataset.csv"
                        df.to_csv(output_path, index=False)
                        return gr.File(value=output_path, visible=True)
                    return gr.File(visible=False)
                
                download_btn.click(
                    fn=prepare_download,
                    outputs=[download_file]
                )
            
            # Tab 4: Data Quality
            with gr.Tab("🔍 Data Quality"):
                gr.Markdown("### Data Quality Assessment")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Missing Data Analysis**")
                        missing_chart = gr.Plot(
                            value=create_missing_data_chart(),
                            label="Missing Data by Column"
                        )
                        
                        if missing_chart.value is None:
                            gr.Markdown("**No missing data found!**")
                    
                    with gr.Column():
                        gr.Markdown("**Basic Statistics**")
                        numeric_df = df.select_dtypes(include=[np.number])
                        stats_df = numeric_df.describe().round(2).T
                        stats_df.insert(0, 'Feature', stats_df.index)
                        stats_df = stats_df.reset_index(drop=True)
                        
                        gr.DataFrame(
                            value=stats_df,
                            label="Descriptive Statistics",
                            interactive=False,
                        )
                
                gr.Markdown("---")
                gr.Markdown("### Dataset Information")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Data Types and Memory Usage**")
                        
                        dtype_info = pd.DataFrame({
                            'Column': df.dtypes.index,
                            'Data Type': df.dtypes.astype(str).values,
                            'Non-Null Count': df.count().values,
                            'Memory (MB)': (df.memory_usage(deep=True).values[1:] / 1024**2).round(2)
                        })
                        
                        gr.DataFrame(
                            value=dtype_info,
                            label="Column Information",
                            interactive=False,
                        )
                    
                    with gr.Column():
                        total_memory = df.memory_usage(deep=True).sum() / 1024**2
                        
                        gr.Markdown(f"""
                        **Memory Usage Summary**
                        
                        - **Total Dataset Size**: {total_memory:.2f} MB
                        - **Average per Row**: {(total_memory / len(df) * 1024):.2f} KB
                        - **Optimization**: {'Applied' if df.select_dtypes(include=['int32']).shape[1] > 0 else '❌ Not Applied'}
                        """)
                        
                        # Memory usage by column (top 10)
                        memory_by_col = df.memory_usage(deep=True)[1:].sort_values(ascending=False).head(10)
                        
                        memory_fig = px.bar(
                            x=memory_by_col.values / 1024**2,
                            y=memory_by_col.index,
                            orientation='h',
                            title="Top 10 Columns by Memory Usage (MB)",
                            labels={'x': 'Memory (MB)', 'y': 'Column'}
                        )
                        memory_fig.update_layout(height=400)
                        
                        gr.Plot(value=memory_fig, label="Memory Usage")
        
        gr.Markdown("---")
        gr.Markdown(
            """
            ### Dashboard Features
            
            - **Real-time Data**: Loads actual BRFSS dataset (450K+ samples)
            - **Interactive Visualizations**: Explore data with Plotly charts
            - **Pagination**: Browse large dataset efficiently
            - **Quality Metrics**: Comprehensive data quality assessment
            - **Export**: Download full dataset in CSV format
            """
        )
