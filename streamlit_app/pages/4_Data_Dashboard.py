import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Setup paths
root_dir = str(Path(__file__).parent.parent.parent.absolute())
dashboard_dir = os.path.join(root_dir, "app")
if not dashboard_dir in sys.path:
    sys.path.insert(0, dashboard_dir)
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

from utils.ui_components import setup_page_config, add_custom_css, add_sidebar_info
from src.config import BRFSS_CLEANED_FILE_PATH

# Page setup
setup_page_config()
add_custom_css()

# Session state initialization for pagination
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
if 'page_size' not in st.session_state:
    st.session_state.page_size = 1000
if 'load_detailed_analysis' not in st.session_state:
    st.session_state.load_detailed_analysis = False
if 'load_charts' not in st.session_state:
    st.session_state.load_charts = False

st.header("BRFSS Dataset Dashboard")
st.markdown("Comprehensive analysis and visualization of BRFSS diabetes dataset")

# Optimized data loading with Arrow-compatible data types
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_and_optimize_brfss_data():
    """Load BRFSS dataset with memory optimization and Arrow compatibility"""
    try:
        df = pd.read_csv(BRFSS_CLEANED_FILE_PATH)
        for col in df.select_dtypes(include=['int64']).columns:
            if not df[col].isnull().any():
                if df[col].max() <= 32767 and df[col].min() >= -32768:
                    df[col] = df[col].astype('int32')
                elif df[col].max() <= 2147483647 and df[col].min() >= -2147483648:
                    df[col] = df[col].astype('int32')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def calculate_summary_statistics(df):
    """Pre-calculate summary statistics to avoid repeated computation"""
    stats = {
        'total_samples': len(df),
        'total_features': len(df.columns),
        'diabetes_rate': (len(df.loc[df['Diabetes'] == 2.0]) / len(df)) * 100 if 'Diabetes' in df.columns else 0,
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'diabetes_distribution': df['Diabetes'].value_counts() if 'Diabetes' in df.columns else None,
        'missing_data': df.isnull().sum().sort_values(ascending=False)
    }
    return stats

@st.cache_data
def get_sample_data(df, sample_size=10000):
    """Get sample data for quick preview"""
    if len(df) > sample_size:
        return df.sample(n=sample_size, random_state=42)
    return df

def paginate_dataframe(df, page_size, page_number):
    """Paginate dataframe for better performance"""
    start_idx = page_number * page_size
    end_idx = start_idx + page_size
    return df.iloc[start_idx:end_idx]

# Load data with spinner
with st.spinner("Loading data..."):
    df = load_and_optimize_brfss_data()

if df is not None:
    # Calculate summary statistics
    stats = calculate_summary_statistics(df)
    
    # Dataset overview - always visible (lightweight)
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{stats['total_samples']:,}")
    
    with col2:
        st.metric("Features", f"{stats['total_features']}")
    
    with col3:
        st.metric("Diabetes Rate", f"{stats['diabetes_rate']:.1f}%")
    
    with col4:
        st.metric("Missing Data", f"{stats['missing_percentage']:.2f}%")
    
    # Lazy loading with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Quick Analysis", "Detailed Analysis", "Data Viewer", "Data Quality"])
    
    with tab1:
        st.subheader("Quick Diabetes Distribution (Sample Data)")
        
        # Use sample data for quick visualization
        sample_df = get_sample_data(df, 10000)
        
        if 'Diabetes' in df.columns:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Quick pie chart with sample data
                diabetes_counts = sample_df['Diabetes'].value_counts(ascending=True)
                labels = ['No Diabetes', "Pre-diabetes", 'Diabetes']
                colors = ['#90EE90', '#67ABF8', '#FF6B6B']
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=diabetes_counts.values,
                    hole=0.4,
                    marker_colors=colors
                )])
                fig_pie.update_layout(
                    title="Diabetes Distribution (Sample)",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Quick age distribution
                if 'Age' in sample_df.columns:
                    fig_age = px.histogram(
                        sample_df, x='Age', color='Diabetes',
                        nbins=20, title="Age Distribution by Diabetes Status (Sample)",
                        labels={'Diabetes': 'Diabetes Status', 'Age': 'Age'},
                        color_discrete_map={0: '#90EE90', 1: '#67ABF8', 2: '#FF6B6B'}
                    )
                    fig_age.update_layout(height=400)
                    st.plotly_chart(fig_age, use_container_width=True)
    
    with tab2:
        st.subheader("Detailed Analysis")
        
        if not st.session_state.load_detailed_analysis:
            if st.button("Load Detailed Analysis", key="load_detailed"):
                st.session_state.load_detailed_analysis = True
                st.rerun()
        else:
            # Health indicators analysis
            st.subheader("Health Indicators Analysis")
            
            health_indicators = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth']
            available_indicators = [col for col in health_indicators if col in df.columns]
            
            if available_indicators and 'Diabetes' in df.columns:
                selected_indicator = st.selectbox(
                    "Select health indicator:", 
                    available_indicators,
                    key="health_indicator"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Box plot
                    fig_box = px.box(
                        df, x='Diabetes', y=selected_indicator,
                        title=f"{selected_indicator} by Diabetes Status",
                        labels={'Diabetes': 'Diabetes Status'},
                        color='Diabetes',
                        color_discrete_map={0: '#90EE90', 1: '#67ABF8', 2: '#FF6B6B'}
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                with col2:
                    # Violin plot
                    fig_violin = px.violin(
                        df, x='Diabetes', y=selected_indicator,
                        title=f"{selected_indicator} Distribution",
                        labels={'Diabetes': 'Diabetes Status'},
                        color='Diabetes',
                        color_discrete_map={0: '#90EE90', 1: '#67ABF8', 2: '#FF6B6B'}
                    )
                    st.plotly_chart(fig_violin, use_container_width=True)
            
            # Risk factors analysis
            if 'Diabetes' in df.columns:
                st.subheader("Risk Factors Analysis")
                
                risk_factors = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 
                               'PhysActivity', 'Depression', 'KidneyDisease', 'COPD']
                available_risks = [col for col in risk_factors if col in df.columns]
                
                if available_risks:
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
                            'Diabetes': diabetes_rate,
                            'Difference (Yes-No)': diabetes_rate - no_diabetes_rate
                        })
                    
                    risk_df = pd.DataFrame(risk_data)
                    
                    # Risk factors comparison chart
                    fig_risk = px.bar(
                        risk_df, 
                        x='Factor', 
                        y=['No Diabetes', 'Pre-diabetes', 'Diabetes'],
                        title="Risk Factor Prevalence by Diabetes Status (%)",
                        barmode='group',
                        color_discrete_map={
                            'No Diabetes': '#90EE90', 
                            'Pre-diabetes': '#67ABF8', 
                            'Diabetes': '#FF6B6B'
                        }
                    )
                    fig_risk.update_layout(height=500, xaxis_tickangle=-45)
                    st.plotly_chart(fig_risk, use_container_width=True)
            
            # Correlation analysis
            if 'Diabetes' in df.columns:
                st.subheader("Feature Correlation")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if 'Diabetes' in numeric_cols:
                    # Calculate correlations with diabetes
                    correlations = df[numeric_cols].corr()['Diabetes'].abs().sort_values(ascending=False)[1:11]
                    
                    fig_corr = px.bar(
                        x=correlations.values, 
                        y=correlations.index,
                        title="Top 10 Features Correlated with Diabetes",
                        labels={'x': 'Absolute Correlation', 'y': 'Feature'},
                        color=correlations.values,
                        color_continuous_scale='Reds'
                    )
                    fig_corr.update_layout(height=500)
                    st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.subheader("Data Viewer with Pagination")
        
        # Pagination controls
        total_pages = (len(df) - 1) // st.session_state.page_size + 1
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("First") and st.session_state.current_page > 0:
                st.session_state.current_page = 0
                st.rerun()
        
        with col2:
            if st.button("Previous") and st.session_state.current_page > 0:
                st.session_state.current_page -= 1
                st.rerun()
        
        with col3:
            st.markdown(f"Page {st.session_state.current_page + 1} of {total_pages}")
            
            # Page size selector
            new_page_size = st.selectbox(
                "Rows per page:", 
                [100, 500, 1000, 2000],
                index=2,
                key="page_size_selector"
            )
            if new_page_size != st.session_state.page_size:
                st.session_state.page_size = new_page_size
                st.session_state.current_page = 0
                st.rerun()
        
        with col4:
            if st.button("Next") and st.session_state.current_page < total_pages - 1:
                st.session_state.current_page += 1
                st.rerun()
        
        with col5:
            if st.button("Last") and st.session_state.current_page < total_pages - 1:
                st.session_state.current_page = total_pages - 1
                st.rerun()
        
        # Display current page data
        page_data = paginate_dataframe(df, st.session_state.page_size, st.session_state.current_page)
        
        st.info(f"Showing rows {st.session_state.current_page * st.session_state.page_size + 1} to {min((st.session_state.current_page + 1) * st.session_state.page_size, len(df))} of {len(df)}")
        
        st.dataframe(
            page_data,
            height=400,
            width='stretch'
        )
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Full Dataset CSV",
            data=csv,
            file_name="brfss_dataset.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.subheader("🔍 Data Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Missing Data by Column:**")
            missing_data = stats['missing_data']
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) > 0:
                fig_missing = px.bar(
                    x=missing_data.values, 
                    y=missing_data.index,
                    title="Missing Data Count",
                    labels={'x': 'Count', 'y': 'Column'},
                    color=missing_data.values,
                    color_continuous_scale='Reds'
                )
                fig_missing.update_layout(height=400)
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("No missing data found!")
        
        with col2:
            st.markdown("**Basic Statistics:**")
            # Show stats for numeric columns only (lighter computation)
            numeric_df = df.select_dtypes(include=[np.number])
            stats_df = numeric_df.describe().round(2)
            st.dataframe(stats_df, height=400)
        
        # Data types information
        st.subheader("Dataset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Types:**")
            # Create Arrow-compatible dtype info
            dtype_info = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.astype(str).values, 
                'Non-Null Count': df.count().values,
                'Memory Usage (MB)': (df.memory_usage(deep=True).values[1:] / 1024**2).round(2)
            })
            st.dataframe(dtype_info, height=400)
        
        with col2:
            st.markdown("**Memory Usage Summary:**")
            total_memory = df.memory_usage(deep=True).sum()
            st.metric("Total Memory Usage", f"{total_memory / 1024**2:.2f} MB")
            
            # Show largest columns by memory
            memory_by_col = df.memory_usage(deep=True)[1:].sort_values(ascending=False).head(10)
            fig_memory = px.bar(
                x=memory_by_col.values / 1024**2,
                y=memory_by_col.index,
                title="Top 10 Columns by Memory Usage (MB)",
                labels={'x': 'Memory (MB)', 'y': 'Column'}
            )
            fig_memory.update_layout(height=400)
            st.plotly_chart(fig_memory, use_container_width=True)

else:
    st.error("Unable to load data. Please check the file path in config.py")

# Add sidebar information
add_sidebar_info()