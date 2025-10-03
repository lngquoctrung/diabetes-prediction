import streamlit as st
import sys
import os

from pathlib import Path

root_dir = str(Path(__file__).parent.parent.parent.absolute())
dashboard_dir = os.path.join(root_dir, "app")
if not dashboard_dir in sys.path:
    sys.path.insert(0, dashboard_dir)
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

from utils.ui_components import (
    setup_page_config, add_custom_css, create_health_input_form, 
    create_prediction_result_box, add_sidebar_info
)
from utils.feature_engineering import create_feature_dataframe
from utils.model_utils import load_trained_model_and_scaler, make_prediction, assess_risk_level, get_model_info
from utils.visualization import create_prediction_chart

# Page setup
setup_page_config()
add_custom_css()

st.header("Diabetes Prediction")
st.markdown("Enter comprehensive health information to accurately predict diabetes risk using XGBoost + Random Oversampling model")

# Load model and scaler
model, scaler = load_trained_model_and_scaler()

if model is None or scaler is None:
    st.error("Unable to load model or scaler! Please check the file paths.")
    st.stop()

# Create health input form
user_inputs = create_health_input_form()

if user_inputs:
    # Create feature dataframe
    feature_df = create_feature_dataframe(user_inputs)
    
    # Make prediction
    predictions = make_prediction(model, scaler, feature_df)
    
    if predictions:
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results from XGBoost + Random Oversampling")
        
        # Create prediction visualization
        fig_pred = create_prediction_chart(predictions)
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Risk assessment
        risk_level, max_class, max_pred, recommendation = assess_risk_level(predictions)
        create_prediction_result_box(risk_level, max_class, max_pred, recommendation)
        
        # Display model information
        model_info = get_model_info()
        st.info(f"""**Model Used**: {model_info['model_name']}
                    **Accuracy**: {model_info['accuracy']}
                    **F1-Score**: {model_info['f1_score']}
                    **Trained on**: {model_info['training_samples']} {model_info['data_source']} data samples""")

# Add sidebar information
add_sidebar_info()
