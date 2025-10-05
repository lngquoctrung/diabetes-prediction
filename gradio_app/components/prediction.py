import gradio as gr
import pandas as pd
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.model_utils import load_trained_model_and_scaler, make_prediction, assess_risk_level
from utils.feature_engineering import create_feature_dataframe


def predict_diabetes(high_bp, high_chol, bmi, smoker, stroke, heart_disease, 
                     phys_activity, alcohol, gen_hlth, ment_hlth, phys_hlth, 
                     diff_walk, age, education, income):
    """Make diabetes prediction based on user inputs"""
    
    # Load model and scaler
    model, scaler = load_trained_model_and_scaler()
    
    if model is None or scaler is None:
        return "Error: Could not load model", None, None
    
    # Prepare user inputs - map to exact feature names
    user_inputs = {
        'high_bp': high_bp,
        'high_chol': high_chol,
        'bmi': bmi,
        'smoker': smoker,
        'stroke': stroke,
        'heart_disease': heart_disease,
        'phys_activity': phys_activity,
        'gen_hlth': gen_hlth,
        'ment_hlth': ment_hlth,
        'phys_hlth': phys_hlth,
        'diff_walk': diff_walk,
        'age': age,
        'education': education,
        'income': income,
        'alcohol_days': 30 if alcohol == 1 else 0,
        'depression': 1 if ment_hlth > 14 else 0,
        'cognitive_issues': 1 if ment_hlth > 20 else 0,
        'chol_meds': 1 if high_chol == 1 else 0,
        'has_doctor': 1,
        'last_checkup': 1,
        'employment': 1 if age < 65 else 2,
        'kidney_disease': 0,
        'copd': 0
    }
    
    # Create feature DataFrame
    feature_df = create_feature_dataframe(user_inputs)
    
    # Make prediction
    prediction_class, probabilities = make_prediction(model, scaler, feature_df)
    
    # Assess risk
    risk_level, recommendation = assess_risk_level(prediction_class, probabilities)
    
    # Format results
    result_text = f"""
    ## Prediction Results
    
    **Risk Level:** {risk_level}
    
    **Prediction:** {prediction_class}
    
    **Probabilities:**
    - No Diabetes: {probabilities[0]:.2%}
    - Pre-diabetes: {probabilities[1]:.2%}
    - Diabetes: {probabilities[2]:.2%}
    
    **Recommendation:**
    {recommendation}
    
    ---
    *Results are for reference only. Please consult a medical professional.*
    """
    
    # Create probability DataFrame
    prob_df = pd.DataFrame({
        "Class": ["No Diabetes", "Pre-diabetes", "Diabetes"],
        "Probability": [f"{p:.2%}" for p in probabilities]
    })
    
    return result_text, prob_df, risk_level



def create_prediction_tab():
    """Create Prediction tab content"""
    
    with gr.Column():
        gr.Markdown(
            """
            ## 🔮 Diabetes Risk Prediction
            
            Enter comprehensive health information to predict diabetes risk using our 
            **XGBoost + Random Oversampling** model (F1-Score: 69.92%)
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Health Information")
                
                # Basic health metrics
                with gr.Group():
                    gr.Markdown("**Basic Metrics**")
                    high_bp = gr.Radio(
                        choices=[0, 1],
                        label="High Blood Pressure",
                        value=0,
                        info="0 = No, 1 = Yes"
                    )
                    high_chol = gr.Radio(
                        choices=[0, 1],
                        label="High Cholesterol",
                        value=0,
                        info="0 = No, 1 = Yes"
                    )
                    bmi = gr.Slider(
                        minimum=10,
                        maximum=60,
                        value=25,
                        step=0.1,
                        label="BMI (Body Mass Index)"
                    )
                
                # Lifestyle factors
                with gr.Group():
                    gr.Markdown("**Lifestyle Factors**")
                    smoker = gr.Radio(
                        choices=[0, 1],
                        label="Smoker",
                        value=0,
                        info="0 = No, 1 = Yes"
                    )
                    phys_activity = gr.Radio(
                        choices=[0, 1],
                        label="Physical Activity (past 30 days)",
                        value=1,
                        info="0 = No, 1 = Yes"
                    )
                    alcohol = gr.Radio(
                        choices=[0, 1],
                        label="Heavy Alcohol Consumption",
                        value=0,
                        info="0 = No, 1 = Yes"
                    )
                
                # Medical history
                with gr.Group():
                    gr.Markdown("**Medical History**")
                    stroke = gr.Radio(
                        choices=[0, 1],
                        label="History of Stroke",
                        value=0,
                        info="0 = No, 1 = Yes"
                    )
                    heart_disease = gr.Radio(
                        choices=[0, 1],
                        label="Heart Disease or Attack",
                        value=0,
                        info="0 = No, 1 = Yes"
                    )
                    diff_walk = gr.Radio(
                        choices=[0, 1],
                        label="Difficulty Walking",
                        value=0,
                        info="0 = No, 1 = Yes"
                    )
            
            with gr.Column(scale=1):
                # Health status
                with gr.Group():
                    gr.Markdown("**Health Status**")
                    gen_hlth = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="General Health (1=Excellent, 5=Poor)"
                    )
                    ment_hlth = gr.Slider(
                        minimum=0,
                        maximum=30,
                        value=0,
                        step=1,
                        label="Mental Health (days not good in past 30 days)"
                    )
                    phys_hlth = gr.Slider(
                        minimum=0,
                        maximum=30,
                        value=0,
                        step=1,
                        label="Physical Health (days not good in past 30 days)"
                    )
                
                # Demographics
                with gr.Group():
                    gr.Markdown("**Demographics**")
                    age = gr.Slider(
                        minimum=18,
                        maximum=80,
                        value=40,
                        step=1,
                        label="Age"
                    )
                    education = gr.Slider(
                        minimum=1,
                        maximum=6,
                        value=4,
                        step=1,
                        label="Education Level (1=Elementary, 6=College Graduate)"
                    )
                    income = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=5,
                        step=1,
                        label="Income Level (1=<$10k, 8=>$75k)"
                    )
                
                # Predict button
                predict_btn = gr.Button(
                    "🔮 Predict Diabetes Risk",
                    variant="primary",
                    size="lg"
                )
        
        gr.Markdown("---")
        
        # Results section
        with gr.Row():
            with gr.Column(scale=2):
                result_output = gr.Markdown(label="Prediction Results")
            
            with gr.Column(scale=1):
                prob_output = gr.DataFrame(label="Probability Breakdown")
                risk_output = gr.Textbox(label="Risk Assessment", lines=2)
        
        # Connect predict button
        predict_btn.click(
            fn=predict_diabetes,
            inputs=[
                high_bp, high_chol, bmi, smoker, stroke, heart_disease,
                phys_activity, alcohol, gen_hlth, ment_hlth, phys_hlth,
                diff_walk, age, education, income
            ],
            outputs=[result_output, prob_output, risk_output]
        )
