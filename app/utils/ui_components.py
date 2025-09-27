import streamlit as st

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Diabetes Prediction Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def add_custom_css():
    """Add custom CSS styling"""
    st.markdown("""
    <style>
        /* Main content styling */
        body, .main {
            color: #222 !important;
            padding-top: 1rem;
        }
        
        /* Metric card styling */
        .stMetric {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            border-left: 5px solid #1f77b4;
            color: #000 !important;
        }
        
        /* Prediction box styling */
        .prediction-box {
            background: linear-gradient(90deg, #ffffff 0%, #eaeaea 100%);
            padding: 20px;
            border-radius: 15px;
            color: black;
            margin: 20px 0;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f0f2f6;
        }
        
        /* Navigation styling */
        .css-17lntkn {
            font-weight: 600;
            color: #1f77b4;
        }
        
        /* Page link styling */
        .css-pkbazv {
            font-weight: 500;
        }
        
        /* Sidebar header */
        .css-1lcbmhc {
            background-color: #1f77b4;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        div[data-testid="stMetric"] * {
            color: #000 !important;
        }
    </style>
    """, unsafe_allow_html=True)

def add_sidebar_info():
    """Add enhanced information to sidebar"""
    st.sidebar.markdown("---")
    
    # Enhanced sidebar with better styling
    st.sidebar.markdown("""
    <div style="background-color: #1f77b4; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">🏥 Diabetes Prediction Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 📊 **Model Information**")
    st.sidebar.info("""
    **🤖 Algorithm**: XGBoost + Random Oversampling  
    **📈 F1-Score**: 69.92%  
    **🎯 Accuracy**: 89.5%  
    **📋 Features**: 34 engineered features
    """)
    
    st.sidebar.markdown("### 📚 **Data Source**")
    st.sidebar.success("""
    **BRFSS 2017-2023**  
    450,445 samples  
    Behavioral Risk Factor Surveillance System
    """)
    
    st.sidebar.markdown("### 🚀 **Quick Start**")
    st.sidebar.markdown("""
    1. **🤖 Prediction**: Enter health data for risk assessment
    2. **📈 Model Analysis**: Compare model performance 
    3. **📋 Project Details**: Technical methodology
    """)

# Rest of your functions remain the same...
def create_metric_card(label, value, delta=None, help=None):
    """Create a custom metric card"""
    st.metric(
        label=label,
        value=value,
        delta=delta,
        help=help
    )

def create_prediction_result_box(risk_level, prediction_class, probability, recommendation):
    """Create prediction result display box"""
    color_map = {
        "LOW": "green",
        "MODERATE": "orange", 
        "HIGH": "red"
    }
    
    color = color_map.get(risk_level, "blue")
    
    st.markdown(f"""
    <div class="prediction-box">
        <h3>🎯 Risk Assessment: <span style="color: {color}; font-weight: bold;">{risk_level}</span></h3>
        <p style="font-size: 18px;">Prediction: <strong>{prediction_class}</strong> ({probability:.1%})</p>
        <p style="font-size: 16px;">💡 Recommendation: {recommendation}</p>
        <p style="font-size: 14px;">⚠️ <em>Results are for reference only. Please consult a medical professional.</em></p>
    </div>
    """, unsafe_allow_html=True)

def create_health_input_form():
    """Create comprehensive health input form"""
    with st.form("comprehensive_prediction_form"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.subheader("📊 Basic Information")
            age = st.slider("Age", 18, 80, 45, help="Your age")
            bmi = st.number_input("BMI", 15.0, 50.0, 25.0, step=0.1, help="Body Mass Index")
            genhlth = st.selectbox("General Health", [1, 2, 3, 4, 5], 
                                 format_func=lambda x: {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}[x])
            education = st.selectbox("Education Level", [1, 2, 3, 4, 5, 6],
                                   format_func=lambda x: {1: "No Education", 2: "Elementary", 3: "Middle School", 4: "High School", 5: "College", 6: "University"}[x])
            income = st.selectbox("Income Level", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                format_func=lambda x: {1: "<$10K", 2: "$10-15K", 3: "$15-20K", 4: "$20-25K", 5: "$25-35K", 
                                                      6: "$35-50K", 7: "$50-75K", 8: "$75-100K", 9: "$100-150K", 10: "$150-200K", 11: ">$200K"}[x])
        
        with col2:
            st.subheader("🩺 Health Conditions")
            high_bp = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x else "No")
            high_chol = st.selectbox("High Cholesterol", [0, 1], format_func=lambda x: "Yes" if x else "No")
            chol_meds = st.selectbox("Cholesterol Medications", [0, 1], format_func=lambda x: "Yes" if x else "No")
            stroke = st.selectbox("History of Stroke", [0, 1], format_func=lambda x: "Yes" if x else "No")
            heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
            kidney_disease = st.selectbox("Kidney Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
            copd = st.selectbox("COPD", [0, 1], format_func=lambda x: "Yes" if x else "No")
        
        with col3:
            st.subheader("🏃 Lifestyle & Behavior")
            phys_activity = st.selectbox("Physical Activity", [0, 1], format_func=lambda x: "Yes" if x else "No")
            smoker = st.selectbox("Smoker", [0, 1], format_func=lambda x: "Yes" if x else "No")
            alcohol_days = st.slider("Alcohol Days per Month", 0, 30, 0)
            diff_walk = st.selectbox("Difficulty Walking", [0, 1], format_func=lambda x: "Yes" if x else "No")
            employment = st.selectbox("Employment Status", [1, 2, 3, 4, 5, 6, 7, 8],
                                    format_func=lambda x: {1: "Employed", 2: "Self-employed", 3: "Unemployed >1yr", 
                                                          4: "Unemployed <1yr", 5: "Homemaker", 6: "Student", 7: "Retired", 8: "Unable to work"}[x])
        
        with col4:
            st.subheader("🧠 Mental Health")
            ment_hlth = st.slider("Poor Mental Health Days/Month", 0, 30, 0)
            phys_hlth = st.slider("Poor Physical Health Days/Month", 0, 30, 0)
            depression = st.selectbox("Depression", [0, 1], format_func=lambda x: "Yes" if x else "No")
            cognitive_issues = st.selectbox("Cognitive Issues", [0, 1], format_func=lambda x: "Yes" if x else "No")
            last_checkup = st.selectbox("Last Medical Checkup", [0, 1, 2, 3],
                                      format_func=lambda x: {0: ">5 years/Never", 1: "Within 1 year", 2: "1-2 years", 3: "2-5 years"}[x])
            has_doctor = st.selectbox("Has Personal Doctor", [0, 1], format_func=lambda x: "Yes" if x else "No")
        
        submit = st.form_submit_button("🔍 Predict with XGBoost", width="stretch")
        
        if submit:
            return {
                'age': age, 'bmi': bmi, 'genhlth': genhlth, 'education': education, 'income': income,
                'high_bp': high_bp, 'high_chol': high_chol, 'chol_meds': chol_meds, 'stroke': stroke,
                'heart_disease': heart_disease, 'kidney_disease': kidney_disease, 'copd': copd,
                'phys_activity': phys_activity, 'smoker': smoker, 'alcohol_days': alcohol_days,
                'diff_walk': diff_walk, 'employment': employment, 'ment_hlth': ment_hlth,
                'phys_hlth': phys_hlth, 'depression': depression, 'cognitive_issues': cognitive_issues,
                'last_checkup': last_checkup, 'has_doctor': has_doctor
            }
    
    return None

def add_sidebar_info():
    """Add information to sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Diabetes Prediction Dashboard**")
    st.sidebar.markdown("Built with Streamlit")
    st.sidebar.markdown("XGBoost + Random Oversampling") 
    st.sidebar.markdown("Based on BRFSS 2017-2023 Data")
