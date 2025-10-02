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

from utils.ui_components import setup_page_config, add_custom_css, add_sidebar_info
import pandas as pd

# Page setup
setup_page_config()
add_custom_css()

st.header("Project Details")

# Project overview
st.subheader("Project Objectives")
st.markdown("""
- **Main Goal**: Build machine learning models to predict diabetes risk with high accuracy
- **Dataset**: BRFSS (Behavioral Risk Factor Surveillance System) 2017-2023
- **Scope**: 3-class classification - No Diabetes, Pre-diabetes, Diabetes
- **Challenge**: Address severe class imbalance (82% No Diabetes, 2.4% Pre-diabetes, 15.6% Diabetes)
""")

# Technical details
st.subheader("Technical Details")

tab1, tab2, tab3, tab4 = st.tabs(["Data Pipeline", "Feature Engineering", "Data Balancing", "Models"])

with tab1:
    st.markdown("""
    ### Data Pipeline
    
    **1. Data Loading & Cleaning**
    - Load and process BRFSS data from 2017-2023
    - Remove missing values and duplicate records
    - Mapping and encoding features
    
    **2. Data Quality**  
    - Original samples: 452,655
    - After cleaning: 450,445 samples
    - Final features: 34 (from 43 originally)
    
    **3. Feature Distribution**
    - No Diabetes: 82% (355,421 samples)
    - Pre-diabetes: 2.4% (11,985 samples) 
    - Diabetes: 15.6% (83,039 samples)
    
    **4. Data Challenges**
    - Severe class imbalance requiring synthetic sampling
    - Pre-diabetes class extremely underrepresented
    - High dimensionality requiring feature selection
    """)

with tab2:
    st.markdown("""
    ### Feature Engineering
    
    **1. Composite Health Scores**
    - **Health Score**: Combines GenHlth, MentHlth, PhysHlth, DiffWalk
    - **Risk Score**: Combines 8 chronic disease risk factors  
    - **Lifestyle Score**: Combines physical activity and bad habits
    - **Cardio Risk**: Combines cardiovascular risk factors
    
    **2. Healthcare & Socioeconomic Scores**
    - **Healthcare Access Score**: Access to healthcare services
    - **Socioeconomic Score**: Socioeconomic status indicators
    - **Cholesterol Management Score**: Cholesterol management effectiveness
    - **Mental Health Score**: Mental health status indicators
    
    **3. Categorical Features**
    - BMI Categories: 6 groups from Underweight to Obesity Class III
    - Age Groups: Young (18-44), Middle (45-64), Senior (65+)
    
    **4. Feature Selection Results**
    - Most Important: PhysHlth (0.44), HlthScore (0.40), CardioRisk (0.34)
    - Correlation-based selection reduced features from 43 to 34
    """)

with tab3:
    st.markdown("""
    ### Data Balancing Techniques
    
    **1. Oversampling Methods**
    - **Random Oversampling**: Simple duplication of minority samples
    - **SMOTE**: Synthetic Minority Oversampling Technique
    - **ADASYN**: Adaptive Synthetic Sampling (focuses on hard examples)
    
    **2. Undersampling Methods**  
    - **Random Undersampling**: Random removal of majority samples
    - **Tomek Links**: Remove borderline majority samples
    - **Edited Nearest Neighbors (ENN)**: Remove noisy samples
    
    **3. Hybrid Methods**
    - **SMOTE + Tomek Links**: Oversample then clean boundaries
    - **SMOTE + ENN**: Oversample then remove noise
    - **ADASYN + Tomek Links**: Adaptive oversampling + boundary cleaning
    
    **4. Best Results**
    - **Accuracy**: ADASYN Tomek Links (90.11%)
    - **F1-Score**: Random Oversampling (69.92%)
    - **Precision**: ADASYN Tomek Links (91.67%)
    """)

with tab4:
    st.markdown("""
    ### Machine Learning Models
    
    **1. Algorithms Tested**
    - Logistic Regression (baseline)
    - Random Forest (ensemble)
    - XGBoost (gradient boosting) 
    - LightGBM (fast gradient boosting)
    - Naive Bayes (probabilistic)
    - SGD Classifier (linear)
    
    **2. Best Performing Models**
    - **Best Accuracy**: LightGBM + ADASYN Tomek Links (90.11%)
    - **Best F1-Score**: XGBoost + Random Oversampling (69.92%)
    - **Best Precision**: LightGBM + ADASYN Tomek Links (91.67%) 
    - **Best Recall**: Multiple models (62.47%)
    
    **3. Model Selection Criteria**
    - 5-fold cross-validation for robust evaluation
    - Multiple metrics optimization (Accuracy, Precision, Recall, F1)
    - Stability across different data balancing methods
    - Computational efficiency for deployment
    
    **4. Key Findings**
    - Gradient boosting methods (XGBoost, LightGBM) perform best
    - Data balancing is crucial for minority class detection
    - Ensemble methods show superior generalization
    """)

# Performance summary - Updated with actual results
st.subheader("Performance Summary")

performance_summary = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Best Score': ['90.11%', '91.67%', '62.47%', '69.92%'],
    'Model': ['LightGBM', 'LightGBM', 'Multiple', 'XGBoost'],
    'Method': ['ADASYN Tomek Links', 'ADASYN Tomek Links', '-', 'Random Oversampling']
}

st.table(pd.DataFrame(performance_summary))

# Detailed comparison table
st.subheader("Detailed Model Comparison")

detailed_results = {
    'Model + Method': [
        'LightGBM + ADASYN Tomek Links',
        'XGBoost + Random Oversampling', 
        'LightGBM + SMOTE Tomek Links',
        'XGBoost + SMOTE',
        'Random Forest + Random Undersampling'
    ],
    'Accuracy': ['90.11%', '89.55%', '90.09%', '89.95%', '64.88%'],
    'Precision': ['91.67%', '77.70%', '91.13%', '89.70%', '44.65%'],
    'Recall': ['62.47%', '65.52%', '62.47%', '62.52%', '54.55%'],
    'F1-Score': ['69.02%', '69.92%', '69.02%', '68.75%', '44.16%'],
    'ROC AUC': ['92.33%', '90.89%', '92.31%', '92.09%', '72.15%']
}

st.table(pd.DataFrame(detailed_results))

# Recommendations
st.subheader("Recommendations")
st.markdown("""
**1. Production Deployment**
- **Primary Model**: LightGBM + ADASYN Tomek Links for highest accuracy (90.11%)
- **Balanced Alternative**: XGBoost + Random Oversampling for better F1-Score (69.92%)

**2. Use Case Considerations**
- **High Accuracy Required**: Use LightGBM + ADASYN Tomek Links
- **Balanced Precision-Recall**: Use XGBoost + Random Oversampling
- **Cost of False Negatives High**: Consider models with higher recall

**3. Future Improvements**
- **Data Collection**: Focus on collecting more pre-diabetes samples
- **Feature Engineering**: Incorporate additional domain knowledge
- **Ensemble Methods**: Combine multiple models for better performance
- **Real-time Updates**: Implement continuous learning from new data

**4. Model Monitoring**
- Track performance degradation over time
- Monitor class distribution shifts
- Regular retraining with new BRFSS data
- A/B testing for model updates
""")

# Add sidebar information
add_sidebar_info()
