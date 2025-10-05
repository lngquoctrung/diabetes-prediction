import gradio as gr
import pandas as pd


def create_project_details_tab():
    """Create Project Details tab content with accurate project information"""
    
    with gr.Column():
        gr.Markdown(
            """
            ## Diabetes Prediction Project
            
            Machine learning project for 3-class diabetes prediction using BRFSS (Behavioral Risk Factor Surveillance System) 
            dataset with comprehensive data balancing techniques and advanced feature engineering.
            """
        )
        
        # Project objectives
        with gr.Accordion("Project Objectives", open=True):
            gr.Markdown(
                """
                ### Main Goals
                
                - **Primary Objective**: Build machine learning models to predict diabetes risk with high accuracy
                - **Dataset Source**: BRFSS (Behavioral Risk Factor Surveillance System) 2017, 2019, 2021, 2023
                - **Classification Type**: 3-class classification
                  - **Class 0**: No Diabetes (82.4% of dataset)
                  - **Class 1**: Pre-diabetes (2.4% of dataset)
                  - **Class 2**: Diabetes (15.6% of dataset)
                - **Challenge**: Severe class imbalance requiring advanced balancing techniques
                - **Target Users**: Healthcare professionals, researchers, individuals monitoring health risks
                
                ### Success Criteria
                
                - Achieve >85% accuracy on test set
                - Maintain balanced performance across all 3 classes
                - Optimize F1-Score for balanced precision and recall
                - Deploy best-performing model for real-time predictions
                """
            )
        
        # Technical stack
        with gr.Accordion("Technical Stack", open=True):
            tech_data = pd.DataFrame({
                "Category": [
                    "Machine Learning",
                    "Data Processing",
                    "Data Balancing",
                    "Visualization",
                    "Web Framework",
                    "Development"
                ],
                "Technologies": [
                    "XGBoost, LightGBM, Random Forest, Logistic Regression, Naive Bayes",
                    "Pandas, NumPy, Scikit-learn",
                    "SMOTE, ADASYN, Tomek Links, ENN, Random Over/Under Sampling",
                    "Plotly, Matplotlib, Seaborn",
                    "Gradio, Streamlit",
                    "Python 3.9+, Jupyter Notebook"
                ]
            })
            gr.DataFrame(value=tech_data, label="Technology Stack", interactive=False)
        
        # Dataset information
        with gr.Accordion("Dataset Information", open=True):
            gr.Markdown(
                """
                ### BRFSS Dataset Statistics
                
                **Original Dataset:**
                - **2017**: 450,016 samples, 358 features
                - **2019**: 418,268 samples, 342 features
                - **2021**: 438,693 samples, 303 features
                - **2023**: 441,456 samples, 328 features
                - **Total**: 1,748,433 samples before processing
                
                **After Cleaning Pipeline:**
                - **Combined Dataset**: 863,745 samples, 33 features
                - **After removing null values**: 452,655 samples (47.59% rows removed)
                - **After removing duplicates**: 449,205 samples (3,450 duplicates removed)
                - **Final Training Dataset**: 449,205 samples, 33 selected features
                
                ### Data Split Strategy
                
                - **Training Data**: 2017, 2019, 2021 combined → 449,205 samples
                - **Testing Data**: 2023 dataset (separate for temporal validation)
                - **Validation**: 5-fold cross-validation on training data
                
                ### Class Distribution (After Cleaning)
                
                - **No Diabetes (Class 0)**: 369,364 samples (82.22%)
                - **Pre-diabetes (Class 1)**: 10,811 samples (2.41%)
                - **Diabetes (Class 2)**: 70,270 samples (15.64%)
                """
            )
        
        # Feature engineering
        with gr.Accordion("Feature Engineering", open=True):
            gr.Markdown(
                """
                ### Original Features (33 selected from 358 total)
                
                **Demographic Features:**
                - Age, Sex, Education, Income, Employment Status, Marital Status
                
                **Health Metrics:**
                - BMI, General Health, Mental Health Days, Physical Health Days
                - High Blood Pressure, High Cholesterol, Cholesterol Check
                
                **Lifestyle Factors:**
                - Physical Activity, Smoking Status, Alcohol Consumption, Heavy Alcohol Consumption
                
                **Medical History:**
                - Stroke, Heart Disease/Attack, Diagnosed Heart Attack, Coronary Heart Disease
                - COPD, Kidney Disease, Depression, Cognitive Issues
                
                **Healthcare Access:**
                - Has Personal Doctor, Last Checkup, Cannot Afford Doctor, Any Healthcare
                - Cholesterol Medications
                
                ### Engineered Features (6 new features)
                
                #### 1. Health Score (HlthScore)
                
                **Formula:**
                ```
                HlthScore = GenHlth/5 + MentHlth/30 + PhysHlth/30 + DiffWalk*0.5
                ```
                
                **Purpose**: Composite health indicator combining general health, mental health, physical health, and mobility.
                
                **Range**: 0.200 to 2.917
                
                ---
                
                #### 2. Risk Score (RiskScore)
                
                **Formula:**
                ```
                RiskScore = HighBP + HighChol + HeartDiseaseorAttack + Stroke + 
                            DiagnosedHeartAttack + CoronaryHeartDisease + COPD + KidneyDisease
                ```
                
                **Purpose**: Sum of chronic disease risk factors.
                
                **Range**: 0.0 to 8.0
                
                ---
                
                #### 3. Lifestyle Score (LifestyleScore)
                
                **Formula:**
                ```
                LifestyleScore = PhysActivity - (AlcoholDays/30 + Smoker)
                ```
                
                **Purpose**: Balance between positive (physical activity) and negative (smoking, drinking) lifestyle factors.
                
                **Range**: -1.333 to 1.000
                
                ---
                
                #### 4. Cardiovascular Risk (CardioRisk)
                
                **Formula:**
                ```
                CardioRisk = HighBP + HighChol + DiagnosedHeartAttack + 
                             CoronaryHeartDisease + (1 if BMI > 30 else 0)
                ```
                
                **Purpose**: Focused cardiovascular disease risk indicator.
                
                **Range**: 0.0 to 5.0
                
                ---
                
                #### 5. Cholesterol Management Score
                
                **Formula:**
                ```
                CholesterolManagementScore = CholCheck + CholesterolMeds - 
                                            (HighChol * (1 - CholesterolMeds * 0.5))
                ```
                
                **Purpose**: Measures proactive cholesterol management (checks + medication) vs. presence of high cholesterol.
                
                **Range**: -1.000 to 2.000
                
                ---
                
                #### 6. Mental Health Score
                
                **Formula:**
                ```
                MentalHealthScore = -(Depression + CognitiveIssues + MentHlth/30)
                ```
                
                **Purpose**: Comprehensive mental health assessment (negative values indicate worse mental health).
                
                **Range**: -2.417 to 0.000
                
                ### Feature Selection Process
                
                **Method**: Correlation-based selection
                - **Threshold**: Absolute correlation ≥ 0.05 with target (Diabetes)
                - **Selected Features**: 33 features out of 43 available
                - **Top 10 Features by Correlation**:
                  1. PhysHlth (0.4401)
                  2. HlthScore (0.3969)
                  3. CardioRisk (0.3386)
                  4. RiskScore (0.3054)
                  5. GenHlth (0.3029)
                  6. CholesterolMeds (0.2956)
                  7. HighBP (0.2645)
                  8. BMI (0.2483)
                  9. AlcoholDays (-0.2444)
                  10. DiffWalk (0.2193)
                """
            )
        
        # Data balancing methods
        with gr.Accordion("Data Balancing Methods (9 Techniques)", open=True):
            gr.Markdown(
                """
                ### Balancing Techniques Compared
                
                Due to severe class imbalance (82.2% / 2.4% / 15.6%), we tested 9 different balancing techniques:
                """
            )
            
            balancing_data = pd.DataFrame({
                "Method": [
                    "Random Oversampling",
                    "SMOTE",
                    "ADASYN",
                    "Random Undersampling",
                    "Tomek Links",
                    "Edited Nearest Neighbors",
                    "SMOTE + Tomek Links",
                    "SMOTE + ENN",
                    "ADASYN + Tomek Links"
                ],
                "Type": [
                    "Oversampling",
                    "Synthetic Oversampling",
                    "Adaptive Synthetic",
                    "Undersampling",
                    "Cleaning",
                    "Cleaning",
                    "Hybrid",
                    "Hybrid",
                    "Hybrid"
                ],
                "Training Samples": [
                    "433,416",
                    "225,948",
                    "429,230",
                    "225,948",
                    "6,516",
                    "254,273",
                    "425,286",
                    "421,740",
                    "426,981"
                ],
                "Best Model": [
                    "XGBoost",
                    "XGBoost",
                    "XGBoost",
                    "LightGBM",
                    "N/A",
                    "XGBoost",
                    "LightGBM",
                    "LightGBM",
                    "LightGBM"
                ],
                "Best F1-Score": [
                    "69.92%",
                    "68.22%",
                    "67.19%",
                    "59.03%",
                    "N/A",
                    "66.85%",
                    "68.98%",
                    "67.31%",
                    "68.14%"
                ]
            })
            gr.DataFrame(value=balancing_data, label="Data Balancing Techniques Comparison", interactive=False)
            
            gr.Markdown(
                """
                ### Key Findings
                
                **Best F1-Score**: Random Oversampling + XGBoost (69.92%)
                
                **Best Accuracy**: ADASYN + Tomek Links + LightGBM (90.11%)
                
                **Hybrid methods** (SMOTE/ADASYN + Tomek/ENN) provide balanced results
                
                **Random Undersampling** significantly reduces performance due to information loss
                
                **Tomek Links alone** has too few samples for effective training
                """
            )
        
        # Model training process
        with gr.Accordion("Model Training & Evaluation", open=True):
            gr.Markdown(
                """
                ### Training Pipeline
                
                **1. Data Preprocessing**
                - Download BRFSS data from CDC website (XPT format)
                - Clean features: handle missing values (47.59% removed), remove duplicates
                - Apply outlier handling using IQR method for BMI, MentHlth, PhysHlth, AlcoholDays
                
                **2. Feature Engineering**
                - Create 6 engineered features (HlthScore, RiskScore, etc.)
                - Encode categorical features (BMI Category, Age Group)
                - Select 33 features with correlation ≥ 0.05
                
                **3. Data Balancing**
                - Apply 9 different balancing techniques
                - Create separate datasets for each method
                - Maintain original test set for fair comparison
                
                **4. Model Training**
                - Train 5 algorithms per balancing method:
                  - XGBoost
                  - LightGBM
                  - Random Forest
                  - Logistic Regression
                  - Naive Bayes
                - Total: 45 models trained (9 methods × 5 algorithms)
                - 5-fold cross-validation for each model
                - Hyperparameters:
                  - XGBoost: n_estimators=100, max_depth=6, learning_rate=0.1
                  - LightGBM: n_estimators=100, max_depth=-1, num_leaves=31
                  - Random Forest: n_estimators=100, max_depth=25, min_samples_split=5
                
                **5. Model Evaluation Metrics**
                - **Accuracy**: Overall correctness
                - **Precision**: TP / (TP + FP)
                - **Recall (Sensitivity)**: TP / (TP + FN)
                - **F1-Score**: Harmonic mean of precision and recall
                - **ROC-AUC**: Area under ROC curve
                - **Per-class metrics**: Precision, Recall, F1 for each of 3 classes
                
                **6. Model Selection Criteria**
                - **Deployed Model**: XGBoost + Random Oversampling
                - **Reason**: Best F1-Score (69.92%) for balanced performance
                - **Alternative**: LightGBM + ADASYN Tomek for highest accuracy (90.11%)
                """
            )
        
        # Performance comparison
        with gr.Accordion("Top 5 Model Performance", open=True):
            top_models = pd.DataFrame({
                "Rank": [1, 2, 3, 4, 5],
                "Model + Method": [
                    "XGBoost + Random Oversampling",
                    "LightGBM + ADASYN Tomek Links",
                    "LightGBM + SMOTE Tomek Links",
                    "XGBoost + SMOTE",
                    "LightGBM + ADASYN Tomek"
                ],
                "Accuracy": ["89.55%", "90.11%", "90.09%", "89.95%", "90.12%"],
                "Precision": ["90.91%", "91.67%", "91.55%", "90.50%", "91.67%"],
                "Recall": ["60.32%", "62.47%", "62.36%", "61.77%", "62.47%"],
                "F1-Score": ["69.92% ⭐", "68.14%", "68.98%", "68.22%", "68.19%"],
                "ROC-AUC": ["92.35%", "92.33%", "92.31%", "92.14%", "92.08%"]
            })
            gr.DataFrame(value=top_models, label="Top 5 Models by F1-Score", interactive=False)
            
            gr.Markdown(
                """
                ### Why F1-Score Matters
                
                In medical diagnosis for diabetes prediction, we need **balance** between:
                
                - **High Recall**: Don't miss actual diabetes cases (minimize false negatives)
                - **High Precision**: Avoid unnecessary worry from false alarms (minimize false positives)
                
                **F1-Score** = Harmonic mean of Precision and Recall = Best metric for balanced performance
                
                ### Deployed Model Performance
                
                **XGBoost + Random Oversampling (Best F1)**
                
                - **Overall Metrics**:
                  - Accuracy: 89.55%
                  - Precision: 90.91%
                  - Recall: 60.32%
                  - F1-Score: 69.92%
                  - ROC-AUC: 92.35%
                
                - **Per-Class Performance**:
                  - **No Diabetes**: Precision 91.60%, Recall 96.32%, F1 93.90%
                  - **Pre-diabetes**: Precision 58.90%, Recall 31.33%, F1 40.90%
                  - **Diabetes**: Precision 80.95%, Recall 68.61%, F1 74.27%
                
                - **Cross-Validation**: Mean Accuracy 82.64% (±0.13%)
                """
            )
        
        # Future improvements
        with gr.Accordion("Future Improvements", open=False):
            gr.Markdown(
                """
                ### Planned Enhancements
                
                **Model Improvements:**
                - Ensemble methods combining top 3 models
                - Deep learning approaches (Neural Networks, TabNet)
                - AutoML for hyperparameter optimization
                - Time-series analysis using multiple years
                
                **Feature Engineering:**
                - Interaction features between key predictors
                - Polynomial features for non-linear relationships
                - Additional domain-specific health scores
                
                **Deployment:**
                - REST API for model serving
                - Docker containerization
                - Cloud deployment (AWS, GCP, Azure)
                - Real-time prediction monitoring
                
                **Explainability:**
                - SHAP values for feature importance
                - LIME for local interpretability
                - Counterfactual explanations
                """
            )
        
        gr.Markdown("---")
        
        # Project references
        with gr.Accordion("References & Resources", open=False):
            gr.Markdown(
                """
                ### Data Source
                
                - **BRFSS (Behavioral Risk Factor Surveillance System)**
                  - Website: https://www.cdc.gov/brfss/
                  - 2017 Data: https://www.cdc.gov/brfss/annual_data/2017/files/LLCP2017XPT.zip
                  - 2019 Data: https://www.cdc.gov/brfss/annual_data/2019/files/LLCP2019XPT.zip
                  - 2021 Data: https://www.cdc.gov/brfss/annual_data/2021/files/LLCP2021XPT.zip
                  - 2023 Data: https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip
                
                ### Key Libraries
                
                - **Machine Learning**: scikit-learn, XGBoost, LightGBM, imbalanced-learn
                - **Data Processing**: pandas, numpy
                - **Visualization**: plotly, matplotlib, seaborn
                - **Web Framework**: gradio, streamlit
                
                ### Project Structure
                
                ```
                diabetes-prediction-project/
                ├── data/               # Raw and processed data
                ├── notebooks/          # Jupyter notebooks for exploration
                ├── src/               # Source code
                │   ├── config.py      # Configuration settings
                │   ├── pipelines/     # Data processing pipelines
                │   ├── features/      # Feature engineering
                │   ├── models/        # Model implementations
                │   └── utils/         # Utility functions
                ├── app/               # Gradio web application
                ├── models/            # Saved trained models
                └── requirements.txt   # Python dependencies
                ```
                """
            )
