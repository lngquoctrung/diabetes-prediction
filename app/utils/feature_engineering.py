import pandas as pd
import numpy as np

def create_feature_dataframe(user_inputs):
    """
    Create DataFrame with all 34 features needed for the model
    """
    # List of all 34 features in training data order
    all_features = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Age', 'Education', 'Income',
        'Depression', 'CognitiveIssues', 'KidneyDisease', 'DiagnosedHeartAttack',
        'CoronaryHeartDisease', 'COPD', 'AlcoholDays', 'LastCheckup',
        'HasPersonalDoctor', 'CholesterolMeds', 'EmploymentStatus',
        'HlthScore', 'RiskScore', 'LifestyleScore', 'CardioRisk',
        'CholesterolManagementScore', 'MentalHealthScore'
    ]
    
    # Initialize feature dictionary
    feature_data = {}
    
    # Map user inputs to features
    feature_data.update({
        'HighBP': user_inputs.get('high_bp', 0),
        'HighChol': user_inputs.get('high_chol', 0),
        'BMI': user_inputs.get('bmi', 25.0),
        'Smoker': user_inputs.get('smoker', 0),
        'Stroke': user_inputs.get('stroke', 0),
        'HeartDiseaseorAttack': user_inputs.get('heart_disease', 0),
        'PhysActivity': user_inputs.get('phys_activity', 1),
        'GenHlth': user_inputs.get('genhlth', 3),
        'Age': user_inputs.get('age', 45),
        'Education': user_inputs.get('education', 4),
        'Income': user_inputs.get('income', 6),
        'Depression': user_inputs.get('depression', 0),
        'CognitiveIssues': user_inputs.get('cognitive_issues', 0),
        'KidneyDisease': user_inputs.get('kidney_disease', 0),
        'COPD': user_inputs.get('copd', 0),
        'MentHlth': user_inputs.get('ment_hlth', 0),
        'PhysHlth': user_inputs.get('phys_hlth', 0),
        'DiffWalk': user_inputs.get('diff_walk', 0),
        'AlcoholDays': user_inputs.get('alcohol_days', 0),
        'LastCheckup': user_inputs.get('last_checkup', 1),
        'HasPersonalDoctor': user_inputs.get('has_doctor', 1),
        'CholesterolMeds': user_inputs.get('chol_meds', 0),
        'EmploymentStatus': user_inputs.get('employment', 1)
    })
    
    # Set default values for missing features
    defaults = {
        'CholCheck': 1, 'HvyAlcoholConsump': 0, 'DiagnosedHeartAttack': 0,
        'CoronaryHeartDisease': 0,
    }
    
    for feature in all_features:
        if feature not in feature_data:
            feature_data[feature] = defaults.get(feature, 0)
    
    # Calculate engineered features
    feature_data = calculate_engineered_features(feature_data)
    
    # Create DataFrame with correct feature order
    df = pd.DataFrame([feature_data])[all_features]
    return df

def calculate_engineered_features(feature_data):
    """Calculate engineered features based on base features"""
    # Health Score: combination of general health indicators
    health_score = (
        feature_data['GenHlth'] / 5 + 
        feature_data['MentHlth'] / 30 + 
        feature_data['PhysHlth'] / 30 + 
        feature_data['DiffWalk'] * 0.5
    )
    feature_data['HlthScore'] = health_score
    
    # Risk Score: sum of chronic disease risk factors
    risk_factors = [
        'HighBP', 'HighChol', 'HeartDiseaseorAttack', 'Stroke',
        'DiagnosedHeartAttack', 'CoronaryHeartDisease', 'COPD', 'KidneyDisease'
    ]
    risk_score = sum(feature_data[rf] for rf in risk_factors)
    feature_data['RiskScore'] = risk_score
    
    # Lifestyle Score: positive vs negative lifestyle factors
    lifestyle_score = (
        feature_data['PhysActivity'] - 
        (feature_data['AlcoholDays'] / 30 + feature_data['Smoker'])
    )
    feature_data['LifestyleScore'] = lifestyle_score
    
    # Cardio Risk: cardiovascular disease risk factors
    cardio_risk = (
        feature_data['HighBP'] + 
        feature_data['HighChol'] + 
        feature_data['DiagnosedHeartAttack'] + 
        feature_data['CoronaryHeartDisease'] +
        (1 if feature_data['BMI'] > 30 else 0)
    )
    feature_data['CardioRisk'] = cardio_risk
    
    # Cholesterol Management Score
    chol_mgmt = (
        1 + feature_data['CholesterolMeds'] - 
        feature_data['HighChol'] * (1 - feature_data['CholesterolMeds'] * 0.5)
    )
    feature_data['CholesterolManagementScore'] = chol_mgmt
    
    # Mental Health Score: negative mental health indicators
    mental_health = -(
        feature_data['Depression'] + 
        feature_data['CognitiveIssues'] + 
        feature_data['MentHlth'] / 30
    )
    feature_data['MentalHealthScore'] = mental_health
    
    return feature_data
