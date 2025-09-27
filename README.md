# 🏥 Diabetes Prediction System

## 📋 Project Overview

An advanced machine learning system for predicting diabetes risk using BRFSS (Behavioral Risk Factor Surveillance System) data from 2017-2023. The project addresses the critical healthcare challenge of early diabetes detection through comprehensive data analysis, feature engineering, and multiple machine learning approaches.

### 🎯 Key Objectives

- **Multi-class Classification**: Predict 3 diabetes states (No Diabetes, Pre-diabetes, Diabetes)
- **Class Imbalance Solution**: Address severe imbalance (82% No Diabetes, 2.4% Pre-diabetes, 15.6% Diabetes)
- **Production-Ready**: Deploy via interactive Streamlit web application
- **Real-world Validation**: Test on latest 2023 BRFSS data

## 🚀 Key Features

### 📊 Data Processing Pipeline

- **Comprehensive Cleaning**: Process 450K+ samples from BRFSS 2017-2021
- **Feature Engineering**: Create 8 composite health scores (Health, Risk, Lifestyle, Cardio, etc.)
- **Smart Selection**: Reduce 43 features to 34 most predictive ones using correlation analysis

### ⚖️ Advanced Data Balancing

- **9 Balancing Techniques**: Random Oversampling, SMOTE, ADASYN, Tomek Links, ENN, and hybrid methods
- **Synthetic Data Generation**: Create balanced datasets for better minority class detection
- **Comprehensive Evaluation**: Test all combinations of 6 models × 9 balancing methods (54 experiments)

### 🤖 Machine Learning Models

- **6 Algorithms**: Logistic Regression, Random Forest, XGBoost, LightGBM, Naive Bayes, SGD Classifier
- **Cross-Validation**: 5-fold CV for robust performance evaluation
- **Multiple Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 📱 Interactive Web Application

- **Real-time Prediction**: Input health parameters and get instant diabetes risk assessment
- **Model Analysis**: Compare performance across different models and balancing techniques
- **Data Visualization**: Comprehensive dashboards with interactive charts
- **Technical Details**: Complete methodology and results documentation

## 🏆 Performance Results

### Top Performing Models:

| Metric | Score | Model | Method |
|--------|--------|--------|---------|
| **Best Accuracy** | 90.11% | LightGBM | ADASYN + Tomek Links |
| **Best F1-Score** | 69.92% | XGBoost | Random Oversampling |
| **Best Precision** | 91.67% | LightGBM | ADASYN + Tomek Links |
| **Best Recall** | 62.47% | Multiple | Various |

### Real-world Performance (2023 Data):

- **XGBoost + Random Oversampling**: 68.93% accuracy, 44.00% F1-score
- **Production Model**: Maintains good performance on unseen data

## 🛠️ Tech Stack

### Data Science & ML

- **Python 3.12+**
- **pandas, numpy**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and preprocessing
- **imbalanced-learn**: Advanced resampling techniques
- **XGBoost, LightGBM**: Gradient boosting frameworks

### Visualization & Web App

- **Streamlit**: Interactive web application framework
- **Plotly**: Dynamic, interactive visualizations
- **Seaborn, Matplotlib**: Statistical plotting

### Development & Deployment

- **Docker**: Containerized deployment
- **Git**: Version control and collaboration

## 📁 Project Structure

```bash
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── app/                          # Streamlit web application
│   ├── Home.py                  # Main dashboard
│   ├── data/
│   │   └── sample_data.py       # Sample data utilities
│   ├── pages/                   # Application pages
│   │   ├── 1_Prediction.py      # Prediction interface
│   │   ├── 2_Model_Analysis.py  # Performance analysis
│   │   ├── 3_Project_Details.py # Technical documentation
│   │   └── 4_Data_Dashboard.py  # Data exploration
│   └── utils/                   # Application utilities
│       ├── feature_engineering.py # Feature processing
│       ├── model_utils.py       # Model loading & inference
│       ├── ui_components.py     # UI components
│       └── visualization.py     # Chart creation
├── artifacts/                   # Model artifacts
│   ├── models/                  # Trained models
│   │   ├── best_accuracy_model.pkl
│   │   ├── best_f1_model.pkl
│   │   ├── best_precision_model.pkl
│   │   └── best_recall_model.pkl
│   ├── scalers/                 # Data scalers
│   │   ├── best_f1_model_scaler.pkl
│   │   ├── best_model_scaler.pkl
│   │   ├── best_precision_model_scaler.pkl
│   │   └── best_recall_model_scaler.pkl
│   └── training_2017_2019_2021/ # Training artifacts
│       ├── encoders.pkl
│       └── selected_features.pkl
├── data/                        # Dataset storage
│   ├── processed/               # Processed datasets
│   │   ├── adasyn_data.pkl
│   │   ├── adasyn_tomek_data.pkl
│   │   ├── brfss_dataset.csv
│   │   ├── edited_neighbors_data.pkl
│   │   ├── random_over_sampling_data.pkl
│   │   ├── random_under_sampling_data.pkl
│   │   ├── smote_data.pkl
│   │   ├── smote_enn_data.pkl
│   │   ├── smote_tomek_links_data.pkl
│   │   ├── testing_data.pkl
│   │   └── tomek_links_data.pkl
│   └── raw/                     # Raw datasets
│       ├── brfss_2017.XPT
│       ├── brfss_2019.XPT
│       └── brfss_2021.XPT
├── logs/                        # Training logs
│   ├── 0_data_pipeline.log
│   ├── 2_model_training_with_original_data.log
│   ├── 3_feature_engineering.log
│   ├── 4_data_balancing.log
│   ├── 5_model_training_with_enhanced_data.log
│   └── 6_diabetes_2023_prediction.log
├── notebooks/                   # Research notebooks
│   ├── 0_data_pipeline.ipynb    # Data processing
│   ├── 1_visualization.ipynb    # EDA & visualization
│   ├── 2_models_with_original_data.ipynb # Baseline models
│   ├── 3_feature_engineering.ipynb # Feature engineering
│   ├── 4_data_balancing.ipynb   # Data balancing
│   ├── 5_models_with_balancing_data.ipynb # Comprehensive training
│   ├── 6_diabetes_prediction_2023.ipynb # Real-world validation
└── src/                         # Source code modules
    ├── balancing.py             # Data balancing methods
    ├── config.py                # Configuration settings
    ├── data.py                  # Data processing utilities
    ├── evaluate.py              # Model evaluation functions
    ├── features.py              # Feature engineering
    ├── models.py                # ML model implementations
    ├── pipelines.py             # ML pipelines
    ├── utils.py                 # General utilities
    └── visualization.py         # Plotting functions
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd diabetes-prediction-system
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit application**

```bash
streamlit run Home.py
```

### Docker Deployment

1. **Build the Docker image**

```bash
docker build -t diabetes-prediction .
```

2. **Run the container**

```bash
docker run -p 8080:8080 diabetes-prediction
```

3. **Access the application**

Open your browser and navigate to `http://localhost:8080`

## 📊 Usage Guide

### 1. Making Predictions

- Navigate to the "🔮 Prediction" page
- Fill in health parameters (BMI, blood pressure, lifestyle factors, etc.)
- Get instant diabetes risk assessment with probability scores

### 2. Analyzing Models

- Visit "📈 Model Analysis" for comprehensive performance comparison
- Interactive charts showing accuracy, precision, recall, and F1-scores
- Confusion matrices for detailed classification analysis

### 3. Exploring Data

- Use "📊 Data Dashboard" to explore the BRFSS dataset
- Interactive visualizations of health indicators and risk factors
- Pagination support for large dataset browsing

### 4. Technical Details

- Check "📋 Project Details" for complete methodology
- Performance metrics across all models and balancing techniques
- Recommendations for production deployment

## 🎯 Key Insights

### Model Performance

1. **Gradient Boosting Excellence**: XGBoost and LightGBM consistently outperform other algorithms
2. **Balancing Impact**: Data balancing techniques crucial for minority class (pre-diabetes) detection
3. **Production Readiness**: Models maintain good performance on real-world 2023 data

### Healthcare Applications

1. **Early Detection**: System can identify pre-diabetes cases often missed by traditional methods
2. **Risk Stratification**: Provides probability scores for informed clinical decision-making
3. **Population Health**: Suitable for large-scale diabetes screening programs

### Technical Achievements

1. **Comprehensive Evaluation**: 54 model-balancing combinations tested systematically
2. **Feature Innovation**: Created interpretable composite health scores
3. **Production Pipeline**: End-to-end system from data processing to web deployment

## 🔬 Research Methodology

### Data Sources

- **BRFSS 2017-2021**: Training data (450K+ samples)
- **BRFSS 2023**: Validation data (250K+ samples)
- **CDC Official**: All data sourced from official CDC BRFSS surveys

### Feature Engineering Process

1. **Composite Scores**: Health, Risk, Lifestyle, Cardiovascular, Healthcare Access, Socioeconomic, Mental Health
2. **Categorical Encoding**: BMI categories, Age groups with proper label encoding
3. **Correlation-Based Selection**: Retain features with >0.05 correlation to target

### Balancing Strategy

1. **Oversampling**: Random, SMOTE, ADASYN for minority class augmentation
2. **Undersampling**: Random, Tomek Links, ENN for majority class reduction
3. **Hybrid Approaches**: Combine oversampling + cleaning for optimal results

## 🔮 Future Enhancements

### Model Improvements

- **Ensemble Methods**: Combine top-performing models for better accuracy
- **Deep Learning**: Explore neural networks for complex pattern recognition
- **Online Learning**: Implement continuous model updates with new data

### Application Features

- **Mobile App**: Native mobile application for wider accessibility
- **API Integration**: RESTful API for healthcare system integration
- **Multi-language**: Support for multiple languages and locales

### Data Expansion

- **Additional Years**: Incorporate more historical BRFSS data
- **External Sources**: Integrate with other health datasets
- **Real-time Monitoring**: Connect with wearable devices and health apps

## 🙏 Acknowledgments

- **CDC BRFSS Program**: For providing comprehensive public health surveillance data
- **Open Source Community**: For excellent machine learning libraries and tools
- **Healthcare Professionals**: For domain expertise and validation of medical relevance

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please reach out through the project's issue tracker or contact the development team.

***
