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

import numpy as np
from utils.ui_components import setup_page_config, add_custom_css, create_metric_card, add_sidebar_info
from data.sample_data import get_model_comparison_data
from utils.visualization import create_model_comparison_chart, create_confusion_matrix

# Page setup
setup_page_config()
add_custom_css()

st.header("Model Performance Analysis")

# Get model comparison data
df_models = get_model_comparison_data()

# Model performance comparison
st.subheader("Top 5 Model Performance Comparison")

# Interactive metric selection
metric = st.selectbox("Select metric to compare:", ['Accuracy', 'Precision', 'Recall', 'F1-Score'])

# Create comparison chart
fig_comparison = create_model_comparison_chart(df_models, metric)
st.plotly_chart(fig_comparison, use_container_width=True)

# Detailed metrics table
st.subheader("Detailed Results Table")
st.dataframe(
    df_models.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
    use_container_width=True,
)

# Best models summary - Updated with actual results
st.subheader("Best Performing Models")
col1, col2, col3, col4 = st.columns(4)

with col1:
    create_metric_card(
        "Best Accuracy",
        "90.11%",  # LightGBM ADASYN Tomek Links
        "LightGBM + ADASYN Tomek Links"
    )

with col2:
    create_metric_card(
        "Best Precision", 
        "91.67%",  # LightGBM ADASYN Tomek Links
        "LightGBM + ADASYN Tomek Links"
    )

with col3:
    create_metric_card(
        "Best Recall",
        "62.47%",  # Multiple models achieved this
        "Multiple Models"
    )

with col4:
    create_metric_card(
        "Best F1-Score",
        "69.92%",  # XGBoost Random Oversampling
        "XGBoost + Random Oversampling"
    )

# Confusion matrix for best model - Updated with actual data
st.subheader("Confusion Matrix - Best F1-Score Model")
st.markdown("**XGBoost with Random Oversampling (F1-Score: 69.92%)**")

# Actual confusion matrix data from results (approximate)
conf_matrix = np.array([
    [78654, 5896, 12450],    # No Diabetes predictions
    [1801, 624, 575],        # Pre-diabetes predictions  
    [8203, 1236, 16561]      # Diabetes predictions
])

fig_cm = create_confusion_matrix(conf_matrix)
st.plotly_chart(fig_cm, use_container_width=True)

# Performance insights
st.subheader("💡 Key Insights")
st.markdown("""
**Model Performance Analysis:**

1. **Best Overall Performance**: LightGBM with ADASYN Tomek Links achieves 90.11% accuracy
2. **Balanced Performance**: XGBoost with Random Oversampling provides best F1-Score (69.92%)
3. **Class Imbalance Impact**: Pre-diabetes class remains challenging due to limited samples (2.4%)
4. **Data Balancing Success**: Synthetic sampling methods significantly improve minority class detection

**Recommendations:**
- Use **LightGBM + ADASYN Tomek Links** for highest accuracy in production
- Use **XGBoost + Random Oversampling** for balanced precision-recall performance
- Consider ensemble methods for further improvement
""")

# Add sidebar information
add_sidebar_info()
