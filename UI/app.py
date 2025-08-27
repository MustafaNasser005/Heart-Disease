import os
from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .main-header {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        color: #333;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #e9ecef;
    }
    
    .prediction-box {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        color: #333;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #28a745;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        color: #155724;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid #c3e6cb;
    }
    
    .info-box {
        background: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    
    .sidebar-section {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stButton > button {
        background: #007bff;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #0056b3;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = (PROJECT_ROOT / "Models").resolve()
DATA_DIR = (PROJECT_ROOT / "Data").resolve()

@st.cache_data
def load_dataset():
    """Load the cleaned heart disease dataset from the specified path."""
    csv_path = DATA_DIR / "data_cleaning_heart_disease_uci.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    return pd.read_csv(csv_path)

def apply_feature_engineering(df):
    """Apply the same feature engineering as in notebooks."""
    df = df.copy()
    
    # Create engineered features (same as notebooks)
    df["age_cholesterol_ratio"] = df["age"] / (df["cholesterol"] + 1)
    df["bp_cholesterol_ratio"] = df["resting_blood_pressure"] / (df["cholesterol"] + 1)
    df["heart_rate_stress"] = df["maximum_heart_rate_achieved"] / (df["age"] + 1)
    df["risk_index"] = df["resting_blood_pressure"] + df["cholesterol"] - df["maximum_heart_rate_achieved"]
    
    # Note: We do NOT drop cholesterol and resting_electrocardiographic_results
    # because the models were trained with these columns present
    
    return df

def encode_categorical_features(df):
    """Apply label encoding to categorical features (same as notebooks)."""
    df = df.copy()
    
    # Categorical columns to encode (same as notebooks)
    categorical_cols = [
        "gender", 
        "dataset", 
        "chest_pain_type", 
        "fasting_blood_sugar", 
        "resting_electrocardiographic_results", 
        "exercise_induced_angina"
    ]
    
    # Apply Label Encoding
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    
    return df

def preprocess_data(df):
    """Complete preprocessing pipeline that matches model training exactly."""
    # The model expects exactly these 10 features:
    # ['age', 'gender', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 
    #  'fasting_blood_sugar', 'resting_electrocardiographic_results', 'maximum_heart_rate_achieved', 
    #  'exercise_induced_angina', 'oldpeak']
    
    # 1. Apply label encoding for categorical variables only
    df = encode_categorical_features(df)
    
    # 2. Select only the features the model expects
    expected_columns = [
        'age', 'gender', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol',
        'fasting_blood_sugar', 'resting_electrocardiographic_results', 'maximum_heart_rate_achieved',
        'exercise_induced_angina', 'oldpeak'
    ]
    
    # Add missing columns with default values
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0.0  # Default numeric value
    
    # Select only the expected columns in the correct order
    df = df[expected_columns]
    
    return df

def load_model(model_path):
    """Load a trained model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def find_model_files():
    """Find available model files."""
    candidates = {
        "Random Forest": MODELS_DIR / "rand_forest_model.pkl",
        "AdaBoost": MODELS_DIR / "adaboost_model.pkl",
    }
    found = {}
    for name, path in candidates.items():
        if path.exists() and path.stat().st_size > 8_000:
            found[name] = path
    return found

def create_user_input_form():
    """Create input form for user data matching the dataset columns."""
    st.subheader("👤 Enter Patient Information")
    
    # Create a more organized form with better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Basic Information**")
        age = st.number_input("Age", min_value=1, max_value=120, value=45, help="Patient's age in years")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Patient's gender")
        dataset = st.selectbox("Dataset Source", ["Cleveland", "Hungary", "Switzerland", "Long Beach"], help="Source of the data")
        
        st.markdown("**💔 Chest Pain Information**")
        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
            help="Type of chest pain experienced"
        )
        
        st.markdown("**🩺 Blood Pressure & Cholesterol**")
        resting_blood_pressure = st.number_input(
            "Resting Blood Pressure (mm Hg)", 
            60, 250, 130, 
            help="Resting blood pressure in mm Hg"
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dl)", 
            80, 700, 230, 
            help="Serum cholesterol in mg/dl"
        )
    
    with col2:
        st.markdown("**🩸 Blood Sugar & ECG**")
        fasting_blood_sugar = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl?", 
            ["False", "True"], 
            help="Whether fasting blood sugar is above 120 mg/dl"
        )
        resting_electrocardiographic_results = st.selectbox(
            "Resting ECG Results",
            ["normal", "lv hypertrophy"],
            help="Results of resting electrocardiogram"
        )
        
        st.markdown("**❤️ Heart Rate & Exercise**")
        maximum_heart_rate_achieved = st.number_input(
            "Maximum Heart Rate Achieved", 
            60, 250, 150, 
            help="Maximum heart rate achieved during exercise"
        )
        exercise_induced_angina = st.selectbox(
            "Exercise Induced Angina", 
            ["False", "True"], 
            help="Whether angina was induced by exercise"
        )
        
        st.markdown("**📈 ST Depression**")
        oldpeak = st.number_input(
            "Oldpeak (ST depression)", 
            -2.0, 7.0, 1.0, 
            step=0.1, 
            help="ST depression induced by exercise relative to rest"
        )
    
    return {
        "age": age,
        "gender": gender,
        "dataset": dataset,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "resting_electrocardiographic_results": resting_electrocardiographic_results,
        "maximum_heart_rate_achieved": maximum_heart_rate_achieved,
        "exercise_induced_angina": exercise_induced_angina,
        "oldpeak": oldpeak
    }

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>❤️ Heart Disease Prediction System</h1>
        <p>Simple and accurate cardiovascular risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    model_files = find_model_files()
    if not model_files:
        st.error("❌ Models not found. Please ensure trained models are in the Models folder.")
        st.stop()
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.subheader("🤖 Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose Model",
        list(model_files.keys()),
        index=0
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Load selected model
    model = load_model(model_files[selected_model])
    
    # Main content
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.info("Enter patient information below to get a heart disease prediction")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load and display dataset info
    try:
        df_dataset = load_dataset()
        
        # Dataset information
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Patients", len(df_dataset))
        with col2:
            st.metric("📈 Features", df_dataset.shape[1])
        with col3:
            st.metric("❌ Missing Values", int(df_dataset.isnull().sum().sum()))
        with col4:
            if 'num' in df_dataset.columns:
                heart_disease_count = df_dataset['num'].sum()
                st.metric("❤️ Heart Disease Cases", heart_disease_count)
            else:
                st.metric("🎯 Target Column", "N/A")
        
        # Dataset preview
        with st.expander("📋 Dataset Preview (First 10 rows)", expanded=False):
            st.dataframe(df_dataset.head(10), use_container_width=True)
            
        with st.expander("📊 Dataset Information", expanded=False):
            st.write("**Column Types:**")
            st.write(df_dataset.dtypes.astype(str))
            
            st.write("**Missing Values:**")
            missing_data = pd.DataFrame({
                'Column': df_dataset.columns,
                'Missing Count': df_dataset.isnull().sum(),
                'Missing Percentage': (df_dataset.isnull().sum() / len(df_dataset) * 100).round(2)
            })
            st.dataframe(missing_data, use_container_width=True)
            
    except Exception as e:
        st.warning(f"⚠️ Could not load dataset: {str(e)}")
    
    # User input form
    user_data = create_user_input_form()
    
    # Prediction button
    if st.button("🔮 Get Prediction", type="primary", use_container_width=True):
        try:
            # Convert user data to DataFrame
            df_user = pd.DataFrame([user_data])
            
            # Apply preprocessing pipeline (exactly as in training)
            df_processed = preprocess_data(df_user)
            
            # Make prediction
            prediction = model.predict(df_processed)[0]
            probability = None
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(df_processed)[0][1]
            
            # Display result
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("## ❌ **HEART DISEASE DETECTED**")
                st.markdown("### High Risk")
            else:
                st.markdown("## ✅ **NO HEART DISEASE**")
                st.markdown("### Low Risk")
            
            if probability is not None:
                st.markdown(f"**Risk Probability: {probability:.1%}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show processed features
            with st.expander("🔧 Processed Features Used"):
                st.dataframe(df_processed, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()