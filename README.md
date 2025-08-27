# Heart Disease Prediction System

A simple and accurate machine learning system for cardiovascular risk assessment using the UCI Heart Disease dataset.

## 🎯 Overview

This project provides a clean, user-friendly interface for heart disease prediction using trained machine learning models. The system follows a complete data science pipeline from preprocessing to model deployment.

**Dataset**: Uses `Data/data_cleaning_heart_disease_uci.csv` - the cleaned UCI Heart Disease dataset with 11 features.

## 📁 Project Structure

```
Heart Disease/
├── Data/
│   ├── data_cleaning_heart_disease_uci.csv    # Original dataset
│   ├── heart_disease_uci_encoding.csv         # Processed dataset
│   └── archive.zip
├── Models/
│   ├── rand_forest_model.pkl                  # Trained Random Forest model
│   └── adaboost_model.pkl                     # Trained AdaBoost model
├── notebooks/
│   ├── 01_data_preprocessing.ipynb           # Data cleaning and preprocessing
│   ├── _02_feature_selection.ipynb           # Feature engineering and selection
│   ├── 03_supervised_learning.ipynb          # Model training and evaluation
│   ├── _04_unsupervised_learning.ipynb       # Unsupervised analysis
│   └── 05_hyperparameter_tuning.ipynb        # Model optimization
├── UI/
│   └── app.py                                # Streamlit web application
├── Deployment/
│   └── ngrok_setup.txt                       # Deployment instructions
├── results/
│   └── evaluation_metrics.txt                # Model performance metrics
├── requirements.txt                          # Python dependencies
├── .gitignore                               # Git ignore rules
└── README.md
```

## 🔧 Features

### Data Pipeline
- **Feature Engineering**: Creates 4 new features from existing ones
  - `age_cholesterol_ratio`: Age / (Cholesterol + 1)
  - `bp_cholesterol_ratio`: Blood Pressure / (Cholesterol + 1)
  - `heart_rate_stress`: Max Heart Rate / (Age + 1)
  - `risk_index`: Blood Pressure + Cholesterol - Max Heart Rate
- **Label Encoding**: Converts categorical variables to numeric
- **Feature Selection**: Removes low-correlation features (`cholesterol`, `resting_electrocardiographic_results`)

### Machine Learning Models
- **Random Forest Classifier**: 81.52% accuracy
- **AdaBoost Classifier**: 80.98% accuracy
- **StandardScaler**: Feature normalization

### Web Application
- **Simple Interface**: Clean, focused design for easy use
- **Single Prediction**: Enter patient data for individual predictions
- **Model Selection**: Choose between Random Forest and AdaBoost
- **Real-time Results**: Instant predictions with probability scores

## 🚀 Quick Start

### Prerequisites
   ```bash
# Clone the repository
git clone <repository-url>
cd "Heart Disease"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
   pip install -r requirements.txt
   ```

### Run the Application
   ```bash
cd UI
   streamlit run app.py
   ```

The application will open in your browser at `http://localhost:8501`

## 📊 Usage

### Single Prediction
1. Select a model (Random Forest or AdaBoost)
2. Enter patient information:
   - Age, Gender, Chest Pain Type
   - Blood Pressure, Cholesterol
   - Heart Rate, ECG Results
   - Exercise Angina, ST Depression
3. Click "Get Prediction" for instant results

## 🔬 Technical Details

### Feature Engineering Pipeline
```python
# New features created
df["age_cholesterol_ratio"] = df["age"] / (df["cholesterol"] + 1)
df["bp_cholesterol_ratio"] = df["resting_blood_pressure"] / (df["cholesterol"] + 1)
df["heart_rate_stress"] = df["maximum_heart_rate_achieved"] / (df["age"] + 1)
df["risk_index"] = df["resting_blood_pressure"] + df["cholesterol"] - df["maximum_heart_rate_achieved"]

# Columns dropped
df.drop(columns=['cholesterol', 'resting_electrocardiographic_results'])
```

### Model Performance
- **Random Forest**: 81.52% accuracy
- **AdaBoost**: 80.98% accuracy
- Both models use hyperparameter tuning for optimal performance

### Data Preprocessing
1. Load original dataset
2. Apply feature engineering
3. Encode categorical variables
4. Remove selected features
5. Scale features (if required by model)
6. Make predictions

## 📈 Model Training

The models were trained using the following process:
1. **Data Preprocessing**: Feature engineering and encoding
2. **Feature Selection**: Correlation-based selection
3. **Model Training**: Multiple algorithms tested
4. **Hyperparameter Tuning**: Grid search optimization
5. **Evaluation**: Cross-validation and metrics

## 🎯 Key Features

- **Simple Interface**: Easy-to-use web application
- **Accurate Predictions**: High-performance ML models
- **Real-time Processing**: Instant predictions
- **Model Flexibility**: Choose between different algorithms
- **Feature Transparency**: View processed features used for prediction

## 🔍 Dataset Information

The UCI Heart Disease dataset contains:
- **918 patients** with cardiovascular data
- **11 features** including age, gender, medical measurements
- **Binary target**: Heart disease presence (0/1)

### Dataset Features
- `age`: Age in years
- `gender`: Gender (Male/Female)
- `chest_pain_type`: Type of chest pain
- `resting_blood_pressure`: Blood pressure in mm Hg
- `cholesterol`: Serum cholesterol in mg/dl
- `fasting_blood_sugar`: Fasting blood sugar > 120 mg/dl
- `resting_electrocardiographic_results`: ECG results
- `maximum_heart_rate_achieved`: Maximum heart rate
- `exercise_induced_angina`: Exercise induced angina
- `oldpeak`: ST depression induced by exercise

## 🚀 Deployment

### Local Deployment
```bash
streamlit run UI/app.py
```

### Cloud Deployment
See `Deployment/ngrok_setup.txt` for instructions on deploying with ngrok.

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Feel free to contribute by:
- Improving the feature engineering pipeline
- Adding new machine learning models
- Enhancing the web interface
- Optimizing model performance

## 📞 Support

If you encounter any issues:
1. Check the requirements are installed correctly
2. Ensure the dataset files are in the correct location
3. Verify the model files are present in the Models folder

---

**Note**: This system is designed for educational purposes and should not be used as the sole basis for medical decisions. Always consult healthcare professionals for medical advice.
