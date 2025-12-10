import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import os

# Page configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for card styling
st.markdown("""
    <style>
    /* Dark theme background */
    .stApp {
        background: linear-gradient(135deg, #0a1929 0%, #1a2332 100%);
    }
    /* Remove default Streamlit spacing */
    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem !important;
        margin-top: 0 !important;
    }
    .main-header {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 0.5rem 2rem 0 2rem !important;
        margin-bottom: 0 !important;
        margin-top: 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        letter-spacing: 2px;
        animation: fadeIn 1s ease-in;
        line-height: 1.05;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .subtitle {
        font-size: 1.5rem;
        color: #b0bec5;
        text-align: center;
        margin-bottom: 0 !important;
        margin-top: 0.25rem !important;
        padding-bottom: 0 !important;
        padding-top: 0 !important;
        font-style: italic;
        line-height: 1.0;
    }
    /* Title container to group title and subtitle */
    .title-container {
        margin-bottom: -0.6rem !important;
        padding-bottom: 0 !important;
    }
    /* Homepage container with dots background */
    .homepage-container {
        background: radial-gradient(circle at 20% 50%, rgba(76, 175, 80, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(76, 175, 80, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 40% 20%, rgba(76, 175, 80, 0.1) 0%, transparent 50%);
        padding: 0 2rem 2rem 2rem !important;
        min-height: 70vh;
        margin-top: -2rem !important;
    }
    /* Remove spacing from Streamlit markdown elements */
    div[data-testid="stMarkdownContainer"] {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="stMarkdownContainer"]:has(.subtitle) {
        margin-top: -1rem !important;
        margin-bottom: 0 !important;
    }
    /* Disease card styling */
    .disease-card-home {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    .disease-card-home:hover {
        transform: translateY(-10px);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .card-icon-container {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1.5rem;
        border: 2px solid rgba(102, 126, 234, 0.3);
    }
    .card-icon-large {
        font-size: 4rem;
    }
    .card-title-home {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4fc3f7;
        margin-bottom: 0.5rem;
    }
    .card-description {
        font-size: 1rem;
        color: #b0bec5;
        margin-bottom: 1.5rem;
        flex-grow: 1;
    }
    /* Button styling will be handled by Streamlit button component */
    .stButton > button {
        background: transparent !important;
        border: 2px solid #4fc3f7 !important;
        color: #ffffff !important;
        padding: 0.75rem 2rem !important;
        border-radius: 25px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: rgba(79, 195, 247, 0.2) !important;
        border-color: #4fc3f7 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(79, 195, 247, 0.3) !important;
    }
    .back-button {
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.5);
        color: #4fc3f7;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-bottom: 2rem;
    }
    .back-button:hover {
        background: rgba(102, 126, 234, 0.4);
    }
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        margin-top: 3rem;
        border-radius: 10px 10px 0 0;
    }
    .footer-content {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 2rem;
        flex-wrap: wrap;
    }
    .footer-text {
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .social-links {
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }
    .social-link {
        color: white;
        text-decoration: none;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        background: rgba(255, 255, 255, 0.2);
    }
    .social-link:hover {
        transform: translateY(-3px);
        background: rgba(255, 255, 255, 0.3);
        color: #fee140;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .disease-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .disease-card:hover {
        transform: translateY(-5px);
    }
    .card-icon {
        font-size: 4rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .card-content {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        color: #e0e0e0;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4fc3f7;
    }
    .prediction-box {
        background-color: rgba(79, 195, 247, 0.1);
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4fc3f7;
    }
    .code-block {
        background-color: rgba(0, 0, 0, 0.3);
        border-radius: 5px;
        padding: 1rem;
        overflow-x: auto;
    }
    /* Streamlit component styling for dark theme */
    .stDataFrame, .stDataFrame table {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #e0e0e0 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #b0bec5;
    }
    .stTabs [aria-selected="true"] {
        color: #4fc3f7 !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    try:
        # Load Diabetes models
        with open('diabetes_models.pkl', 'rb') as f:
            models['diabetes'] = pickle.load(f)
        
        # Load Heart Disease models
        with open('heart_disease_models.pkl', 'rb') as f:
            models['heart'] = pickle.load(f)
        
        # Load Kidney Disease models
        with open('kidney_disease_models.pkl', 'rb') as f:
            models['kidney'] = pickle.load(f)
        
        # Load Thyroid Disease models
        with open('thyroid_disease_models.pkl', 'rb') as f:
            models['thyroid'] = pickle.load(f)
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please run the model training scripts first.")
        st.stop()
    return models

@st.cache_data
def load_dataset(csv_file):
    """Load dataset from CSV file"""
    try:
        return pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def read_model_code(file_path):
    """Read model code from file"""
    try:
        # Skip reading app.py if it's the current file to avoid recursion
        if file_path == 'app.py':
            # Read the file normally - this should work fine
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def display_evaluation_metrics(results, disease_name):
    """Display evaluation metrics for a disease"""
    st.subheader(f"📊 {disease_name} Model Evaluation Metrics")
    
    # Create DataFrame for metrics
    metrics_df = pd.DataFrame({
        model: {
            "Accuracy": f"{results[model]['Accuracy']:.4f}",
            "Precision": f"{results[model]['Precision']:.4f}",
            "Recall": f"{results[model]['Recall']:.4f}",
            "F1 Score": f"{results[model]['F1 Score']:.4f}"
        }
        for model in results
    }).T
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Display confusion matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(2)
    for idx, (model_name, result) in enumerate(results.items()):
        with cols[idx % 2]:
            st.write(f"**{model_name}**")
            st.write(result['Confusion Matrix'])

def create_disease_card(icon, title, color_gradient):
    """Create a styled disease card"""
    st.markdown(f"""
        <div class="disease-card" style="background: {color_gradient};">
            <div class="card-icon">{icon}</div>
            <div class="card-title">{title}</div>
        </div>
    """, unsafe_allow_html=True)

def diabetes_prediction_tab(models_data):
    """Diabetes prediction tab"""
    # Card header
    create_disease_card("🩺", "Diabetes Prediction", "linear-gradient(135deg, #667eea 0%, #764ba2 100%)")
    
    st.markdown("---")
    
    # Create sub-tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Evaluation Metrics", "🔮 Prediction", "💻 Model Code", "📁 Dataset"])
    
    with tab1:
        display_evaluation_metrics(models_data['results'], "Diabetes")
    
    with tab2:
        st.subheader("🔮 Make a Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3, 
                                         help="Number of times pregnant (Range: 0-17)")
            glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120, 
                                     help="Plasma glucose concentration (Range: 0-199 mg/dL)")
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=72, 
                                             help="Diastolic blood pressure (Range: 0-122 mm Hg)")
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=23, 
                                             help="Triceps skin fold thickness (Range: 0-99 mm)")
        
        with col2:
            insulin = st.number_input("Insulin", min_value=0, max_value=850, value=80, 
                                     help="2-Hour serum insulin (Range: 0-846 μU/ml)")
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0, step=0.1, 
                                 help="Body mass index (Range: 0-67.1)")
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, 
                                 help="Diabetes pedigree function (Range: 0.078-2.42)")
            age = st.number_input("Age", min_value=21, max_value=81, value=33, 
                                 help="Age in years (Range: 21-81)")
        
        if st.button("Predict Diabetes", type="primary"):
            # Prepare input data
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                   insulin, bmi, dpf, age]])
            
            # Scale input
            scaler = models_data['scaler']
            input_scaled = scaler.transform(input_data)
            
            # Get predictions from all models
            predictions = {}
            for model_name, model in models_data['models'].items():
                if model_name in ["KNN", "Logistic Regression"]:
                    pred = model.predict(input_scaled)[0]
                    prob = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
                else:
                    pred = model.predict(input_data)[0]
                    prob = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None
                predictions[model_name] = {
                    'prediction': pred,
                    'probability': prob
                }
            
            # Display results
            st.subheader("Prediction Results")
            result_cols = st.columns(4)
            
            for idx, (model_name, pred_data) in enumerate(predictions.items()):
                with result_cols[idx]:
                    result = "Positive" if pred_data['prediction'] == 1 else "Negative"
                    color = "🔴" if pred_data['prediction'] == 1 else "🟢"
                    st.metric(f"{model_name}", f"{color} {result}")
                    if pred_data['probability'] is not None:
                        prob = pred_data['probability'][1] if pred_data['prediction'] == 1 else pred_data['probability'][0]
                        st.write(f"Confidence: {prob*100:.2f}%")
    
    with tab3:
        st.subheader("💻 Code")
        code_tab1, code_tab2 = st.tabs(["Model Training Code", "Application Code"])
        
        with code_tab1:
            st.write("**Diabetes Model Training Code (diabetes_model.py)**")
            code = read_model_code('diabetes_model.py')
            st.code(code, language='python')
        
        with code_tab2:
            st.write("**Streamlit Application Code (app.py)**")
            app_code = read_model_code('app.py')
            st.code(app_code, language='python')
    
    with tab4:
        st.subheader("📁 Diabetes Dataset")
        df = load_dataset('diabetes.csv')
        if df is not None:
            st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(100), use_container_width=True)
            st.write("**Dataset Statistics:**")
            st.dataframe(df.describe(), use_container_width=True)

def heart_disease_prediction_tab(models_data):
    """Heart disease prediction tab"""
    # Card header
    create_disease_card("❤️", "Heart Disease Prediction", "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)")
    
    st.markdown("---")
    
    # Create sub-tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Evaluation Metrics", "🔮 Prediction", "💻 Model Code", "📁 Dataset"])
    
    with tab1:
        display_evaluation_metrics(models_data['results'], "Heart Disease")
    
    with tab2:
        st.subheader("🔮 Make a Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=50, key="heart_age",
                                 help="Age in years (Range: 18-80)")
            gender = st.selectbox("Gender", ["Male", "Female"], key="heart_gender")
            blood_pressure = st.number_input("Blood Pressure", min_value=120, max_value=180, value=150, key="heart_bp",
                                            help="Blood pressure (Range: 120-180 mm Hg)")
            cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200, key="heart_chol",
                                         help="Cholesterol level (Range: 100-400 mg/dL)")
            exercise = st.selectbox("Exercise Habits", ["Low", "Medium", "High"], key="heart_exercise")
            smoking = st.selectbox("Smoking", ["Yes", "No"], key="heart_smoking")
            family_history = st.selectbox("Family Heart Disease", ["Yes", "No"], key="heart_family")
        
        with col2:
            diabetes = st.selectbox("Diabetes", ["Yes", "No"], key="heart_diabetes")
            bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1, key="heart_bmi",
                                 help="Body mass index (Range: 15-50)")
            high_bp = st.selectbox("High Blood Pressure", ["Yes", "No"], key="heart_highbp")
            low_hdl = st.selectbox("Low HDL Cholesterol", ["Yes", "No"], key="heart_lowhdl")
            high_ldl = st.selectbox("High LDL Cholesterol", ["Yes", "No"], key="heart_highldl")
            alcohol = st.selectbox("Alcohol Consumption", ["Low", "Medium", "High"], key="heart_alcohol")
            stress = st.selectbox("Stress Level", ["Low", "Medium", "High"], key="heart_stress")
        
        with col3:
            sleep_hours = st.number_input("Sleep Hours", min_value=4.0, max_value=10.0, value=7.0, step=0.1, key="heart_sleep",
                                         help="Average sleep hours per night (Range: 4-10)")
            sugar_consumption = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"], key="heart_sugar")
            triglyceride = st.number_input("Triglyceride Level", min_value=50, max_value=500, value=150, key="heart_trig",
                                          help="Triglyceride level (Range: 50-500 mg/dL)")
            fasting_bs = st.number_input("Fasting Blood Sugar", min_value=70, max_value=200, value=100, key="heart_fbs",
                                         help="Fasting blood sugar (Range: 70-200 mg/dL)")
            crp_level = st.number_input("CRP Level", min_value=0.0, max_value=15.0, value=7.0, step=0.1, key="heart_crp",
                                       help="C-reactive protein level (Range: 0-15 mg/L)")
            homocysteine = st.number_input("Homocysteine Level", min_value=5.0, max_value=20.0, value=12.0, step=0.1, key="heart_homo",
                                          help="Homocysteine level (Range: 5-20 μmol/L)")
        
        if st.button("Predict Heart Disease", type="primary", key="heart_predict"):
            # Encode categorical variables
            label_encoders = models_data['label_encoders']
            
            # Prepare input data
            input_dict = {
                'Age': age,
                'Gender': label_encoders['Gender'].transform([gender])[0],
                'Blood Pressure': blood_pressure,
                'Cholesterol Level': cholesterol,
                'Exercise Habits': label_encoders['Exercise Habits'].transform([exercise])[0],
                'Smoking': label_encoders['Smoking'].transform([smoking])[0],
                'Family Heart Disease': label_encoders['Family Heart Disease'].transform([family_history])[0],
                'Diabetes': label_encoders['Diabetes'].transform([diabetes])[0],
                'BMI': bmi,
                'High Blood Pressure': label_encoders['High Blood Pressure'].transform([high_bp])[0],
                'Low HDL Cholesterol': label_encoders['Low HDL Cholesterol'].transform([low_hdl])[0],
                'High LDL Cholesterol': label_encoders['High LDL Cholesterol'].transform([high_ldl])[0],
                'Alcohol Consumption': label_encoders['Alcohol Consumption'].transform([alcohol])[0],
                'Stress Level': label_encoders['Stress Level'].transform([stress])[0],
                'Sleep Hours': sleep_hours,
                'Sugar Consumption': label_encoders['Sugar Consumption'].transform([sugar_consumption])[0],
                'Triglyceride Level': triglyceride,
                'Fasting Blood Sugar': fasting_bs,
                'CRP Level': crp_level,
                'Homocysteine Level': homocysteine
            }
            
            # Create input array in correct order
            feature_names = models_data['feature_names']
            input_data = np.array([[input_dict[feature] for feature in feature_names]])
            
            # Scale input
            scaler = models_data['scaler']
            input_scaled = scaler.transform(input_data)
            
            # Get predictions
            predictions = {}
            target_encoder = models_data['target_encoder']
            
            for model_name, model in models_data['models'].items():
                if model_name in ["KNN", "Logistic Regression"]:
                    pred = model.predict(input_scaled)[0]
                    prob = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
                else:
                    pred = model.predict(input_data)[0]
                    prob = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None
                predictions[model_name] = {
                    'prediction': pred,
                    'probability': prob
                }
            
            # Display results
            st.subheader("Prediction Results")
            result_cols = st.columns(4)
            
            for idx, (model_name, pred_data) in enumerate(predictions.items()):
                with result_cols[idx]:
                    pred_label = target_encoder.inverse_transform([pred_data['prediction']])[0]
                    result = "Yes" if pred_label == "Yes" else "No"
                    color = "🔴" if result == "Yes" else "🟢"
                    st.metric(f"{model_name}", f"{color} {result}")
                    if pred_data['probability'] is not None:
                        prob_idx = 1 if pred_data['prediction'] == 1 else 0
                        prob = pred_data['probability'][prob_idx]
                        st.write(f"Confidence: {prob*100:.2f}%")
    
    with tab3:
        st.subheader("💻 Code")
        code_tab1, code_tab2 = st.tabs(["Model Training Code", "Application Code"])
        
        with code_tab1:
            st.write("**Heart Disease Model Training Code (heart_disease_model.py)**")
            code = read_model_code('heart_disease_model.py')
            st.code(code, language='python')
        
        with code_tab2:
            st.write("**Streamlit Application Code (app.py)**")
            app_code = read_model_code('app.py')
            st.code(app_code, language='python')
    
    with tab4:
        st.subheader("📁 Heart Disease Dataset")
        df = load_dataset('heart_disease.csv')
        if df is not None:
            st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(100), use_container_width=True)
            st.write("**Dataset Statistics:**")
            st.dataframe(df.describe(), use_container_width=True)

def kidney_disease_prediction_tab(models_data):
    """Kidney disease prediction tab"""
    # Card header
    create_disease_card("🫘", "Kidney Disease Prediction", "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)")
    
    st.markdown("---")
    
    # Create sub-tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Evaluation Metrics", "🔮 Prediction", "💻 Model Code", "📁 Dataset"])
    
    with tab1:
        display_evaluation_metrics(models_data['results'], "Kidney Disease")
    
    with tab2:
        st.subheader("🔮 Make a Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            bp = st.number_input("Blood Pressure (Bp)", min_value=50, max_value=180, value=80,
                                help="Blood pressure (Range: 50-180 mm Hg)")
            sg = st.number_input("Specific Gravity (Sg)", min_value=1.005, max_value=1.025, value=1.020, step=0.001,
                                help="Specific gravity (Range: 1.005-1.025)")
            al = st.number_input("Albumin (Al)", min_value=0, max_value=5, value=1,
                                help="Albumin level (Range: 0-5)")
            su = st.number_input("Sugar (Su)", min_value=0, max_value=5, value=0,
                                help="Sugar level (Range: 0-5)")
            rbc = st.number_input("Red Blood Cells (Rbc)", min_value=0.0, max_value=2.0, value=1.0, step=0.1,
                                 help="Red blood cells (Range: 0-2)")
            bu = st.number_input("Blood Urea (Bu)", min_value=10, max_value=200, value=50,
                                help="Blood urea (Range: 10-200 mg/dL)")
            sc = st.number_input("Serum Creatinine (Sc)", min_value=0.5, max_value=10.0, value=1.2, step=0.1,
                                help="Serum creatinine (Range: 0.5-10 mg/dL)")
        
        with col2:
            sod = st.number_input("Sodium (Sod)", min_value=100, max_value=150, value=140,
                                 help="Sodium level (Range: 100-150 mEq/L)")
            pot = st.number_input("Potassium (Pot)", min_value=2.0, max_value=6.0, value=4.5, step=0.1,
                                 help="Potassium level (Range: 2.0-6.0 mEq/L)")
            hemo = st.number_input("Hemoglobin (Hemo)", min_value=5.0, max_value=20.0, value=15.0, step=0.1,
                                  help="Hemoglobin (Range: 5.0-20.0 g/dL)")
            wbcc = st.number_input("White Blood Cell Count (Wbcc)", min_value=2000, max_value=20000, value=8000,
                                 help="White blood cell count (Range: 2000-20000)")
            rbcc = st.number_input("Red Blood Cell Count (Rbcc)", min_value=2.0, max_value=8.0, value=5.0, step=0.1,
                                  help="Red blood cell count (Range: 2.0-8.0 million cells/μL)")
            htn = st.selectbox("Hypertension (Htn)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No",
                              help="Hypertension (0=No, 1=Yes)")
        
        if st.button("Predict Kidney Disease", type="primary", key="kidney_predict"):
            # Prepare input data
            input_data = np.array([[bp, sg, al, su, rbc, bu, sc, sod, pot, hemo, wbcc, rbcc, htn]])
            
            # Scale input
            scaler = models_data['scaler']
            input_scaled = scaler.transform(input_data)
            
            # Get predictions
            predictions = {}
            for model_name, model in models_data['models'].items():
                if model_name in ["KNN", "Logistic Regression"]:
                    pred = model.predict(input_scaled)[0]
                    prob = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
                else:
                    pred = model.predict(input_data)[0]
                    prob = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None
                predictions[model_name] = {
                    'prediction': pred,
                    'probability': prob
                }
            
            # Display results
            st.subheader("Prediction Results")
            result_cols = st.columns(4)
            
            for idx, (model_name, pred_data) in enumerate(predictions.items()):
                with result_cols[idx]:
                    result = "Positive" if pred_data['prediction'] == 1 else "Negative"
                    color = "🔴" if pred_data['prediction'] == 1 else "🟢"
                    st.metric(f"{model_name}", f"{color} {result}")
                    if pred_data['probability'] is not None:
                        prob = pred_data['probability'][1] if pred_data['prediction'] == 1 else pred_data['probability'][0]
                        st.write(f"Confidence: {prob*100:.2f}%")
    
    with tab3:
        st.subheader("💻 Code")
        code_tab1, code_tab2 = st.tabs(["Model Training Code", "Application Code"])
        
        with code_tab1:
            st.write("**Kidney Disease Model Training Code (kidney_disease_model.py)**")
            code = read_model_code('kidney_disease_model.py')
            st.code(code, language='python')
        
        with code_tab2:
            st.write("**Streamlit Application Code (app.py)**")
            app_code = read_model_code('app.py')
            st.code(app_code, language='python')
    
    with tab4:
        st.subheader("📁 Kidney Disease Dataset")
        df = load_dataset('Kidney disease.csv')
        if df is not None:
            st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(100), use_container_width=True)
            st.write("**Dataset Statistics:**")
            st.dataframe(df.describe(), use_container_width=True)

def thyroid_disease_prediction_tab(models_data):
    """Thyroid disease prediction tab"""
    # Card header
    create_disease_card("🦋", "Thyroid Disease Prediction", "linear-gradient(135deg, #fa709a 0%, #fee140 100%)")
    
    st.markdown("---")
    
    # Create sub-tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Evaluation Metrics", "🔮 Prediction", "💻 Model Code", "📁 Dataset"])
    
    with tab1:
        display_evaluation_metrics(models_data['results'], "Thyroid Disease")
    
    with tab2:
        st.subheader("🔮 Make a Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=20, max_value=80, value=40, key="thyroid_age",
                                 help="Age in years")
            gender = st.selectbox("Gender", ["F", "M"], key="thyroid_gender")
            smoking = st.selectbox("Smoking", ["No", "Yes"], key="thyroid_smoking")
            hx_smoking = st.selectbox("History of Smoking", ["No", "Yes"], key="thyroid_hx_smoking")
            hx_radiotherapy = st.selectbox("History of Radiotherapy", ["No", "Yes"], key="thyroid_radiotherapy")
            thyroid_function = st.selectbox("Thyroid Function", 
                                            ["Euthyroid", "Clinical Hyperthyroidism", "Clinical Hypothyroidism", 
                                             "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism"], 
                                            key="thyroid_function")
        
        with col2:
            physical_exam = st.selectbox("Physical Examination", 
                                         ["Normal", "Single nodular goiter-left", "Single nodular goiter-right", 
                                          "Multinodular goiter", "Diffuse goiter"], 
                                         key="thyroid_physical")
            adenopathy = st.selectbox("Adenopathy", ["No", "Right", "Left", "Bilateral", "Posterior", "Extensive"], 
                                     key="thyroid_adenopathy")
            pathology = st.selectbox("Pathology", ["Micropapillary", "Papillary", "Follicular", "Hurthel cell"], 
                                     key="thyroid_pathology")
            focality = st.selectbox("Focality", ["Uni-Focal", "Multi-Focal"], key="thyroid_focality")
            risk = st.selectbox("Risk", ["Low", "Intermediate", "High"], key="thyroid_risk")
            t = st.selectbox("T Stage", ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"], key="thyroid_t")
        
        with col3:
            n = st.selectbox("N Stage", ["N0", "N1a", "N1b"], key="thyroid_n")
            m = st.selectbox("M Stage", ["M0", "M1"], key="thyroid_m")
            stage = st.selectbox("Stage", ["I", "II", "III", "IVA", "IVB"], key="thyroid_stage")
            response = st.selectbox("Response", ["Excellent", "Indeterminate", "Biochemical Incomplete", 
                                                "Structural Incomplete"], key="thyroid_response")
        
        if st.button("Predict Thyroid Disease Recurrence", type="primary", key="thyroid_predict"):
            # Encode categorical variables
            label_encoders = models_data['label_encoders']
            
            # Prepare input data
            input_dict = {
                'Age': age,
                'Gender': label_encoders['Gender'].transform([gender])[0],
                'Smoking': label_encoders['Smoking'].transform([smoking])[0],
                'Hx Smoking': label_encoders['Hx Smoking'].transform([hx_smoking])[0],
                'Hx Radiothreapy': label_encoders['Hx Radiothreapy'].transform([hx_radiotherapy])[0],
                'Thyroid Function': label_encoders['Thyroid Function'].transform([thyroid_function])[0],
                'Physical Examination': label_encoders['Physical Examination'].transform([physical_exam])[0],
                'Adenopathy': label_encoders['Adenopathy'].transform([adenopathy])[0],
                'Pathology': label_encoders['Pathology'].transform([pathology])[0],
                'Focality': label_encoders['Focality'].transform([focality])[0],
                'Risk': label_encoders['Risk'].transform([risk])[0],
                'T': label_encoders['T'].transform([t])[0],
                'N': label_encoders['N'].transform([n])[0],
                'M': label_encoders['M'].transform([m])[0],
                'Stage': label_encoders['Stage'].transform([stage])[0],
                'Response': label_encoders['Response'].transform([response])[0]
            }
            
            # Create input array in correct order
            feature_names = models_data['feature_names']
            input_data = np.array([[input_dict[feature] for feature in feature_names]])
            
            # Scale input
            scaler = models_data['scaler']
            input_scaled = scaler.transform(input_data)
            
            # Get predictions
            predictions = {}
            target_encoder = models_data['target_encoder']
            
            for model_name, model in models_data['models'].items():
                if model_name in ["KNN", "Logistic Regression"]:
                    pred = model.predict(input_scaled)[0]
                    prob = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
                else:
                    pred = model.predict(input_data)[0]
                    prob = model.predict_proba(input_data)[0] if hasattr(model, 'predict_proba') else None
                predictions[model_name] = {
                    'prediction': pred,
                    'probability': prob
                }
            
            # Display results
            st.subheader("Prediction Results")
            result_cols = st.columns(4)
            
            for idx, (model_name, pred_data) in enumerate(predictions.items()):
                with result_cols[idx]:
                    pred_label = target_encoder.inverse_transform([pred_data['prediction']])[0]
                    result = "Yes" if pred_label == "Yes" else "No"
                    color = "🔴" if result == "Yes" else "🟢"
                    st.metric(f"{model_name}", f"{color} {result}")
                    if pred_data['probability'] is not None:
                        prob_idx = 1 if pred_data['prediction'] == 1 else 0
                        prob = pred_data['probability'][prob_idx]
                        st.write(f"Confidence: {prob*100:.2f}%")
    
    with tab3:
        st.subheader("💻 Code")
        code_tab1, code_tab2 = st.tabs(["Model Training Code", "Application Code"])
        
        with code_tab1:
            st.write("**Thyroid Disease Model Training Code (thyroid_disease_model.py)**")
            code = read_model_code('thyroid_disease_model.py')
            st.code(code, language='python')
        
        with code_tab2:
            st.write("**Streamlit Application Code (app.py)**")
            app_code = read_model_code('app.py')
            st.code(app_code, language='python')
    
    with tab4:
        st.subheader("📁 Thyroid Disease Dataset")
        df = load_dataset('Thyroid_Diff.csv')
        if df is not None:
            st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(100), use_container_width=True)
            st.write("**Dataset Statistics:**")
            st.dataframe(df.describe(), use_container_width=True)

def display_footer():
    """Display footer with credits and social links"""
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <div class="footer-content">
                <p class="footer-text">✨ Made by Aditya Raj ✨</p>
                <div class="social-links">
                    <a href="https://www.linkedin.com/in/adityaraj2112/" target="_blank" class="social-link">
                        <span>🔗</span> LinkedIn
                    </a>
                    <a href="https://github.com/Adityaraj86" target="_blank" class="social-link">
                        <span>💻</span> GitHub
                    </a>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_homepage():
    """Display homepage with disease cards"""
    st.markdown('<div class="homepage-container">', unsafe_allow_html=True)
    
    # Create cards in a grid with spacing
    col1, col2, col3, col4 = st.columns(4, gap="large")
    
    diseases = [
        {
            'icon': '❤️',
            'title': 'Heart Disease',
            'description': 'Cardiovascular health assessment',
            'key': 'heart'
        },
        {
            'icon': '🩺',
            'title': 'Diabetes',
            'description': 'Blood glucose & insulin analysis',
            'key': 'diabetes'
        },
        {
            'icon': '🫘',
            'title': 'Kidney Disease',
            'description': 'Renal function evaluation',
            'key': 'kidney'
        },
        {
            'icon': '🦋',
            'title': 'Thyroid Disease',
            'description': 'Thyroid function & recurrence analysis',
            'key': 'thyroid'
        }
    ]
    
    for idx, col in enumerate([col1, col2, col3, col4]):
        disease = diseases[idx]
        with col:
            st.markdown(f"""
                <div class="disease-card-home">
                    <div class="card-icon-container">
                        <div class="card-icon-large">{disease['icon']}</div>
                    </div>
                    <div class="card-title-home">{disease['title']}</div>
                    <div class="card-description">{disease['description']}</div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Explore Model", key=f"btn_{disease['key']}", use_container_width=True):
                st.session_state.selected_disease = disease['key']
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application"""
    # Enhanced title - combine in single container to reduce spacing
    st.markdown("""
        <div class="title-container">
            <h1 class="main-header">🏥 Disease Prediction System</h1>
            <p class="subtitle">AI-Powered Medical Diagnosis Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    # Standard tabs navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩺 Diabetes",
        "❤️ Heart Disease",
        "🫘 Kidney Disease",
        "🦋 Thyroid Disease"
    ])
    
    with tab1:
        diabetes_prediction_tab(models['diabetes'])
    with tab2:
        heart_disease_prediction_tab(models['heart'])
    with tab3:
        kidney_disease_prediction_tab(models['kidney'])
    with tab4:
        thyroid_disease_prediction_tab(models['thyroid'])
    
    # Display footer
    display_footer()

if __name__ == "__main__":
    main()
