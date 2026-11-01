import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for background gradient and professional form styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main-header {
        color: #2c3e50;
        text-align: center;
        font-weight: bold;
    }
    .sub-text {
        text-align: center;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='main-header'>🎓 Student Exam Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Please fill out the required fields below to predict the expected student exam score using Machine Learning.</p>", unsafe_allow_html=True)

# 1. Load dataset securely
@st.cache_data
def load_data():
    df = pd.read_csv('student_performance_messy.csv')
    df = df.dropna(subset=['exam_score'])
    return df

df = load_data()

# 2. Process features and target with numeric validation
features = ['study_hours', 'attendance_percent', 'previous_score', 'sleep_hours']
X = df[features].apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())
y = pd.to_numeric(df['exam_score'], errors='coerce')

valid_idx = y.notna()
X = X[valid_idx]
y = y[valid_idx]

# Train Machine Learning Model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# 3. Professional Form Style with Required Markers (*)
st.markdown("---")
with st.form("exam_prediction_form"):
    st.subheader("📋 Student Performance Details Form")
    st.markdown("Fields marked with <span style='color:red;'>*</span> are required.", unsafe_allow_html=True)
    
    study_hours = st.slider("Daily Study Hours <span style='color:red;'>*</span>", 1.0, 10.0, 5.0, help="Required field")
    attendance_percent = st.slider("Attendance Percentage (%) <span style='color:red;'>*</span>", 50.0, 100.0, 80.0, help="Required field")
    previous_score = st.slider("Previous Exam Score <span style='color:red;'>*</span>", 30.0, 100.0, 70.0, help="Required field")
    sleep_hours = st.slider("Daily Sleep Hours <span style='color:red;'>*</span>", 4.0, 10.0, 7.0, help="Required field")
    
    # Form Submit Button
    submitted = st.form_submit_button("🔮 Predict Exam Score", use_container_width=True)

if submitted:
    new_data = pd.DataFrame({
        'study_hours': [study_hours],
        'attendance_percent': [attendance_percent],
        'previous_score': [previous_score],
        'sleep_hours': [sleep_hours]
    })
    
    prediction = model.predict(new_data)
    
    st.markdown("---")
    st.success(f"### 🎉 Expected Exam Score: **{round(prediction[0], 2)} / 100**")
