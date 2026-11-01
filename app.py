import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Page configuration with wide layout to cover side spaces
st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="🎓",
    layout="wide"
)

# Title and description
st.title("🎓 Student Exam Score Predictor")
st.markdown("Please fill out the form below with the required student details to predict the expected exam score using Machine Learning.")

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

# 3. Form layout using columns so it looks balanced on wide screens
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    with st.form("exam_prediction_form"):
        st.subheader("📋 Student Performance Form")
        st.markdown("Fields marked with <span style='color:red;'>*</span> are required.", unsafe_allow_html=True)
        
        study_hours = st.number_input("Daily Study Hours *", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
        attendance_percent = st.number_input("Attendance Percentage (%) *", min_value=50.0, max_value=100.0, value=80.0, step=1.0)
        previous_score = st.number_input("Previous Exam Score *", min_value=30.0, max_value=100.0, value=70.0, step=1.0)
        sleep_hours = st.number_input("Daily Sleep Hours *", min_value=4.0, max_value=10.0, value=7.0, step=0.5)
        
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

# Made by tag in the corner
st.markdown("---")
st.markdown("<p style='text-align: right; color: gray; font-size: 13px;'>Made by Abdul Rehman</p>", unsafe_allow_html=True)
