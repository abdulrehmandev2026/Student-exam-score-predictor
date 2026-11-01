import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Page title
st.title("Student Exam Score Predictor")
st.write("Apni study hours, attendance aur baqi details enter karein taake expected exam score pata chal sake.")

# 1. Dataset Load karna
@st.cache_data
def load_data():
    df = pd.read_csv('student_performance_messy.csv')
    df = df.dropna(subset=['exam_score'])
    return df

df = load_data()

# 2. Sirf numeric columns lena taake median mein koi error na aaye
features = ['study_hours', 'attendance_percent', 'previous_score', 'sleep_hours']
X = df[features].apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())
y = pd.to_numeric(df['exam_score'], errors='coerce')

# Missing target values drop karna agar koi hon
valid_idx = y.notna()
X = X[valid_idx]
y = y[valid_idx]

# Model Train karna
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# 3. User se Inputs lena (Sliders)
st.header("Apni Details Darj Karein:")
study_hours = st.slider("Study Hours (Rozana kitni parhai karte hain)", 1.0, 10.0, 5.0)
attendance_percent = st.slider("Attendance Percent (%)", 50.0, 100.0, 80.0)
previous_score = st.slider("Previous Score", 30.0, 100.0, 70.0)
sleep_hours = st.slider("Sleep Hours (Neend ke ghantay)", 4.0, 10.0, 7.0)

# 4. Predict button
if st.button("Predict Exam Score"):
    new_data = pd.DataFrame({
        'study_hours': [study_hours],
        'attendance_percent': [attendance_percent],
        'previous_score': [previous_score],
        'sleep_hours': [sleep_hours]
    })
    
    # Prediction karna
    prediction = model.predict(new_data)
    
    # Result show karna
    st.success(f"Mutawaqa (Predicted) Exam Score: {round(prediction[0], 2)}")
