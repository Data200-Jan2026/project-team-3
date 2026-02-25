import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Page Config
st.set_page_config(
    page_title="AI Salary Change Predictor",
    page_icon="📊",
    layout="centered"
)

# Title Section
st.title("📊 AI Salary Change Predictor")
st.markdown(
    "This application predicts **salary change percentage** based on AI adoption, "
    "automation risk, experience, and skill shift level."
)

st.divider()

# Load Dataset
df = pd.read_csv("ai_job_replacement_skill_shift_100.csv")
df.columns = df.columns.str.strip()

# Features and Target
X = df[
    [
        "experience_years",
        "ai_adoption_level",
        "automation_risk",
        "skill_shift_level",
    ]
]

y = df["salary_change_percent"]

# Train Model
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
accuracy = r2_score(y, y_pred)

st.info(f"Model R² Score: {accuracy:.2f}")

st.divider()

# Sidebar Inputs
st.sidebar.header("Enter Input Values")

experience = st.sidebar.slider("Experience Years", 0, 30, 5)
ai_level = st.sidebar.slider("AI Adoption Level", 0, 10, 5)
automation = st.sidebar.slider("Automation Risk", 0, 10, 5)
skill_shift = st.sidebar.slider("Skill Shift Level", 0, 10, 5)

# Prediction Section
st.subheader("Prediction")

if st.button("Predict Salary Change"):

    input_data = np.array([[experience, ai_level, automation, skill_shift]])
    prediction = model.predict(input_data)

    predicted_value = max(0, min(100, prediction[0]))

    st.success(f"Predicted Salary Change: {predicted_value:.2f}%")

    # Prediction Visualization
    fig1, ax1 = plt.subplots()
    ax1.bar(["Predicted Salary Change"], [predicted_value])
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Percentage (%)")
    st.pyplot(fig1)

st.divider()

# Feature Importance Section
st.subheader("Feature Importance")

coefficients = model.coef_
features = X.columns

fig2, ax2 = plt.subplots()
ax2.bar(features, coefficients)
ax2.set_ylabel("Coefficient Value")
ax2.set_xticklabels(features, rotation=45)
st.pyplot(fig2)

st.markdown(
    """
    **Interpretation:**  
    Higher positive coefficient → greater impact on salary increase.  
    Negative coefficient → may reduce salary growth.
    """
)