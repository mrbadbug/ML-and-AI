import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import shap

# ----------------------
# Project Paths
# ----------------------
project_path = r"C:\Users\Monir Hussain\Downloads\Dmml Project_"
model_path = os.path.join(project_path, "new_model.pkl")

# ----------------------
# Load Model
# ----------------------
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
    st.stop()

# Load the trained model
model = joblib.load(model_path)

# Get the feature names the model expects
if hasattr(model, "feature_names_in_"):
    feature_names = model.feature_names_in_
else:
    st.error("Model does not have feature_names_in_ attribute. Cannot determine input features.")
    st.stop()

# ----------------------
# App Title
# ----------------------
st.title("🧠 Health Risk Prediction Dashboard")
st.markdown("---")

# ----------------------
# Sidebar: User Inputs
# ----------------------
st.sidebar.header("Enter Your Details")
input_data = {}

for feature in feature_names:
    # Treat certain features as binary / categorical
    if any(feature.startswith(prefix) for prefix in [
        'Gender_', 'Sleep_Quality_', 'Physical_Activity_', 'Diet_Quality_',
        'Social_Support_', 'Relationship_Status_', 'Substance_Use_',
        'Counseling_Service_Use_', 'Family_History_', 'Chronic_Illness_',
        'Extracurricular_Involvement_', 'Residence_Type_', 'Course_'
    ]):
        # Binary features: select 0 or 1
        input_data[feature] = st.sidebar.selectbox(f"{feature.replace('_',' ')}?", [0, 1])
    else:
        # Numerical features: integer input
        input_data[feature] = st.sidebar.number_input(
            f"{feature.replace('_',' ')}", 
            value=0, 
            step=1, 
            format="%d"
        )

# Convert input dictionary to DataFrame for model
input_df = pd.DataFrame([input_data])

# ----------------------
# Prediction Logic
# ----------------------
if st.sidebar.button("🔍 Predict Risk"):

    # Predict class and probabilities
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    # ----------------------
    # Display Prediction
    # ----------------------
    st.subheader("📌 Prediction Result")
    st.write(f"### 🩺 **Predicted Risk Level:** `{prediction}`")

    # ----------------------
    # 1. Prediction Probabilities (Progress Bars)
    # ----------------------
    st.write("### 📊 Prediction Probabilities")
    risk_labels = model.classes_
    for i, label in enumerate(risk_labels):
        st.progress(float(prediction_proba[i]))
        st.write(f"**{label}**: {prediction_proba[i]*100:.2f}%")

    # ----------------------
    # 2. Gauge Chart for Highest Risk Probability
    # ----------------------
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_proba.max() * 100,
        title={'text': "Predicted Risk Probability (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red"},
               'steps': [
                   {'range': [0, 40], 'color': "green"},
                   {'range': [40, 70], 'color': "yellow"},
                   {'range': [70, 100], 'color': "red"}],
               }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ----------------------
    # 3. Risk Distribution Bar Chart
    # ----------------------
    st.subheader("📊 Risk Probability Distribution")
    df_proba = pd.DataFrame({
        "Risk Level": risk_labels,
        "Probability": prediction_proba
    })
    fig_bar = px.bar(df_proba, x="Risk Level", y="Probability",
                     color="Risk Level", text="Probability", title="Risk Prediction Probability")
    fig_bar.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)
