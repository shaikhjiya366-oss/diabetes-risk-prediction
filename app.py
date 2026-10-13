
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# Load the saved machine learning model
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "diabetes_rf_model.pkl"
    return joblib.load(model_path)

try:
    model = load_model()
except Exception:
    st.error("Model file could not be loaded.")
    st.write(
        "Please check that diabetes_rf_model.pkl "
        "is uploaded in the same repository as app.py."
    )
    st.stop()

# Application heading
st.title("🩺 Diabetes Risk Prediction System")
st.write(
    "An educational application using Machine Learning "
    "to classify diabetes-related data."
)

st.warning(
    "This application is for educational demonstration only. "
    "It does not diagnose diabetes or replace medical advice."
)

st.divider()

# Input form
st.subheader("Enter the sample measurements")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Number of pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=1.0,
            max_value=300.0,
            value=120.0
        )

        blood_pressure = st.number_input(
            "Blood pressure (mm Hg)",
            min_value=1.0,
            max_value=200.0,
            value=70.0
        )

        skin_thickness = st.number_input(
            "Skin thickness (mm)",
            min_value=1.0,
            max_value=100.0,
            value=20.0
        )

    with col2:
        insulin = st.number_input(
            "Insulin",
            min_value=1.0,
            max_value=900.0,
            value=80.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=1.0,
            max_value=80.0,
            value=28.0
        )

        pedigree = st.number_input(
            "Diabetes pedigree function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=30
        )

    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True
    )

# Run the prediction when the button is clicked
if submitted:

    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }])

    try:
        prediction = int(model.predict(input_data)[0])

        st.subheader("Prediction result")

        if prediction == 1:
            st.warning(
                "The model classified this input "
                "as the diabetes-positive class."
            )
        else:
            st.success(
                "The model classified this input "
                "as the diabetes-negative class."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            positive_index = classes.index(1)
            probability = probabilities[positive_index]

            st.metric(
                "Model-estimated positive-class probability",
                f"{probability:.1%}"
            )

            st.caption(
                "This score is not a clinically validated "
                "estimate of an individual's medical risk."
            )

        with st.expander("View entered measurements"):
            st.dataframe(input_data, hide_index=True)

    except Exception:
        st.error(
            "Prediction failed. Check the saved model "
            "and scikit-learn version."
        )

# Project information
st.divider()
st.subheader("About this project")

st.write("""
- Domain: Machine Learning and Healthcare Data Analytics
- Task: Supervised binary classification
- Algorithm: Random Forest Classifier
- Dataset: Pima Indians Diabetes dataset
- Input features: Eight numerical measurements
""")

st.caption(
    "Educational prototype only. Do not use this application "
    "to make medical decisions or enter real patient information."
)
