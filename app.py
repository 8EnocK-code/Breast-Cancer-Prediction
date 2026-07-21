import streamlit as st
import pandas as pd

from src.predict import load_model, predict

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

model = load_model()

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🩺 Breast Cancer Prediction System")
st.caption(
    "Machine Learning Classification using the Breast Cancer Wisconsin Diagnostic Dataset"
)
st.markdown("""
Predict whether a breast tumor is **Benign** or **Malignant**
using a trained Machine Learning model.

⚠️ This project is for educational purposes only.
""")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("About")

st.sidebar.success("✅ Model Loaded")

st.sidebar.markdown("""
### Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
- Naive Bayes

The best performing model was saved
after comparison.
""")

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
with st.expander("📘 How to Use This Application"):

    st.markdown("""
1. Enter the patient's measurements.

2. Click **Predict**.

3. Review the prediction and confidence score.

4. Compare the probability of each diagnosis.

**Reminder:** This application is intended for learning and demonstration purposes only.
""")
    
with st.expander("ℹ️ About this Project"):

    st.markdown("""
### Project Overview

This application predicts whether a breast tumor is **Benign** or **Malignant**
using a trained Machine Learning model.

### Workflow

- Data Cleaning
- Exploratory Data Analysis
- Feature Scaling
- Model Training
- Model Evaluation
- Prediction

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
""")
st.header("Patient Measurements")

st.caption(
    "Enter the patient's measurements below."
)
st.write(
    "Fill in all measurements below."
)

# ==================================================
# MEAN MEASUREMENTS
# ==================================================

with st.expander("Mean Measurements", expanded=True):

    col1, col2 = st.columns(2)

    with col1:

        radius_mean = st.number_input(
            "Radius Mean",
            value=14.0
        )

        texture_mean = st.number_input(
            "Texture Mean",
            value=19.0
        )

        perimeter_mean = st.number_input(
            "Perimeter Mean",
            value=92.0
        )

        area_mean = st.number_input(
            "Area Mean",
            value=650.0
        )

        smoothness_mean = st.number_input(
            "Smoothness Mean",
            value=0.10,
            format="%.5f"
        )

    with col2:

        compactness_mean = st.number_input(
            "Compactness Mean",
            value=0.10,
            format="%.5f"
        )

        concavity_mean = st.number_input(
            "Concavity Mean",
            value=0.08,
            format="%.5f"
        )

        concave_points_mean = st.number_input(
            "Concave Points Mean",
            value=0.05,
            format="%.5f"
        )

        symmetry_mean = st.number_input(
            "Symmetry Mean",
            value=0.18,
            format="%.5f"
        )

        fractal_dimension_mean = st.number_input(
            "Fractal Dimension Mean",
            value=0.06,
            format="%.5f"
        )

# ==================================================
# STANDARD ERROR
# ==================================================

with st.expander("Standard Error Measurements"):

    col1, col2 = st.columns(2)

    with col1:

        radius_se = st.number_input(
            "Radius SE",
            value=0.40
        )

        texture_se = st.number_input(
            "Texture SE",
            value=1.20
        )

        perimeter_se = st.number_input(
            "Perimeter SE",
            value=2.80
        )

        area_se = st.number_input(
            "Area SE",
            value=40.0
        )

        smoothness_se = st.number_input(
            "Smoothness SE",
            value=0.005,
            format="%.5f"
        )

    with col2:

        compactness_se = st.number_input(
            "Compactness SE",
            value=0.020,
            format="%.5f"
        )

        concavity_se = st.number_input(
            "Concavity SE",
            value=0.020,
            format="%.5f"
        )

        concave_points_se = st.number_input(
            "Concave Points SE",
            value=0.010,
            format="%.5f"
        )

        symmetry_se = st.number_input(
            "Symmetry SE",
            value=0.020,
            format="%.5f"
        )

        fractal_dimension_se = st.number_input(
            "Fractal Dimension SE",
            value=0.003,
            format="%.5f"
        )

        # ==================================================
# WORST MEASUREMENTS
# ==================================================

with st.expander("Worst Measurements"):

    col1, col2 = st.columns(2)

    with col1:

        radius_worst = st.number_input(
            "Radius Worst",
            value=16.0
        )

        texture_worst = st.number_input(
            "Texture Worst",
            value=25.0
        )

        perimeter_worst = st.number_input(
            "Perimeter Worst",
            value=105.0
        )

        area_worst = st.number_input(
            "Area Worst",
            value=850.0
        )

        smoothness_worst = st.number_input(
            "Smoothness Worst",
            value=0.14,
            format="%.5f"
        )

    with col2:

        compactness_worst = st.number_input(
            "Compactness Worst",
            value=0.25,
            format="%.5f"
        )

        concavity_worst = st.number_input(
            "Concavity Worst",
            value=0.30,
            format="%.5f"
        )

        concave_points_worst = st.number_input(
            "Concave Points Worst",
            value=0.15,
            format="%.5f"
        )

        symmetry_worst = st.number_input(
            "Symmetry Worst",
            value=0.30,
            format="%.5f"
        )

        fractal_dimension_worst = st.number_input(
            "Fractal Dimension Worst",
            value=0.08,
            format="%.5f"
        )

# --------------------------------------------------
# BUILD INPUT DATAFRAME
# --------------------------------------------------

patient_data = pd.DataFrame({

    "radius_mean": [radius_mean],
    "texture_mean": [texture_mean],
    "perimeter_mean": [perimeter_mean],
    "area_mean": [area_mean],
    "smoothness_mean": [smoothness_mean],
    "compactness_mean": [compactness_mean],
    "concavity_mean": [concavity_mean],
    "concave points_mean": [concave_points_mean],
    "symmetry_mean": [symmetry_mean],
    "fractal_dimension_mean": [fractal_dimension_mean],

    "radius_se": [radius_se],
    "texture_se": [texture_se],
    "perimeter_se": [perimeter_se],
    "area_se": [area_se],
    "smoothness_se": [smoothness_se],
    "compactness_se": [compactness_se],
    "concavity_se": [concavity_se],
    "concave points_se": [concave_points_se],
    "symmetry_se": [symmetry_se],
    "fractal_dimension_se": [fractal_dimension_se],

    "radius_worst": [radius_worst],
    "texture_worst": [texture_worst],
    "perimeter_worst": [perimeter_worst],
    "area_worst": [area_worst],
    "smoothness_worst": [smoothness_worst],
    "compactness_worst": [compactness_worst],
    "concavity_worst": [concavity_worst],
    "concave points_worst": [concave_points_worst],
    "symmetry_worst": [symmetry_worst],
    "fractal_dimension_worst": [fractal_dimension_worst],
})

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.warning(
    "Please ensure that all measurements are entered correctly before making a prediction."
)

st.divider()

if st.button("🔍 Predict", use_container_width=True):

    prediction, probability = predict(model, patient_data)

    diagnosis = "Malignant" if prediction == 1 else "Benign"

    st.header("Prediction Results")

    if diagnosis == "Benign":

        st.success("## 🟢 Benign Tumor")

        st.write("""
    The model predicts that the tumor is **Benign**.

    This indicates that the tumor is unlikely to be cancerous.

    **Note:** This prediction is for educational purposes only and should not replace professional medical advice.
    """)

    else:

        st.error("## 🔴 Malignant Tumor")

        st.write("""
    The model predicts that the tumor is **Malignant**.

    This indicates that the tumor may be cancerous and requires further medical evaluation.

    **Note:** This prediction is for educational purposes only and should not replace professional medical advice.
    """)

    if probability is not None:

        confidence = max(probability) * 100

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Prediction",
                diagnosis
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        st.progress(confidence / 100)

        if confidence >= 90:

            st.success("The model is highly confident in this prediction.")

        elif confidence >= 75:

            st.info("The model is reasonably confident in this prediction.")

        else:

            st.warning("The model has relatively low confidence in this prediction.")

        st.caption(
            "Higher confidence indicates that the model is more certain about its prediction."
        )
        probability_df = pd.DataFrame({

            "Diagnosis": [
                "Benign",
                "Malignant"
            ],

            "Probability (%)": [
                probability[0] * 100,
                probability[1] * 100
            ]

        })

        st.subheader("Prediction Probabilities")

        st.dataframe(
            probability_df.style.format({
                "Probability (%)": "{:.2f}"
            }),
            use_container_width=True
        )
        st.divider()

        st.caption(
            "Built with ❤️ using Python, Scikit-learn and Streamlit."
        )
        st.divider()

        st.caption(
            "Developed by Enock Ombaso | Data Science Portfolio Project"
        )