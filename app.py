import streamlit as st
import pandas as pd

from src.predict import load_model

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

st.header("Patient Measurements")

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
# PREDICT BUTTON (Coming in Version 2)
# --------------------------------------------------

st.divider()

st.button(
    "🔍 Predict",
    disabled=True,
    use_container_width=True
)

st.info(
    "Prediction functionality will be enabled in Version 2."
)