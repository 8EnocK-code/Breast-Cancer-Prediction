# Breast Cancer Prediction Using Machine Learning

## Project Overview

This project applies machine learning techniques to predict whether a breast tumor is **benign** or **malignant** using the Breast Cancer Wisconsin Diagnostic Dataset. The goal is to build a reliable classification model while following a structured data science workflow, from data preprocessing and exploration to  model training, evaluation, and prediction.

This project was developed as part of my data science learning journey to strengthen my understanding of classification algorithms, model evaluation, and building reproducible machine learning pipelines.

---

## Problem Statement

Breast cancer is one of the most common forms of cancer worldwide. Early detection plays a significant role in improving treatment outcomes. Using measurements computed from digitized images of breast cell nuclei, this project aims to classify tumors as either:

* **Malignant (M)** – Cancerous
* **Benign (B)** – Non-cancerous

---

## Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset**, which contains numerical features describing characteristics of cell nuclei obtained from breast mass images.

Examples of features include:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Symmetry

The target variable is **Diagnosis**, where:

* **1 = Malignant**
* **0 = Benign**

---

## Project Structure

```text
Breast-Cancer-Prediction/

├── data/
│   └── data.csv
│
├── models/
│   └── breast_cancer_pipeline.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── reports/
│   └── classification_report.txt
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Pickle

---

## Machine Learning Workflow

The project follows the following workflow:

1. Load the dataset.
2. Clean and preprocess the data.
3. Split the dataset into training and testing sets.
4. Train multiple machine learning models.
5. Compare model performance.
6. Select the best-performing model.
7. Save the trained model for future predictions.

---

## Models Compared

The following classification algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Gaussian Naive Bayes

---

## Model Evaluation

The models are evaluated using several performance metrics, including:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Cross Validation
* Classification Report

These metrics provide a more complete assessment of model performance than accuracy alone.

---

## How to Run the Project

### Clone the repository

```bash
git clone <repository-url>
```

### Move into the project directory

```bash
cd Breast-Cancer-Prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the models

```bash
python src/train.py
```

The best-performing model will be saved in the `models` directory.

---

## Future Improvements

Some planned improvements include:

* Hyperparameter tuning
* Feature selection
* Interactive Streamlit web application
* Model deployment
* Additional visualization dashboards

---

## What I Learned

Through this project, I gained practical experience in:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Classification algorithms
* Model comparison
* Performance evaluation
* Machine learning pipelines
* Organizing a machine learning project for reproducibility

---

## Author

This project was built as part of my journey toward becoming a professional Data Scientist. Feedback, suggestions, and contributions are always welcome.
