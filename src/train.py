from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from preprocess import (
    load_data,
    clean_data,
    get_features_target
)


def get_models():
    """
    Returns all models to compare.
    """

    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Decision Tree": DecisionTreeClassifier(random_state=42),

        "Random Forest": RandomForestClassifier(random_state=42),

        "K-Nearest Neighbors": KNeighborsClassifier(),

        "Support Vector Machine": SVC(),

        "Naive Bayes": GaussianNB()
    }


def train_models():

    # Load and prepare data
    df = clean_data(load_data())

    X, y = get_features_target(df)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    results = []

    best_model = None
    best_accuracy = 0

    for name, model in get_models().items():

        # Models that require feature scaling
        if name in [
            "Logistic Regression",
            "K-Nearest Neighbors",
            "Support Vector Machine"
        ]:

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])

        else:

            pipeline = Pipeline([
                ("model", model)
            ])

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        predictions = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, predictions)

        # Metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        # Save the best model in memory
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = pipeline

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    print("\nModel Comparison")
    print("=" * 70)

    print(results_df)

    print("\nConfusion Matrix of Best Model")
    print("=" * 70)

    best_predictions = best_model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        best_predictions
    )

    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.show()

    # Save the best model
    models_folder = (
        Path(__file__).resolve().parent.parent
        / "models"
    )

    models_folder.mkdir(exist_ok=True)

    with open(models_folder / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("\nBest model saved successfully!")


def main():

    train_models()


if __name__ == "__main__":
    main()