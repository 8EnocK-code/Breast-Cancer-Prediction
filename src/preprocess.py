from pathlib import Path

import pandas as pd


def load_data():
    """
    Load the breast cancer dataset.
    """

    data_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "data.csv"
    )

    df = pd.read_csv(data_path)

    return df


def clean_data(df):
    """
    Clean the dataset.
    """

    # Remove unnecessary columns
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    # Convert diagnosis to numbers
    df["diagnosis"] = df["diagnosis"].map({
        "M": 1,
        "B": 0
    })

    return df


def get_features_target(df):
    """
    Split the dataframe into features and target.
    """

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    return X, y


def main():

    df = load_data()

    print("=" * 50)
    print("Original Dataset")
    print("=" * 50)
    print(df.head())

    df = clean_data(df)

    print("\n")
    print("=" * 50)
    print("Clean Dataset")
    print("=" * 50)
    print(df.head())

    print("\n")
    print(df.info())


if __name__ == "__main__":
    main()