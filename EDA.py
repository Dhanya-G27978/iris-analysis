from sklearn.datasets import load_iris
import pandas as pd


def load_iris_data():
    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["species"] = [
        iris.target_names[i]
        for i in iris.target
    ]

    return df

def missing_values(df):
    print("\nMissing Values:")
    print(df.isnull().sum())


def main():
    df = load_iris_data()
    missing_values(df)

if __name__ == "__main__":
    main()