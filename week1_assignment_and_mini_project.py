"""Week 1 Assignment + Titanic Data Cleaning Mini Project."""
from pathlib import Path
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    plt = sns = None

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "titanic_raw.csv"
OUTPUT_FILE = BASE_DIR / "titanic_cleaned.csv"
PLOT_FILE = BASE_DIR / "age_distribution.png"

def main():
    # Assignment answers: info() reports dtypes/nulls and describe() gives stats.
    # Age uses median, Embarked uses mode, Cabin is dropped; Sex is label encoded
    # and Embarked is one-hot encoded.
    df = pd.read_csv(INPUT_FILE)
    print("First 10 rows:", df.head(10))
    print("Dataset information:")
    df.info()
    print("Statistics:", df.describe())
    print("Missing values before cleaning:", df.isna().sum())

    df = df.drop(columns=["Cabin"])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Sex_encoded"] = df["Sex"].map({"female": 0, "male": 1})
    ports = pd.get_dummies(df["Embarked"], prefix="Embarked", dtype=int)
    df = pd.concat([df.drop(columns=["Sex", "Embarked"]), ports], axis=1)

    if plt is not None and sns is not None:
        sns.set_theme(style="whitegrid")
        sns.histplot(df["Age"], bins=30, kde=True, color="#2a6fbb")
        plt.title("Titanic Passenger Age Distribution")
        plt.xlabel("Age (years)")
        plt.ylabel("Number of passengers")
        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=180)
        plt.close()
    df.to_csv(OUTPUT_FILE, index=False)
    print("Missing values after cleaning:", df.isna().sum())
    print(f"Saved {OUTPUT_FILE.name}: {df.shape[0]} rows, {df.shape[1]} columns")

if __name__ == "__main__":
    main()
