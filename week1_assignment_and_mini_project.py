"""
WEEK 1 - ASSIGNMENT + MINI PROJECT (single-file submission)

Assignment answers
1. Dataset summary: pd.read_csv() loads the CSV; df.info() reports column
   types/nulls, and df.describe() provides numeric statistics.
2. Missing data: Age is filled with its median (robust to outliers), Embarked
   is filled with its mode, and Cabin is removed because 77% of it is missing.
3. Encoding: Sex is label encoded (female=0, male=1); Embarked is one-hot
   encoded so ports are not interpreted as ordered values.

Mini project tasks completed
- Clean Titanic missing data
- Encode Sex and Embarked
- Visualize passenger age distribution
- Export a cleaned CSV

Run: python week1_assignment_and_mini_project.py
Input: titanic_raw.csv
Outputs: titanic_cleaned.csv and age_distribution.png (or .svg fallback)
"""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    plt = None
    sns = None


INPUT_FILE = BASE_DIR / "titanic_raw.csv"
OUTPUT_FILE = BASE_DIR / "titanic_cleaned.csv"
PLOT_FILE = BASE_DIR / "age_distribution.png"
FALLBACK_PLOT_FILE = BASE_DIR / "age_distribution.svg"


def create_age_distribution(df: pd.DataFrame) -> Path:
    """Create the required Age chart using Seaborn/Matplotlib when available."""
    if plt is not None and sns is not None:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(9, 5.5))
        sns.histplot(data=df, x="Age", bins=30, kde=True, color="#2a6fbb")
        plt.title("Titanic Passenger Age Distribution")
        plt.xlabel("Age (years)")
        plt.ylabel("Number of passengers")
        plt.tight_layout()
        plt.savefig(PLOT_FILE, dpi=180)
        plt.close()
        return PLOT_FILE

    # A portable fallback for systems without optional plotting libraries.
    counts = pd.cut(df["Age"], bins=20, include_lowest=True).value_counts(sort=False)
    width, height, margin = 900, 500, 70
    chart_width, chart_height = width - 2 * margin, height - 2 * margin
    bar_width = chart_width / len(counts)
    max_count = counts.max()
    bars = []
    for index, count in enumerate(counts):
        bar_height = (count / max_count) * chart_height
        x = margin + index * bar_width
        y = height - margin - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" '
            f'height="{bar_height:.1f}" fill="#2a6fbb"/>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="35" text-anchor="middle" font-family="Arial" font-size="22">Titanic Passenger Age Distribution</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>
{''.join(bars)}
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="16">Age (years)</text>
<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-family="Arial" font-size="16">Number of passengers</text>
</svg>'''
    FALLBACK_PLOT_FILE.write_text(svg, encoding="utf-8")
    return FALLBACK_PLOT_FILE


def main() -> None:
    """Run Assignment tasks 1-3 and the Titanic mini project."""
    df = pd.read_csv(INPUT_FILE)

    # Assignment 1: load the CSV and summarize it.
    print("FIRST 10 ROWS:\n", df.head(10))
    print("\nDATASET INFORMATION:")
    df.info()
    print("\nNUMERICAL STATISTICS:\n", df.describe())
    print("\nMISSING VALUES BEFORE CLEANING:\n", df.isna().sum())

    # Assignment 2: handle missing values.
    df = df.drop(columns=["Cabin"])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Assignment 3: encode categorical variables.
    df["Sex_encoded"] = df["Sex"].map({"female": 0, "male": 1})
    embarked = pd.get_dummies(df["Embarked"], prefix="Embarked", dtype=int)
    df = pd.concat([df.drop(columns=["Sex", "Embarked"]), embarked], axis=1)

    # Mini project output: chart and cleaned dataset.
    plot_file = create_age_distribution(df)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nMISSING VALUES AFTER CLEANING:\n", df.isna().sum())
    print(f"\nSaved: {OUTPUT_FILE.name} - {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Saved visualization: {plot_file.name}")


if __name__ == "__main__":
    main()
