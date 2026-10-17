import pandas as pd
import os

dataset_dir = r"C:\QASense-AI\datasets\NASA-promise-dataset-repository-main"

files = [
    "cm1.csv",
    "jm1.csv",
    "kc1.csv",
    "kc2.csv",
    "pc1.csv"
]

for file in files:
    path = os.path.join(dataset_dir, file)

    print("\n" + "=" * 70)
    print(f"DATASET: {file}")
    print("=" * 70)

    df = pd.read_csv(path)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)