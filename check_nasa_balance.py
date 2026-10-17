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
    df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print(file)
    print("=" * 60)

    if "defects" in df.columns:
        target = "defects"
    elif "problems" in df.columns:
        target = "problems"
    else:
        print("Target column not found")
        continue

    print(df[target].value_counts())

    print("\nPercentages:")
    print(
        df[target]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )