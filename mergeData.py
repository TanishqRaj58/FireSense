import pandas as pd

df23 = pd.read_csv("Data\dataset_2023.csv")
df24 = pd.read_csv("Data\dataset_2024.csv")

train_df = pd.concat([df23, df24], ignore_index=True)
train_df.to_csv("train_data.csv", index=False)

print("✅ Training data ready")