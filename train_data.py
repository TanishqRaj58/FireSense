import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("🔥 FireSense AI Training Started")
# ===============================
# STEP 1: LOAD DATA
# ===============================
df23 = pd.read_csv("Data/dataset_2023.csv")
df24 = pd.read_csv("Data/dataset_2024.csv")
df25 = pd.read_csv("Data/dataset_2025.csv")
# ===============================
# STEP 2: TRAIN–TEST SPLIT
# ===============================
train_df = pd.concat([df23, df24], ignore_index=True)
test_df = df25.copy()
print("Training rows:", len(train_df))
print("Testing rows :", len(test_df))
# ===============================
# STEP 3: FEATURES
# ===============================
features = [
    "temp_C",
    "CH4_percent",
    "CO_ppm",
    "O2_percent",
    "airflow_m_s",
    "humidity_percent",
    "dust_mg_m3",
    "pressure_Pa"
]
X_train = train_df[features]
X_test = test_df[features]
# ===============================
# STEP 4: LABEL ENCODING
# ===============================
le = LabelEncoder()
y_train = le.fit_transform(train_df["safety_status"])
y_test = le.transform(test_df["safety_status"])
print("Classes:", le.classes_)
# ===============================
# STEP 5: TRAIN MODEL
# ===============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
print("Training Random Forest...")
model.fit(X_train, y_train)
# ===============================
# STEP 6: TEST MODEL (2025)
# ===============================
y_pred = model.predict(X_test)
print("\n📊 ACCURACY")
print(accuracy_score(y_test, y_pred))
print("\n📊 CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("📊 CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))
# ===============================
# STEP 7: SAVE MODEL

with open("firesense_rf_model.pkl", "wb") as f:
    pickle.dump((model, le, features), f)
print("\n✅ MODEL SAVED SUCCESSFULLY")