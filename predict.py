import pickle
import pandas as pd
with open("firesense_rf_model.pkl", "rb") as f:
    model, encoder, features = pickle.load(f)
def predict_status(sensor_input):
    df = pd.DataFrame([sensor_input])[features]
    pred = model.predict(df)[0]
    prob = model.predict_proba(df).max()
    label = encoder.inverse_transform([pred])[0]
    return label, prob