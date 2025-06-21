import pandas as pd
import pickle

# Load models & preprocessing tools
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("models/target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)

with open("models/voting_classifier.pkl", "rb") as f:
    model = pickle.load(f)

def predict_accident_severity(form_data):
    df = pd.DataFrame([form_data])
    df["Latitude"] = float(df["Latitude"])
    df["Longitude"] = float(df["Longitude"])
    df["Number_of_Casualties"] = int(df["Number_of_Casualties"])
    df["Number_of_Vehicles"] = int(df["Number_of_Vehicles"])
    df["Speed_limit"] = int(df["Speed_limit"])
    df["Hour"] = int(str(df["Time"].iloc[0]).split(":")[0])

    # Fill missing
    df["Carriageway_Hazards"].fillna("No Hazards", inplace=True)
    df["Road_Surface_Conditions"].fillna("Unknown", inplace=True)
    df["Road_Type"].fillna("Unknown", inplace=True)
    df["Time"].fillna("Unknown", inplace=True)
    df["Weather_Conditions"].fillna("Unknown", inplace=True)

    # Encode
    for col in label_encoders:
        le = label_encoders[col]
        if df[col].iloc[0] in le.classes_:
            df[col] = le.transform(df[col])
        else:
            df[col] = le.transform([le.classes_[0]])
    df = df.drop(columns=["Time"])

    # Scale and predict
    scaled = scaler.transform(df)
    pred = model.predict(scaled)
    label = target_encoder.inverse_transform(pred)
    return label[0], form_data
