import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load dataset
csv_file_path = "Data/crop_yield_dataset.csv"
try:
    df = pd.read_csv(csv_file_path)
    logger.info("✅ Dataset loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load dataset: {e}")
    exit()

# Rename columns for consistency
df.rename(columns={"Soil_pH": "Soil pH"}, inplace=True)

# Encode categorical variables
crop_encoder = LabelEncoder()
region_encoder = LabelEncoder()

df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Region"] = region_encoder.fit_transform(df["Region"])
logger.info("✅ Encoding completed for Crop & Region")

# Add New Features
df["Soil Quality Index"] = (df["Soil pH"] * df["Nitrogen"]) / 10
df["Temp-Humidity Ratio"] = df["Temperature"] / df["Humidity"]
df["Rainfall-Adjusted Nutrients"] = df["Rainfall"] / (df["Nitrogen"] + df["Phosphorus"] + df["Potassium"] + 1)
logger.info("✅ New features added successfully")

# Prepare input and output variables
X = df.drop(columns=["Yield"])
y = df["Yield"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape for LSTM (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(1, X_train.shape[2])),
    Dropout(0.2),
    LSTM(30, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(20, activation='relu'),
    Dense(1)
])

# Compile Model
model.compile(optimizer='adam', loss='mse')

# Train Model
logger.info("🚀 Training LSTM Model...")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
logger.info("✅ LSTM Model Training Completed")

# Save Model
model.save("MODEL/lstm_model.h5")
with open("MODEL/scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

logger.info("✅ Model and Scaler saved successfully")
