import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
import pickle

# Load data
df = pd.read_csv("Admission_Predict_Ver1.1.csv")
df.drop(columns=['Serial No.'], inplace=True)

# Define features (x) and target (y)
x = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create the model
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=7))  # Increase neurons
model.add(Dense(8, activation='relu'))  # Add more layers
model.add(Dense(1, activation='linear'))


# Compile and train the model
model.compile(loss='mean_squared_error', optimizer='Adam')
history = model.fit(x_train_scaled, y_train, epochs=100, validation_split=0.2)

# Save the model
model.save("admission_model.h5")

# Save the scaler for future use (so the user can use it for predictions)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Make predictions and evaluate
y_pred = model.predict(x_test_scaled)
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
