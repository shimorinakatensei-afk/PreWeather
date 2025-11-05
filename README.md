# PreWeather
A Python app that predicts weather using AI.
# weather_ai.py
"""
AI Weather Predictor
This script predicts the next day's temperature using historical weather data and a simple AI model.
The model is based on Linear Regression and can be extended to more advanced ML models.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# =======================
# 1. Sample Historical Data
# =======================
# For demonstration, we use a small dataset. You can replace this with real historical weather data.
data = {
    "date": ["2025-11-01", "2025-11-02", "2025-11-03", "2025-11-04", "2025-11-05", "2025-11-06", "2025-11-07"],
    "temp_day": [18, 20, 19, 21, 22, 20, 23],
    "humidity": [50, 55, 52, 48, 50, 53, 49],
    "wind_speed": [3, 2, 4, 3, 2, 3, 4],
}

# =======================
# 2. Convert data to DataFrame
# =======================
df = pd.DataFrame(data)

# =======================
# 3. Feature Preparation
# =======================
# Using previous day's temperature as the main feature to predict the next day's temperature
X = df['temp_day'].shift(1).dropna().values.reshape(-1, 1)  # predictor
y = df['temp_day'][1:].values  # target

# =======================
# 4. Split data into training and testing
# =======================
# Using 80% of data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# 5. Initialize and train the model
# =======================
# Linear Regression is used for simplicity
model = LinearRegression()
model.fit(X_train, y_train)

# =======================
# 6. Evaluate the model
# =======================
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy (R^2 Score): {accuracy:.2f}")

# =======================
# 7. User input for prediction
# =======================
# Ask the user to enter the temperature of the last known day
while True:
    try:
        last_temp = float(input("Enter the last day's temperature (°C): "))
        break
    except ValueError:
        print("Invalid input. Please enter a number.")

# =======================
# 8. Predict the next day's temperature
# =======================
predicted_temp = model.predict([[last_temp]])
print(f"Predicted temperature for tomorrow: {predicted_temp[0]:.2f} °C")

# =======================
# 9. Optional: Future enhancements
# =======================
# - Use more features like humidity, wind_speed, precipitation, etc.
# - Use advanced models like Random Forest, XGBoost, or LSTM for time-series prediction
# - Load real historical data from CSV or API like OpenWeatherMap
