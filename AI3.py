from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow

# Download and prepare data
stock_data = yf.download('AAPL', start='2022-01-01', end='2024-04-18')
scalar = MinMaxScaler(feature_range=(0, 1))
scaled_data = scalar.fit_transform(stock_data['Close'].values.reshape(-1,1))

# Helper function to create dataset 
def create_dataset(data, time_step):
    x, y = [], []
    for i in range(len(data) - time_step - 1):
        x.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)


# Define time step
time_step = 100
x, y = create_dataset(scaled_data, time_step)
train_size = 0.8

# Split data into training and testing
x_train, x_test = x[:int(x.shape[0]*train_size)], x[int(x.shape[0]*train_size):]
y_train, y_test = y[:int(y.shape[0]*train_size)], y[int(y.shape[0]*train_size):]

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(64),
    Dense(64),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=64)
test_loss = model.evaluate(x_test, y_test)

# Predict using the model
predictions = model.predict(x_test)
predictions = scalar.inverse_transform(predictions)

# Prepare the plot for original and predicted data
original_data = stock_data['Close'].values
predicted_data = np.empty_like(original_data)
predicted_data[:] = np.nan
predicted_data[-len(predictions):] = predictions.reshape(-1)

# Predict the last 90 days and extend them into the future
new_predictions = model.predict(x_test[-90:])
new_predictions = scalar.inverse_transform(new_predictions)

# Extend predictions into the future
last_date = stock_data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=90)  # Create 90 future dates

# Print new predictions with their corresponding future dates
for date, value in zip(future_dates, new_predictions.flatten()):
    print(f"Date: {date.strftime('%Y-%m-%d')}, Predicted Close: {value:.2f}")

# Create plots
plt.figure(figsize=(14, 7))
plt.plot(stock_data.index, original_data, label='Actual Close Prices')
plt.plot(np.append(stock_data.index, future_dates), np.append(predicted_data, new_predictions), label='Predicted Close Prices', linestyle='--')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Time (Days)')
plt.ylabel('Stock Price')
plt.legend()
plt.show()