import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
import warnings

warnings.filterwarnings("ignore")

# Generate synthetic financial dataset
def generate_synthetic_data(n=1000):
    np.random.seed(42)
    date_range = pd.date_range(start='1/1/2020', periods=n, freq='D')
    price = np.cumsum(np.random.randn(n)) + 100
    return pd.DataFrame({'Date': date_range, 'Price': price})

data = generate_synthetic_data()
data.set_index('Date', inplace=True)

# Feature Engineering
data['Returns'] = data['Price'].pct_change().fillna(0)
data['Lag_1'] = data['Price'].shift(1).fillna(method='bfill')

def evaluate_model(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }

# Train-Test Split
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Cross-Validation Setup
tscv = TimeSeriesSplit(n_splits=5)

# Hyperparameter tuning for RandomForest using Bayesian Optimization
def rf_bo(n_estimators, max_depth):
    rf = RandomForestRegressor(n_estimators=int(n_estimators), max_depth=int(max_depth), random_state=42)
    rf.fit(train[['Lag_1']], train['Price'])
    pred = rf.predict(test[['Lag_1']])
    return -mean_squared_error(test['Price'], pred)

rf_optimizer = BayesianOptimization(
    f=rf_bo,
    pbounds={'n_estimators': (50, 200), 'max_depth': (5, 20)},
    random_state=42,
    verbose=0
)
rf_optimizer.maximize(init_points=3, n_iter=5)

best_rf_params = rf_optimizer.max['params']
rf = RandomForestRegressor(n_estimators=int(best_rf_params['n_estimators']), max_depth=int(best_rf_params['max_depth']), random_state=42)
rf.fit(train[['Lag_1']], train['Price'])
rf_pred = rf.predict(test[['Lag_1']])
rf_metrics = evaluate_model(test['Price'], rf_pred)

# ARIMA Model
arima = ARIMA(train['Price'], order=(5,1,0)).fit()
arima_pred = arima.forecast(len(test))
arima_metrics = evaluate_model(test['Price'], arima_pred)

# LSTM Model
def build_lstm():
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(1,1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

X_train_lstm = np.array(train[['Lag_1']]).reshape(-1, 1, 1)
y_train_lstm = np.array(train['Price'])
X_test_lstm = np.array(test[['Lag_1']]).reshape(-1, 1, 1)

lstm = build_lstm()
lstm.fit(X_train_lstm, y_train_lstm, epochs=10, verbose=0)
lstm_pred = lstm.predict(X_test_lstm).flatten()
lstm_metrics = evaluate_model(test['Price'], lstm_pred)

# Stacking Ensemble
def stack_models():
    X_train, y_train = train[['Lag_1']], train['Price']
    X_test = test[['Lag_1']]

    rf_pred_train = rf.predict(X_train)
    arima_pred_train = arima.predict(start=0, end=len(X_train)-1)
    lstm_pred_train = lstm.predict(X_train_lstm).flatten()

    meta_train = np.column_stack((rf_pred_train, arima_pred_train, lstm_pred_train))
    meta_test = np.column_stack((rf_pred, arima_pred, lstm_pred))

    meta_model = XGBRegressor()
    meta_model.fit(meta_train, y_train)
    ensemble_pred = meta_model.predict(meta_test)

    return ensemble_pred, evaluate_model(test['Price'], ensemble_pred)

ensemble_pred, ensemble_metrics = stack_models()

# Plotting Function
def plot_predictions(true, preds, title):
    plt.figure(figsize=(12, 5))
    plt.plot(true.index, true.values, label='Actual Price', linewidth=2)
    plt.plot(true.index, preds, label='Predicted Price', linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot for each model
plot_predictions(test['Price'], rf_pred, "Random Forest - Actual vs Predicted")
plot_predictions(test['Price'], arima_pred, "ARIMA - Actual vs Predicted")
plot_predictions(test['Price'], lstm_pred, "LSTM - Actual vs Predicted")
plot_predictions(test['Price'], ensemble_pred, "Stacked Ensemble - Actual vs Predicted")

# Print Results
print("Random Forest:", rf_metrics)
print("ARIMA:", arima_metrics)
print("LSTM:", lstm_metrics)
print("Ensemble Model:", ensemble_metrics)