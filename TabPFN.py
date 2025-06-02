# !pip install yfinance pandas numpy tabpfn


import yfinance as yf
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import warnings

warnings.filterwarnings("ignore")

# Ticker list
tickers = ['AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'NBIX', 'ALNY']

# Define window for future prediction (e.g., 5 days)
future_window = 5
stock_data = {}

# Feature engineering + target creation function
def prepare_features(df):
    df = df.copy()
    df['5d_return'] = df['Close'].pct_change(5)
    df['10d_return'] = df['Close'].pct_change(10)
    df['volatility_10d'] = df['Close'].rolling(10).std()
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()
    # Corrected RSI calculation to avoid potential division by zero and use rolling mean/std of changes
    price_change = df['Close'].pct_change()
    gain = price_change.mask(price_change < 0, 0)
    loss = -price_change.mask(price_change > 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['future_return'] = df['Close'].shift(-future_window) / df['Close'] - 1
    df['Target'] = (df['future_return'] > 0).astype(int)
    df = df.dropna()
    return df

# Combine all tickers into one dataset
all_data = []

for ticker in tickers:
    print(f"Downloading and processing {ticker}")
    # Use start and end dates for consistent data periods if '10y' is not precise enough
    data = yf.Ticker(ticker).history(period="10y")
    df = prepare_features(data)
    df['Ticker'] = ticker  # Optional: keep ticker info
    all_data.append(df)

# Combine and shuffle
full_df = pd.concat(all_data)

features = ['5d_return', '10d_return', 'volatility_10d', 'ma_10', 'ma_50', 'rsi']
# Ensure only relevant columns are kept before sampling
full_df = full_df[features + ['Target']]

# ⚠️ TabPFN works best on small sets (≤1000), so sample from the combined DataFrame
# Sample directly from the full_df to keep X and y aligned
sampled_df = full_df.sample(n=1000, random_state=42)

X_sample = sampled_df[features]
y_sample = sampled_df['Target']

# Train/test split using the sampled data
X_train, X_test, y_train, y_test = train_test_split(X_sample.values, y_sample.values, test_size=0.2, random_state=42)

# Train TabPFN
# Check if GPU is available and use it, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clf = TabPFNClassifier(device=device)

# TabPFN requires inputs to be numpy arrays with float32 dtype
clf.fit(X_train.astype(np.float32), y_train)

# Evaluate
y_pred = clf.predict(X_test.astype(np.float32))
print("\nClassification Report:\n", classification_report(y_test, y_pred))









# Predict the next 5-day movement for each ticker
for ticker_symbol in tickers:
    print(f"\nPredicting for {ticker_symbol}")
    # Access the ticker data directly from the yfinance library
    current_ticker = yf.Ticker(ticker_symbol)
    latest_df = current_ticker.history(period="10y") # Download the latest data for the ticker

    latest_processed = prepare_features(latest_df)

    # Check if processed data is not empty before trying to predict
    if not latest_processed.empty:
        # Get the most recent row (latest features)
        # Ensure features exist in the processed dataframe before selecting
        latest_row = latest_processed[features].iloc[-1:].values
        # Ensure the input data for prediction is float32, as required by TabPFN
        prediction = clf.predict(latest_row.astype(np.float32))

        print(f"TabPFN prediction for {ticker_symbol}'s next 5-day move: {'UP' if prediction[0] == 1 else 'DOWN'}")
    else:
        print(f"Could not process enough data for {ticker_symbol} to make a prediction.")









# Add also Random Forest evaluation

print("\n🔮 Random Forest \n")

# --- RANDOM FOREST CLASSIFIER ---
clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
clf.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# # --- CONFUSION MATRIX ---
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

# # 📈 Optional: Plot Feature Importances
# importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
# importances.plot(kind='bar', title='Feature Importance (Random Forest)')
# plt.ylabel("Importance")
# plt.show()


print("\n🔮 Predictions for Next 5-Day Movement:\n")

all_data = []

for ticker in tickers:
    print(f"Downloading and processing {ticker}...")
    data = yf.Ticker(ticker).history(period="10y")
    df = prepare_features(data)
    if not df.empty:
        df['Ticker'] = ticker  # Make sure this is after dropping NaNs
        all_data.append(df)

# Combine all data
full_df = pd.concat(all_data)


for ticker in tickers:
    df = full_df[full_df['Ticker'] == ticker]
    latest_row = df[features].iloc[-1:].copy()
    prediction = clf.predict(latest_row)[0]
    prob = clf.predict_proba(latest_row)[0][prediction]
    direction = "UP 📈" if prediction == 1 else "DOWN 📉"
    print(f"{ticker}: {direction} (confidence: {prob:.2f})")















