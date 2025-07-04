# !pip install yfinance pandas numpy tabpfn matplotlib seaborn

import yfinance as yf
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import torch
import warnings

warnings.filterwarnings("ignore")

# List of tickers to analyze
#tickers = ['AMGN', 'GILD', 'BIIB', 'REGN']
tickers = ['SOPH','QGEN','ILMN','ONT.L','PACB','A','TMO','PKI','TXG']
future_window = 50
features = ['5d_return', '10d_return', 'volatility_10d', 'ma_10', 'ma_50', 'rsi']

# --- Feature engineering function ---
def prepare_features(df):
    df = df.copy()
    df['5d_return'] = df['Close'].pct_change(5)
    df['10d_return'] = df['Close'].pct_change(10)
    df['volatility_10d'] = df['Close'].rolling(10).std()
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()
    price_change = df['Close'].pct_change()
    gain = price_change.mask(price_change < 0, 0)
    loss = -price_change.mask(price_change > 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['future_return'] = df['Close'].shift(-future_window) / df['Close'] - 1
    df['Target'] = (df['future_return'] > 0).astype(int)
    return df.dropna()

# --- Load and process data ---
all_data = []
for ticker in tickers:
    data = yf.Ticker(ticker).history(period="10y")
    df = prepare_features(data)
    df['Ticker'] = ticker
    all_data.append(df)

full_df = pd.concat(all_data)
full_df = full_df[features + ['Target']].dropna()

# --- Sample and split for model training ---
sampled_df = full_df.sample(n=1000, random_state=42)
X = sampled_df[features].values.astype(np.float32)
y = sampled_df['Target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train and evaluate TabPFN ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clf_tabpfn = TabPFNClassifier(device=device)
clf_tabpfn.fit(X_train, y_train)
y_pred_tabpfn = clf_tabpfn.predict(X_test)

print("\n📊 Classification Report: TabPFN\n")
print(classification_report(y_test, y_pred_tabpfn))

# --- Train and evaluate Random Forest ---
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

print("\n📊 Classification Report: Random Forest\n")
print(classification_report(y_test, y_pred_rf))





print("\n📈 Recent 5-Day Trend Predictions vs. Actuals:\n")

results = []

# Look at last N rows of each ticker to simulate real recent forecasts
N = 5  # how many recent windows to test
for ticker in tickers:
    df = yf.Ticker(ticker).history(period="6mo")  # recent 6 months
    df = prepare_features(df)
    df['Date'] = df.index

    if len(df) >= N:
        recent_df = df.iloc[-N:]
        for _, row in recent_df.iterrows():
            x_feat = row[features].values.astype(np.float32).reshape(1, -1)

            pred_tabpfn = clf_tabpfn.predict(x_feat)[0]
            pred_rf = clf_rf.predict(x_feat)[0]
            actual = "Up" if row["Target"] == 1 else "Down"

            results.append({
                "Ticker": ticker,
                "Date": row["Date"].strftime('%Y-%m-%d'),
                "TabPFN": "Up" if pred_tabpfn == 1 else "Down",
                "RandomForest": "Up" if pred_rf == 1 else "Down",
                "Actual": actual
            })

# Display the result
df_result = pd.DataFrame(results)
print(df_result.to_string(index=False))




print("just check")
sop_data = yf.Ticker("SOPH").history(period="1mo")
print(sop_data.tail(15))  # Compare this with what Google says

