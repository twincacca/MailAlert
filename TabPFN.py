# !pip install yfinance pandas numpy tabpfn matplotlib seaborn


import yfinance as yf
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings




warnings.filterwarnings("ignore")

# Ticker list
# tickers = ['AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'NBIX', 'ALNY']
tickers = ['AMGN', 'GILD', 'BIIB', 'REGN']

# Define window for future prediction (e.g., 5 days)
future_window = 5
stock_data = {}

# Feature engineering + target creation function
def prepare_features(df):
    df = df.copy()
    df['5d_return'] = df['Close'].pct_change(5)     # Return over past 5 and 10 days
    df['10d_return'] = df['Close'].pct_change(10)
    df['volatility_10d'] = df['Close'].rolling(10).std() # Volatility (standard deviation) over 10 days
    df['ma_10'] = df['Close'].rolling(10).mean()  # Moving averages (10-day and 50-day)
    df['ma_50'] = df['Close'].rolling(50).mean()
    # Corrected RSI (relative strength index, simple version) calculation to avoid potential division by zero and use rolling mean/std of changes
    price_change = df['Close'].pct_change()
    gain = price_change.mask(price_change < 0, 0)
    loss = -price_change.mask(price_change > 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['future_return'] = df['Close'].shift(-future_window) / df['Close'] - 1
    df['Target'] = (df['future_return'] > 0).astype(int) # Target = 1 if stock closes higher in 5 days, else 0
    df = df.dropna()
    return df

# Combine all tickers into one dataset
all_data = []

for ticker in tickers:
    print(f"Downloading and processing {ticker}")
    # Use start and end dates for consistent data periods if '10y' is not precise enough
    data = yf.Ticker(ticker).history(period="10y") # Downloads up to 10 years of data for each ticker
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

# Train/test split using the sampled data (80% training and 20% testing sets)
X_train, X_test, y_train, y_test = train_test_split(X_sample.values, y_sample.values, test_size=0.2, random_state=42)


# Train TabPFN
# Check if GPU is available and use it, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clf_TabPFN = TabPFNClassifier(device=device)

# TabPFN requires inputs to be numpy arrays with float32 dtype
clf_TabPFN.fit(X_train.astype(np.float32), y_train)

# Evaluate
y_pred = clf_TabPFN.predict(X_test.astype(np.float32))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




# Classification Report:
#                precision    recall  f1-score   support
#
#            0       0.25      0.01      0.02       107     # Stock goes down or stays flat
#            1       0.46      0.97      0.62        93     # Stock goes up
#
#     accuracy                           0.46       200     # % of all predictions that were correct
#    macro avg       0.35      0.49      0.32       200     # Unweighted average of precision, recall, F1 across classes. Treats each class equally (good if you care about both directions equally)
# weighted avg       0.35      0.46      0.30       200     # Weighted by support (i.e., class frequency). More representative of total performance if the dataset is imbalanced
#
#
#
#
# Metric	Meaning
# Precision	Of all the times the model said "class X", how many were actually X?    TP/(TP+FP)  high if low FP
# Recall	Of all actual class X examples, how many did the model catch?           TP/(TP+FN)  high if low FN
# F1-score	Harmonic mean of precision and recall. High only if both are high.
# Support	Number of true examples in the test set for that class



# # A)
# #
# # Predict the next 5-day movement for AMGN
# # latest_df = stock_data['AMGN'].copy() # Original line causing error
# # Access the AMGN data directly from the yfinance library again
# amgn_ticker = yf.Ticker("AMGN")
# latest_df = amgn_ticker.history(period="10y") # Download the latest data for AMGN
#
# latest_processed = prepare_features(latest_df)
#
# # Get the most recent row (latest features)
# latest_row = latest_processed[features].iloc[-1:].values
# # Ensure the input data for prediction is float32, as required by TabPFN
# prediction = clf_TabPFN.predict(latest_row.astype(np.float32))
#
# print(f"\nTabPFN prediction for AMGN's next 5-day move: {'UP' if prediction[0] == 1 else 'DOWN'}")
#








# B)
#
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
        prediction = clf_TabPFN.predict(latest_row.astype(np.float32))

        print(f"TabPFN prediction for {ticker_symbol}'s next 5-day move: {'UP' if prediction[0] == 1 else 'DOWN'}")
    else:
        print(f"Could not process enough data for {ticker_symbol} to make a prediction.")















# Add also Random Forest evaluation

print("\n🔮 Random Forest \n")

# --- RANDOM FOREST CLASSIFIER ---
clf_RF = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
clf_RF.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = clf_RF.predict(X_test)
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
# importances = pd.Series(clf_RF.feature_importances_, index=features).sort_values(ascending=False)
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
    prediction = clf_RF.predict(latest_row)[0]
    prob = clf_RF.predict_proba(latest_row)[0][prediction]
    direction = "UP 📈" if prediction == 1 else "DOWN 📉"
    print(f"{ticker}: {direction} (confidence: {prob:.2f})")


















# Combine evaluation and output in same format for both TabPFN and RF

def evaluate_model(name, model, X_test, y_test, features, full_df, tickers):
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    print(f"\n🔍 Evaluation Report for {name}:\n")

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # --- Ticker-Level Prediction ---
    print(f"\n🔮 Next 5-Day Predictions by {name}:\n")
    predictions = []
    for ticker in tickers:
        df = full_df[full_df['Ticker'] == ticker]
        if df.empty:
            continue
        latest = df[features].iloc[-1:].copy()
        pred = model.predict(latest)[0]
        prob = model.predict_proba(latest)[0][pred] if hasattr(model, "predict_proba") else 1.0
        predictions.append({
            'Ticker': ticker,
            'Prediction': 'Up' if pred == 1 else 'Down',
            'Confidence': round(prob, 3)
        })

    pred_df = pd.DataFrame(predictions)
    print(pred_df.to_string(index=False))
    return pred_df





# Evaluate Random Forest
evaluate_model("Random Forest", clf_RF, X_test, y_test, features, full_df, tickers)

# Evaluate TabPFN (assuming `tabpfn_model` and preprocessed data)
evaluate_model("TabPFN", clf_TabPFN, X_test, y_test, features, full_df, tickers)
