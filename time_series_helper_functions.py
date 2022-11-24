import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import pandas_datareader as pdr
import yfinance as yf
import talib as ta
from talib import MA_Type
from statsmodels.tsa.stattools import adfuller
import matplotlib as mpl
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import random
from random import choice
import os
import pmdarima as pm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# function of obtaining data
def get_data(symbol, data_length):
    now = dt.datetime.now()
    start = now.replace(now.year - data_length).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")
    
    yf.pdr_override()
    df = pdr.data.get_data_yahoo(symbol, start=start, end=end)
    return df

def preprocess_data(df, data_for, SMA=True, sma_s=5, sma_m=10, sma_l=20,
              MACD=True, short_span=8, long_span=17, macd_span=9, 
              RSI=True, timeperiod=14, SOI=True, slowk_period=3, fastk_period=14, 
              Bband=True):
    """
    data_for: pick one of them ["Adj Close", "Open", "high", "Low"], then add all the features I chose to add. 
    """
    # Fill the zero value on open price to previous closing price
    for i, open_price in enumerate(df["Open"]):
        if open_price == 0.0:
            df["Open"].iloc[i] = df["Adj Close"].iloc[i-1]
            
    df["DiffClose$"] = df["Adj Close"].diff()
    df["DiffOpen$"] = df["Open"].diff()
    df["DiffHigh$"] = df["High"].diff()
    df["DiffLow$"] = df["Low"].diff()
          
    closeps = []
    openps = []
    highs = []
    lows = []
    for i in range(len(df)):
        closep = df["DiffClose$"].iloc[i] / df["Adj Close"].iloc[i-1]
        closeps.append(closep)
        
    for i in range(len(df)):
        openp = df["DiffOpen$"].iloc[i] / df["Open"].iloc[i-1]
        openps.append(openp)
        
    for i in range(len(df)):
        highp = df["DiffHigh$"].iloc[i] / df["High"].iloc[i-1]
        highs.append(highp)
        
    for i in range(len(df)):
        lowp = df["DiffLow$"].iloc[i] / df["Low"].iloc[i-1]
        lows.append(lowp)
        
    df["Closep"] = closeps
    df["Openp"] = openps
    df["Highp"] = highs
    df["Lowp"] = lows
    
    if data_for == "Adj Close":
        percent = "Closep"
    elif data_for == "Open":
        percent = "Openp"
    elif data_for == "High":
        percent = "Highp"
    else:
        percent = "Lowp"
    
    # SMA
    if SMA:
        df["SMA_s"] = df[data_for].rolling(sma_s).mean()
        df["SMA_m"] = df[data_for].rolling(sma_m).mean()
        df["SMA_l"] = df[data_for].rolling(sma_l).mean()
        
        df["smap_s"] = df[percent].rolling(sma_s).mean()
        df["smap_m"] = df[percent].rolling(sma_m).mean()
        df["smap_l"] = df[percent].rolling(sma_l).mean()
        
        
    # MACD
    if MACD:
        # Calculate the short term exponential moving average
        shortEMA = df[data_for].ewm(span=short_span, adjust=False).mean()
        # Calculate the long term exponential moving average
        longEMA = df[data_for].ewm(span=long_span, adjust=False).mean()
        # Calculate MACD line
        MACD = shortEMA - longEMA
        # calculate the signal line
        signal = MACD.ewm(span=macd_span, adjust=False).mean()
        # Add to the DataFrame
        df["MACD"] = MACD
        df["Signal Line"] = signal
        
        # Calculate the short term exponential moving average with stationary data
        shortEMA = df[percent].ewm(span=short_span, adjust=False).mean()
        # Calculate the long term exponential moving average
        longEMA = df[percent].ewm(span=long_span, adjust=False).mean()
        # Calculate MACD line
        MACD = shortEMA - longEMA
        # calculate the signal line
        signal = MACD.ewm(span=macd_span, adjust=False).mean()
        # Add to the DataFrame
        df["macdp"] = MACD
        df["signal_linep"] = signal
        
    # RSI
    if RSI:
        df["RSI"] = ta.RSI(df[data_for], timeperiod=timeperiod)
        
        df["rsip"] = ta.RSI(df[percent], timeperiod=timeperiod)
        
    # Stochastic
    if SOI:
        df["SlowK"], df["SlowD"] = ta.STOCH(high=df["High"],
                                   low=df["Low"],
                                   close=df["Adj Close"], 
                                   slowk_period=slowk_period,
                                   fastk_period=fastk_period)
        
        df["slowKp"], df["slowDp"] = ta.STOCH(high=df["Highp"],
                                   low=df["Lowp"],
                                   close=df["Closep"], 
                                   slowk_period=slowk_period,
                                   fastk_period=fastk_period)
        
    # Bollinger Band
    if Bband:
        df["Bband_upper"], df["Bband_mid"], df["Bband_lower"] = ta.BBANDS(df[data_for], matype=MA_Type.T3)
        
        df["BBuperp"], df["BBmidp"], df["BBlowp"] = ta.BBANDS(df[percent], matype=MA_Type.T3)
        
    df.dropna(inplace=True)
        
    return df

from statsmodels.tsa.stattools import kpss
def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print(f'Critical Values:')
    for key, value in critical_values.items():
        print(f'   {key}: {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
    
from statsmodels.tsa.stattools import adfuller

def adf_test(series):
    result = adfuller(series)
    adf = result[0]
    p = result[1]
    usedlag = result[2]
    nobs = result[3]
    cri_val_1 = result[4]["1%"]
    cri_val_5 = result[4]["5%"]
    cri_val_10 = result[4]["10%"]
    icbest = result[5]
    print(f"Test Statistic: {adf}\np-value: {p}\n#Lags Used: {usedlag}\nNumber of Observations: {nobs}\nCritical Value (1%): {cri_val_1}\nCritical Value (5%): {cri_val_5}\nCritical Value (10%): {cri_val_10}\nicbest: {icbest}\n")
    print(f'Result: The series is {"not " if p > 0.05 else ""}stationary')

# Create function to label windowed data
def get_labelled_windows(x, horizon):
  """
  Create labels for windowed dataset.

  e.g. if horizon=1
  Input: [0, 1, 2, 3, 4, 5, 6, 7] -> Output: ([0, 1, 2, 3, 4, 5, 6], [7])
  """
  return x[:, :-horizon], x[:, -horizon:]

# Create function to view NumPy arrays as windows
def make_windows(x, window_size, horizon):
  """
  Turns a 1D array into a 2D array of sequential labelled windows of windows of window_size with horizon size labels.
  
  Return:
      Windows, labels : 
  """
  #1. Create a window of specific window_size (add the horizon on the end for labelling later)
  window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)

  #2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
  window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis=0).T # Create 2D array of windows of size window_size

  # print(f"Window indexes:\n {window_indexes, window_indexes.shape}")

  #3. Index on the target array (a tim series) with 2D array of multiple window steps
  windowed_array = x[window_indexes]
  # print(f"\nWindowed array:\n{windowed_array}")
  
  #4. Get the labelled windows
  windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
  return windows, labels 

# Make the train/splits on windows/horizen
def make_train_test_split(X, y, test_split=0.2):
  """
  Splits matching pairs of windows and labels into train and test splits.
  
  X_train, X_test, y_train, y_test: 
  """
  split_size = int(len(X) * (1 - test_split)) 
  X_train = X[:split_size]
  y_train = y[:split_size]
  X_test = X[split_size:]
  y_test = y[split_size:]
  return X_train, X_test, y_train, y_test


# Create the function to split the train and test data
def train_test_split_time_series(timesteps, prices, split_size=0.2):
  """
  Split train and test dataset for time series. 

  Parameters
  ------------
  split_size : (Float number) size of train dataset. Default is 20%. 
  timesteps : array of timestep value. Index of the dataset.
  prices : array of price value. The values to predict. 

  Return
  ------------
  X_train, y_train, X_test, y_test : tuple 
  """
  # Create train and test splits the right way for time series data
  split_size = int(split_size * len(prices)) # 80% train, 20% test - you can change these values

  # Create train data splits (everything before the split)
  X_train, y_train = timesteps[:split_size], prices[:split_size]

  # Create test data splits (everything beyond the split)
  X_test, y_test = timesteps[split_size:], prices[split_size:]

  return X_train, y_train, X_test, y_test

# Create a functon to implement a ModelCheckpoint callback with a specific filename
def create_model_checkpoint(model_name, save_path):
  return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                            verbose=0,
                                            save_best_only=True, 
                                           )

def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch
    
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error [$]")
    plt.plot(hist["epoch"], hist["mae"],
            label="Train Error")
    plt.plot(hist["epoch"], hist["val_mae"],
            label="Val Error")
    plt.legend()
#     plt.ylim()
    
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss [$]")
    plt.plot(hist["epoch"], hist["loss"],
            label="Train loss")
    plt.plot(hist["epoch"], hist["val_loss"],
            label="Val loss")
    plt.legend()
    
def plot_loss_curves(history):
    """
    Returns Separate loss curves for training and validation metrics.
    
    Arg: 
        history: Tensorflow History object.
        
    Returns: 
        Plots of training/validation loss and accuracy metrics.
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    
    epochs = range(len(history.history["loss"]))
    
    # Plot loss
    plt.figure(figsize=(14, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend();
    
    return ax

# Create a function to plot time series data
def plot_time_series(symbol, timesteps, values, format=".", start=0, end=None, label=None):
  """
  Plots timesteps (a series of points in time) against values (a series of values across timesteps).

  Parameters
  -----------
  symbol : what time series are you looking
  timesteps : array of timesteps values
  values : array of values across time
  format : style of plot, default "."
  start : where to start the plot (setting a value will index from start of timesteps & values)
  end : where to end the plot (similar to start but for the end)
  label : label to show on plot about values
  """
  # Plot the series
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel("time", fontsize=14)
  plt.ylabel(f"{symbol} Price", fontsize=14)
  if label:
    plt.legend(fontsize=14) # make label bigger
  plt.grid(True)

# MASE implementation
def mean_absolute_scaled_error(y_true, y_pred):
  """
  Implement MASE (mean absolute scaled error) assuming no seasonlity data.
  """
  mae = tf.reduce_mean(tf.abs(y_true - y_pred))

  # Find MAE of naive forcast (no seasonality)
  mean_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))

  return mae / mean_naive_no_season

# Function to predict the model result
def make_preds(model, input_data):
  """
  Uses model to make predictions input_data.
  """
  forecast = model.predict(input_data)
  return tf.squeeze(forecast) # return 1D array of predictions

# Create the function to take in model predictions and truth values and return evaluation metrics
def evaluate_preds(y_true, y_pred):
  # Make sure float32 datatype for metric calculations
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  # Calculate various evaluation metrics
  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
  mase = mean_absolute_scaled_error(y_true, y_pred)

  # Account for different sized metrics (for longer horizons, we want to reduce metics to single value)
  if mae.ndim > 0:
    mae = tf.reduce_mean(mae)
    mse = tf.reduce_mean(mse)
    rmse = tf.reduce_mean(rmse)
    mape = tf.reduce_mean(mape)
    mase = tf.reduce_mean(mase)

  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}

def price_inverse(original_df, test_df, y_preds, predict_price):
    """
    From the percent change prediction, inverse to the original price. 
    
    original_df: Data Frame before any preprocess. 
    y_preds: Predict result.
    predict_price: What price are you predicted.["Open", "Close", "High", "Low"]
    
    return: dataframe with inversed price, inversed difference then drop 1st row. 
    """
    if predict_price == "Open":
        columns = ["Open", "DiffOpen$", "Openp"]
    elif predict_price == "Close":
        columns = ["Adj Close", "DiffClose$", "Closep"]
    elif predict_price == "High":
        columns = ["High", "DiffHigh$", "Highp"]
    else:
        columns = ["Low", "DiffLow$", "Lowp"]
    convert_df = original_df[columns].tail(len(test_df))
    add_df = original_df[columns][-(len(test_df))-1:-len(test_df)]
    convert_df = pd.concat([add_df, convert_df], ignore_index=False, sort=False)
    convert_df["PredResults"] = y_preds
    
    inverse_diff = []
    for i in range(len(convert_df)):
        inverse = convert_df["PredResults"].iloc[i] * convert_df["Adj Close"].iloc[i-1]
        inverse_diff.append(inverse)
    convert_df["inverseDiff"] = inverse_diff
    
    pred_price = []
    for i in range(len(convert_df)):
        pred = convert_df["inverseDiff"].iloc[i] + convert_df["Adj Close"].iloc[i-1]
        pred_price.append(pred)
    convert_df["PredictedPrice"] = pred_price
    
    return convert_df.iloc[1:, :]

def price_inverse_sma(original_df, train_df, test_df, y_preds, pred_column, 
                      predict_price, sma_length, model_name, data_used, forecast_length):
    """
    From the percent change prediction, inverse to the original price. 
    
    original_df: Data Frame before any preprocess. 
    train_df: Data used to train. This data is to calculate SMA by taking last sma_length data. 
    test_df: Data used to test. 
    y_preds: Predict result.
    pred_column: Which value predicted. This time which length of SMA predicted. The purpose is to 
                 add correct Series when you do inverse process.
    predict_price: What price are you predicted.["Open", "Close", "High", "Low"]
    sma_length: Length of SMA.
    model_name: Which model to get the forecast. 
    data_used: Which data you used. "Symbol" and "Date"
    forecast_length: How many days of forecast. 
    
    return: dataframe with inversed price, inversed difference then drop 1st row. 
    """
    # Get columns based on which prediction you did.
    if predict_price == "Open":
        columns = ["Open", "DiffOpen$", "Openp", pred_column]
    elif predict_price == "Close":
        columns = ["Adj Close", "DiffClose$", "Closep", pred_column]
    elif predict_price == "High":
        columns = ["High", "DiffHigh$", "Highp", pred_column]
    else:
        columns = ["Low", "DiffLow$", "Lowp", pred_column]
    # DataFrame for testing part
    convert_df = original_df[columns].tail(len(test_df))
    convert_df["Prediction"] = y_preds
    # Add end of train data to be able to inverse moving average
    add_df = original_df[columns][-(len(test_df)+(sma_length-1)): -len(test_df)]
    add_df["Prediction"] = add_df[columns[3]]
    # Concat aboves together
    convert_df = pd.concat([add_df, convert_df], ignore_index=False, sort=False)
    
    #print(convert_df.head())
    
    # Calculate predicted Closep by inversing SMA. 
    invs = []
    start = 0
    end = sma_length - 1
    for i in range(len(convert_df)):
        try:
            sma_sum = convert_df["Prediction"].iloc[end] * sma_length
            inv = sma_sum - sum(convert_df[columns[2]].values[start:end])
            invs.append(inv)
            start += 1
            end += 1
        except:
            i > len(test_df) - 1
    #print(invs)
    # Keep last day of training data to be able to get first day forecast
    convert_df = convert_df.tail(len(y_preds)+1)
    later_add = convert_df.head(1)
    later_add["Inversed_p"] = convert_df[columns[2]].iloc[0]
    # Split dataset for testing data length so I can add inveresed number I got previously
    convert_df = convert_df.tail(len(y_preds))
    convert_df["Inversed_p"] = invs
    # Concat last day of training data
    convert_df = pd.concat([later_add, convert_df], ignore_index=False, sort=False)
    #print(convert_df.head())
    
    inverse_diff = []
    for i in range(len(convert_df)):
        inverse = convert_df["Inversed_p"].iloc[i] * convert_df[columns[0]].iloc[i-1]
        inverse_diff.append(inverse)
    convert_df["inverseDiff"] = inverse_diff
    
    pred_price = []
    for i in range(len(convert_df)):
        pred = convert_df["inverseDiff"].iloc[i] + convert_df[columns[0]].iloc[i-1]
        pred_price.append(pred)
    convert_df["PredictedPrice"] = pred_price
    
    tomorrow_price = np.round(convert_df["PredictedPrice"].iloc[1], 2)
    
    convert_df["ActualDirection"] = "NaN"
    convert_df["PredictedDirection"] = "NaN"
    for i in range(len(convert_df)):
        if convert_df[columns[1]].iloc[i] > 0:
            convert_df["ActualDirection"].iloc[i] = "UP"
        elif convert_df[columns[1]].iloc[i] < 0:
            convert_df["ActualDirection"].iloc[i] = "DOWN"
        elif convert_df[columns[1]].iloc[i] == 0:
            convert_df["ActualDirection"].iloc[i] = "SAME"
        
    for i in range(len(convert_df)):
        if convert_df["inverseDiff"].iloc[i] > 0:
            convert_df["PredictedDirection"].iloc[i] = "UP"
        elif convert_df["inverseDiff"].iloc[i] < 0:
            convert_df["PredictedDirection"].iloc[i] = "DOWN"
        elif convert_df["inverseDiff"].iloc[i] == 0:
            convert_df["PredictedDirection"].iloc[i] = "SAME"    
            
    #print(convert_df.head())
    
    df_len = len(convert_df)
    unique, counts = np.unique(np.where(convert_df["ActualDirection"] == convert_df["PredictedDirection"]), return_counts=True)
    right_direc = sum(counts)
    accuracy = right_direc / df_len * 100
                                                              
    # Change columns order for better visualization
    cols = convert_df.columns.tolist()
    new_cols = [cols[0], cols[7], cols[1], cols[6], cols[2], cols[5], cols[3], cols[4], cols[8], cols[9]]
    convert_df = convert_df[new_cols]                                                         
                                                              
    
    print("\n")
    print(f"I am predicting {forecast_length} days forecast. ")
    print(f"And 1st day forecast is ${tomorrow_price}")
    print(f"I predicted SMA{str(sma_length)} on {predict_price} Dataset.")
    print(f"I used {data_used} data and predicted with {model_name} model. ")                        
    print(f"Direction Accuracy: {np.round(accuracy, 2)}%")
    
    return convert_df.iloc[1:, :]

def future_forecast(original_df, key, pred_val="smap_m", extra_val="smap_l", forecast_len=10, pred_sma=20):
    df_to_use = original_df[pred_val]
    train = df_to_use
    train_X = original_df[extra_val]
    
    # Train with ARIMAX model and predict forecast
    sxmodel = pm.auto_arima(train, exogenous=train_X.to_numpy().reshape(-1, 1), max_p=5, max_q=5, seasonal=False, 
                        trace=True, error_action='ignore', suppress_warnings=True)
    sxmodel.fit(train)
    forecast = sxmodel.predict(n_periods=forecast_len)
    forecast = pd.DataFrame(forecast, columns=['Prediction'])
    
    # Obtain columns name 
    if key == "Open":
        cols = ["Open", "DiffOpen$", "Openp", pred_val]
    elif key == "Close":
        cols = ["Adj Close", "DiffClose$", "Closep", pred_val]
    elif key == "High":
        cols = ["High", "DiffHigh$", "Highp", pred_val]
    else:
        cols = ["Low", "DiffLow$", "Lowp", pred_val]
        
    # Get previous price for later print and calculation
    prev_price = np.round(original_df[cols[0]][-1], 2)
        
    # create df to inverse SMA to actual price
    df = original_df[cols].tail(pred_sma-1)
    # Create list of future dates then make Datframe with forecast. Set Date as index.     
    dates = pd.date_range(datetime.today(), periods=forecast_len).strftime("%Y-%m-%d").tolist()
    forecast["Date"] = dates
    forecast["Date"] = pd.to_datetime(forecast['Date'])
    forecast.set_index("Date", inplace=True)
    # Concat created df and forecast
    df = pd.concat([df, forecast])
    
    start = 0
    end = pred_sma - 1
    for i in range(len(df)):
        try:
            sma_sum = df["Prediction"].iloc[end] * pred_sma
            inv = sma_sum - sum(df[cols[2]].values[start: end])
            df[cols[2]].iloc[end] = inv
            start += 1
            end += 1
        except:
            i > forecast_len-1
            
    # Inverse process
    df = df.fillna(0)
    df["inverseDiff"] = 0
    for i in range(len(df)):
        inverse = df[cols[2]].iloc[i] * df[cols[0]].iloc[i-1]
        df["inverseDiff"].iloc[i] = inverse
        if df[cols[0]].iloc[i] == 0:
            df[cols[0]].iloc[i] = df["inverseDiff"].iloc[i] + df[cols[0]].iloc[i-1]
    
    # Define outputs
    f_forecast = df[cols[0]][-forecast_len:]
    tomorrow_forecast = np.round(f_forecast[0], 2)
    change = np.round((tomorrow_forecast - prev_price) / prev_price * 100, 2)
    
    # Print 
    print(f"Next {forecast_len} days forecast for {key} price:\n{f_forecast}")
    print(f"Tomorrow's price is ${tomorrow_forecast} and changed {change}% from previous price ${prev_price}")
    
    return f_forecast, tomorrow_forecast, change, df

def evaluate_result(inversed_df, predicted_price="Close"):
    """
    Evaluate the result from the inversed price dataframe with actural stock price comapreson. 
    """
    if predicted_price == "Close":
        column = "Adj Close"
    elif predicted_price == "Open":
        column = "Open"
    elif predicted_price == "High":
        column = "High"
    else:
        column = "Low"
    result = evaluate_preds(inversed_df[column].to_numpy(), inversed_df["PredictedPrice"].to_numpy())
    return result

def plot_pred_actural_price(inverted_df, price_to_show=50, predicted_price="Close"):
    if predicted_price == "Close":
        column = "Adj Close"
    elif predicted_price == "Open":
        column = "Open"
    elif predicted_price == "High":
        column = "High"
    else:
        column = "Low"
    plt.figure(figsize=(8, 5))
    inverted_df[column].iloc[-price_to_show:].plot(label="Actural Price")
    inverted_df["PredictedPrice"].iloc[-price_to_show:].plot(label="Predicted Price")
    plt.ylabel("Price", fontsize=14)
    plt.title(f"Last {price_to_show} Days of Actural Price vs Predicted Price", fontsize=16)
    plt.legend()
    plt.show();
    
def print_shape(df_list, df_names):
    for i, d in enumerate(df_list):
        if type(df_list[i]) == tuple:
            print(f"{df_names[i]} Feature shape: {d[0].shape} / {df_names[i]} Label shape: {d[1].shape}")
        else:
           print(f"{df_names[i]}: {d.shape}\n")

def create_feature_label(feature_data, label_data, WINDOW, HORIZON):
    """
    When I want to create multiple variable feature dataset and single prediction label set. 
    """
    window, label_full = make_windows(train_data_open_scaled, window_size=WINDOW, horizon=HORIZON)
    win, label = make_windows(train_open_label, window_size=WINDOW, horizon=HORIZON)
    return window, label