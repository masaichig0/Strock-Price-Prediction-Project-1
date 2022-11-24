import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader as pdr
import yfinance as yf
import talib as ta
from talib import MA_Type
from statsmodels.tsa.stattools import adfuller
import matplotlib as mpl

def price_inverse(original_df, y_preds, predict_price):
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

def evaluate_result(inversed_df, predicted_price="Close"):
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
    window, label_full = make_windows(train_data_open_scaled, window_size=WINDOW, horizon=HORIZON)
    win, label = make_windows(train_open_label, window_size=WINDOW, horizon=HORIZON)
    return window, label

def preprocess_data(df, data_for, SMA=True, sma5=5, sma10=10, sma20=20,
              MACD=True, short_span=8, long_span=17, macd_span=9, 
              RSI=True, timeperiod=14, SOI=True, slowk_period=3, fastk_period=14, 
              Bband=True):
    """
    data_for: pick one of them ["Adj Close", "Open", "high", "Low"]
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
        df["SMA5"] = df[data_for].rolling(sma5).mean()
        df["SMA10"] = df[data_for].rolling(sma10).mean()
        df["SMA20"] = df[data_for].rolling(sma20).mean()
        
        df["smap5"] = df[percent].rolling(sma5).mean()
        df["smap10"] = df[percent].rolling(sma10).mean()
        df["smap20"] = df[percent].rolling(sma20).mean()
        
        
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

def print_shape(df_list, df_names):
    for i, d in enumerate(df_list):
        if type(df_list[i]) == tuple:
            print(f"{df_names[i]} Feature shape: {d[0].shape} / {df_names[i]} Label shape: {d[1].shape}")
        else:
           print(f"{df_names[i]}: {d.shape}\n") 
        
def create_feature_label(feature_data, label_data, WINDOW, HORIZON):
    window, label_full = make_windows(feature_data, window_size=WINDOW, horizon=HORIZON)
    win, label = make_windows(label_data, window_size=WINDOW, horizon=HORIZON)
    return window, label

def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    


def train_val_test_pridict_feature(df, test_split=0.2, random_state=42):
    """
    Get the feature to predict tomorrows price (z).
    
    Shift the label and feature so today's feature will predict tomorrows price. Then, 
    split train dataset (train_features, train_labels) and (X_test, y_test). 
    
    Split train dataset and validation dataset and will return:
    
        X_test, y_tets, z, X_train, X_val, y_train, y_val. 
    
    z: the last day of the feature to predict the future stock price.
    
    test_split: size of the test data. default is 20%.
    """
    
    df = df.drop(["Open", "Adj Close", "Change", "Direction"], axis=1)
    
    # Split labels and features
    X = df.drop("Close", axis=1)
    y = df["Close"]
    
    # set the final day of the feature as the data to predict the future price
    z = X[-1:]
    
    # Drop last row of the feature
    X = X[:-1]
    # Drop first row of the label
    y = y[1:]
    print(f"X: {len(X)}, y: {len(y)}")
    
    # Split train and test dataset
    from sklearn.model_selection import train_test_split
    train_features, X_test, train_labels, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
    
    print(f"final test feature is from {z.index[0].date()}")
    print(f"Train dataset length (90% of entire data): {len(train_features)}")
    print(f"Test_dataset length (10% of entire data): {len(X_test)}")
    
    # Split train dataset to train and validation dataset
    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=test_split, random_state=random_state)
    
    print(f"Length of data:\nX_train: {len(X_train)}\nX_val: {len(X_val)}\ny_train: {len(y_train)}\ny_val: {len(y_val)}")
    
    return X_test, y_test, z, X_train, X_val, y_train, y_val


def train_test_pridict_feature(df, test_split=0.2, random_state=42):
    """
    Drop all the columns that indicate the closing price.
    Set up X (features) and y (labels)
    Last day of the dataframe will predict the next day price, so store in z.
    Return z, X_train, X_test, y_train, y_test. 
    
    z: the last day of the feature to predict the future stock price.
    
    test_split: size of the test data. default is 20%.
    """
    df = df.drop(["Open", "Adj Close", "Change", "Direction"], axis=1)
    X = df.drop(["Close"], axis=1)
    y = df["Close"]
    
    # set the final day of the feature as the data to predict the future price
    z = X[-1:]
    
    print(f"final test feature is from {z.index[0].date()}")
    
    # Drop last row of the feature
    X = X[:-1]
    # Drop first row of the label
    y = y[1:]
    print(f"X: {len(X)}, y: {len(y)}")
    
    # import scikit learn to split randomly
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_state)
    
    print(f"Length of data:\nX_train: {len(X_train)}\nX_test: {len(X_test)}\ny_train: {len(y_train)}\ny_test: {len(y_test)}")
    
    return z, X_train, X_test, y_train, y_test

def train_test_feature_split_classification(df, test_split=0.2, random_seed=42):
    """
    Split data to z(last day of the feature to predict the next day), X_train, X_test, y_train, y_test.
    This split function for classification model. 
    y = direction of the stock price movement. 
    """
    df = df.drop(["Open", "Adj Close", "Change", "Close"], axis=1)
    X = df.drop(["Direction"], axis=1)
    y = df["Direction"]
    
    # Final day of the feature to predict later
    z = X[-1:]
    print(f"final test feature is from {z.index[0].date()}\n")
    print(f"Feature to predict:\n{z}")
    
    # Drop last row of the feature
    X = X[:-1]
    # Drop first row of label
    y = y[1:]
    print(f"X: {len(X)}, y: {len(y)}")
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_seed)
    
    print(f"Length of data:\nX_train: {len(X_train)}\nX_test: {len(X_test)}\ny_train: {len(y_train)}\ny_test: {len(y_test)}")
    
    return z, X_train, X_test, y_train, y_test

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

def get_data_your_option(symbol, start, end, expiry_date):
    """
    It will return the data from the date you purchased the option. 
    
    Args: 
        symbol: Your company symbol. (str)
        start: Purchased date("year-month-day")
        end: end date.
        expiry_date: your option's expiry date. ("year-month-day")
    """
    yf.pdr_override()
    df = pdr.data.get_data_yahoo(symbol, start=start, end=end)
    
    days_to_expiry = expiry_date - pd.to_datetime(df.index[-1])
    print(f"Your option will expire in {days_to_expiry.days} days. \n")
    
    return df

# function of obtaining data
def get_data(symbol, data_length):
    now = dt.datetime.now()
    start = now.replace(now.year - data_length).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")
    
    yf.pdr_override()
    df = pdr.data.get_data_yahoo(symbol, start=start, end=end)
    return df

import talib as ta
# Function to create the data from the data I get from yahoo finance.
def data_prep(symbol, data_length, 
              OBV=True, SMA=True, sma_short=15, sma_long=50, 
              MACD=True, short_span=12, long_span=26, macd_span=9, 
              RSI=True, timeperiod=14, SOI=True, fastk_period=14):
    """
    Add the indicators for later data manupulation. 
    
    Args:
        symbol: Company simbol you want to obtain the data. (dtype=str)
        data_length: The length of the data you want to get. (e.g. 10 mean 10 years of data).
        
    All the indicator's default is True. If you don't want to get the data, set for False.
        OBV: On-Balance-Volume indicator. 
        SMA: Simple Moving Average.
        sma_short: Short period of SMA. Default is 15.
        sma_long: Long period of SMA. Default is 50.
        MACD: MACD indicator.
        short_span: EMA short span for MACD.
        long_span: EMA long span for MACD. 
        macd_span: MACD signal span.
            default (9, 12, 26) - (macd_span, short_span, long_span)
        RSI: RSI indicator.
        timeperiod: RSI time period. default is 14
        SOI: Stochastic Oscillator indicator. 
        fastk_period: %K period. default is 14.
    """
    # Get the dataset and make a copy of it. 
    data = get_data(symbol, data_length)
    df = data.copy()
    
    # Difference in the closing price each day
    df["Change"] = df["Close"].diff()
    # Show the direction of the movement
    df = df.dropna()
    df["Direction"] = None
    for i in range(len(df)):
        if df["Change"][i] > 0:
            df["Direction"][i] = "UP"
        elif df["Change"][i] < 0:
            df["Direction"][i] = "DOWN"
        else:
            df["Direction"][i] = "SAME"
            
        # OBV 
    if OBV:
        df["OBV"] = 0
        for i in range(len(df)):
            if df["Direction"][i] == "UP":
                df["OBV"][i] = df["Volume"][i] + df["OBV"][i-1] 
            elif df["Direction"][i] == "DOWN":
                df["OBV"][i] = df["OBV"][i-1] - df["Volume"][i]
            else:
                df["OBV"][i] = 0 + df["OBV"][i-1]
                
    # SMA
    if SMA:
        df["SMA/short"] = df["Close"].rolling(sma_short).mean()
        df["SMA/long"] = df["Close"].rolling(sma_long).mean()
        
    # MACD
    if MACD:
        # Calculate the short term exponential moving average
        shortEMA = df["Close"].ewm(span=short_span, adjust=False).mean()
        # Calculate the long term exponential moving average
        longEMA = df["Close"].ewm(span=long_span, adjust=False).mean()
        # Calculate MACD line
        MACD = shortEMA - longEMA
        # calculate the signal line
        signal = MACD.ewm(span=macd_span, adjust=False).mean()
        # Add to the DataFrame
        df["MACD"] = MACD
        df["Signal Line"] = signal
        
    # RSI
    if RSI:
        df["RSI"] = ta.RSI(df["Close"], timeperiod=timeperiod)
        
    # Stochastic
    if SOI:
        df["SlowK"], df["SlowD"] = ta.STOCH(high=df["High"],
                                   low=df["Low"],
                                   close=df["Close"], 
                                   fastk_period=fastk_period)
        
    return df


# Functon to evaluate: accuracy, precision,  recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calculate model accuracy, precision, recall and f1-score of a binary classification model.
  """
  # Calculate mode accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1-score "weighted" average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                   "precision": model_precision,
                   "recall": model_recall,
                   "f1": model_f1}
  return model_results


# Plot confusion matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.
    
    If classes is passed, confusion matrix will be labelled, if not, integer classes values will be used.
    
    Args:
        y_true: Arrey of truth labels (must be same shape as y_pred).
        y_pred: Array of predicted labels (must be same shape as y_true).
        classes: Array of class labels (e.g. strings form). If `None`, integer labels are used.
        figsize: Size of output figure (default=(10, 10)).
        text_size: Size of output figure text (defalut=15).
        norm: normalize values or not (default=False).
        savefig: save confusion matrix to file (default=False).
        
    Returns:
        A labelled confusion matrix plot compareing y_true and y_pred.
        
    Example usage:
        make_confusion_matrix(y_true=test_labels, # ground truth test labels
                              y_pred=y_preds, # predicted labels
                              classes=class_names, # array of class label names
                              figsize=(15, 15),
                              test_size=10)
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:,np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with
    
    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # color will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)
    
    # Are there a list classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
        
    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes), # Create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels, # axes will labeled with class names (if they exist) or ints
           yticklabels=labels)
    
    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    
    # plot x-labels vertically
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)
    
    # set the threshold for diffrent colors
    threshold = (cm.max() + cm.min()) / 2.
    
    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%", 
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
            
    # Save the figure to the current working directory
    if savefig:
        fig.save("confusion_matrix.png")


# Plot loss curves 
import matplotlib.pyplot as plt

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

def compare_historys(original_history, new_history, initial_epochs=5):
     """
     Compare two TensorFlow History objects.
     """
     # Get original history measurements
     acc = original_history.history['accuracy']
     loss = original_history.history["loss"]

     val_acc = original_history.history["val_accuracy"]
     val_loss = original_history.history["val_loss"]

     # Combine original history metrics with new_history metorics
     total_acc = acc + new_history.history["accuracy"]
     total_loss = loss + new_history.history["loss"]

     total_val_acc = val_acc + new_history.history["val_accuracy"]
     total_val_loss = val_loss + new_history.history["val_loss"]

     # Make plots
     plt.figure(figsize=(8, 8))
     plt.subplot(2, 1, 1)
     plt.plot(total_acc, label="Training Accuracy")
     plt.plot(total_val_acc, label="Val Accuracy")
     plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start Fine Tuning")
     plt.legend(loc="lower right")
     plt.title("Training and Validation Accuracy")
        
     # Make loss
     plt.figure(figsize=(8, 8))
     plt.subplot(2, 1, 2)
     plt.plot(total_loss, label="Training Loss")
     plt.plot(total_val_loss, label="Val Loss")
     plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start Fine Tuning")
     plt.legend(loc="upper right")
     plt.title("Training and Validation Loss")

# Function to create the model from URL
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

def create_model(model_url, num_classes=10, acitivation="softmax", input_shape=(224, 224, 3)):
    """
    Takes a TensorFlow Hub URL and create a Keras Sequential model with it.
    
    Args:
        model_url (str): A TensorFlow Hub feature extraction URL.
        num_classes (int): Number of output neurons in the output layer,
        should be equal to number of target classes, default 10.
        activation: Output activation function. Defalult "softmax".
        input_shape (tuple): Defalt (224, 224, 3)
        
    Returns:
        All uncompiled Keras Sequential model with model_url as feature extractor
        layer and Dense output layer with num_classes output neurons.
    """
    # Download the pretrained model and save it as Keras Layer
    feature_extractor_layer = hub.KerasLayer(model_url, 
                                             trainable=False, # freeeze the already learned patterns
                                             name="feature_extraction_layer",
                                             input_shape=input_shape)
    
    # Create the model
    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(num_classes, activation, name="output_layer")
    ])
    
    return model

# Compile the model
def compile_model(model, learning_rate):
    """
    Compile the model.
    
    Args:
        model - model you created.
        loss : Change apporopreate loss function for your purpose. 
        Learning rate: 
    """
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
        
# Unzip file
import zipfile
def unzip_file(filename):
    """
    Unzip filename into current working directory.
    """
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()
    

# Make a function to unfreeze trainable layers
def unfreeze_trainable_layers(model, num_layers_unfreeze, base_model):
    """
    Unfreeze trainable layers for fine tuning.

    Args:
      model: The model you are working on. This is not the same as base_model. 
      num_layers_unfreeze (int): number of layers you want to unfreeze
    """
    base_model.trainable = True

    for layer in base_model.layers[:-num_layers_unfreeze]:
      layer.trainable = False

    print(f"Number of trainable layers in feature extraction model is {len(model.layers[2].trainable_variables)}")

# Create TensorBoard Callback 
import datetime as dt

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = os.path.join(dir_name + "/" + experiment_name + "/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


# Walk through data directory
import os

def walk_through_dir(filename):
    for dirpath, dirnames, filenames in os.walk(filename):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
        
# prepare the image to predict
def load_and_prep_image(filename, img_size=224, scale=True):
    """
    Prepare the image file to predict. This function is inside the pred_and_plot function. 
    
    Args: 
        filename: string file name of target image.
        img_size: Size to resize target image to, default 224
        scale (bool): whether to scale pixel values to range(0, 1), default=True
   
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, [img_size, img_size])
    
    if scale:                            
        return img/255.
    else:
        return img
    
        
# Predict new image and plot it
import numpy as np
def pred_and_plot(model, file_name, class_name, img_shape):
    """
    Predict the image for your training model. Import the load_and_prep_image function to prepare the image to predict.
    
    Args:
        model: Trained model that you want to predict.
        filename: The full path of the image.
        class_name (str): Default is class_names. Set the variable class_names before call the function.
        img_shape: Tuple. Image size. 
    Return:
         Result and image.
    """
    img = load_and_prep_image(file_name)
    pred = model.predict(tf.expand_dims(img, axis=0))
    
    if len(pred[0]) > 1:
        pred_class = class_name[tf.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]
        
    plt.imshow(img)
    plt.title(f"Prediction {pred_class} {np.round(pred[0][0] * 100, 2)}%")
    plt.axis(False);
        
# View a random image and compare it to its augmented virsion
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import tensorflow as tf

def view_random_image_norm_and_aug(train_data, target_dir, data_augmentation):
    """
    View random training image and augmented image. 
    
    Args:
        train_data: Preprocess the images from directory. 
                example: train_data = tf.keras.preprocesssing.image_dataset_from_directory(data_directory, 
                                                                                           image_size=(224, 224,3),
                                                                                           label_mode="categorical")
        target_dir: image directory.
        data_augmentation: Image data that processed augmented.
        
    Returns:
       Show both training image and augmented image.
    """
    target_class = random.choice(train_data.class_names)
    target_dir = target_dir + "/" + target_class
    random_image = random.choice(os.listdir(target_dir))
    random_image_path = target_dir + "/" + random_image
    
    # Read in the random image
    img = mpimg.imread(random_image_path)
    ax = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original random image from class: {target_class}")
    plt.axis(False)
    
    # Plot random augmented image
    augmented_img = data_augmentation(tf.expand_dims(img, axis=0))
    plt.subplot(1, 2, 2)
    plt.imshow(tf.squeeze(augmented_img)/255.)
    plt.title(f"Augmented random image from class: {target_class}")
    plt.axis(False);
    
# View random image from the file
from random import choice
from tensorflow.keras import preprocessing
import os

def view_random_image(img_data_directory, num_img_show=8, rows=2, columns=4, figsize=(12, 6)):
    """
    Show random images from the file. The file format must be match with the format I set up. 
    
    Args:
        img_data_directory (str): 1st layer of the file directory.
        num_img_show (int): number of the image you want to display. Defalt is 8.
        rows (int): Number of rows to show the image (It must match it the number of total image you want to display).
                    Defalt is 2.
        columns (int): Number of columns to show the image (It must match it the number of total image you want to display).
                        Defalt is 4.
        figsize tuple(width, height): Defalt is (12, 6)
        
    Return:
        Random Image
    """
    # Obtain random image path
    img_data_dir = img_data_directory
    train_test = choice(os.listdir(img_data_dir))
    img_file_dir = img_data_dir + "/" + train_test
    data = preprocessing.image_dataset_from_directory(img_file_dir)
    class_name = choice(data.class_names)
    target_dir = img_file_dir + "/" + class_name
    print(f"Those images are from {train_test} dataset,\nClass name is {class_name}.")

    
    # Set the plot
    plt.figure(figsize=figsize)
    
    for i in range(num_img_show):
        # Plot random image
        ax = plt.subplot(rows, height, i + 1)
        
        random_image = choice(os.listdir(target_dir))
        random_image_path = target_dir + "/" + random_image
        
        img = mpimg.imread(random_image_path)
        plt.imshow(img)
        plt.title(f"{class_name}")
        plt.axis(False);
        
# Functions using with NLP

# predict own text
def predict_text(model, text):
    """
    Predict binary classification (disaster tweet or not disaster tweet)
    """
    pred_prob = tf.squeeze(model.predict([text]))
    pred = tf.round(pred_prob)
    print(f"Pred: {pred},\nPred_prob: {pred_prob},\nText:\n{text}")
    
# Functon to evaluate: accuracy, precision,  recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
    """
    Calculate model accuracy, precision, recall and f1-score of a binary classification model.
    """
    # Calculate mode accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1-score "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results

# Function to read the lines a document
def get_lines(filename):
    """
    Reads filename (a text filename) and returns of text as a list.
    
    Arg:
        fiename: a string containing the target filepath.
    
    Retrns:
        A list of strings with one string per line from the target filename.
    """
    with open(filename, "r") as f:
        return f.readlines()

# preprocess the text into dictionary to be able to train the model
def preprocess_text_with_line_numbers(filename):
    """
    Returns a list of dictionaries of abstract line data.
    
    Take in filename, reads it constants and sorts of through each line,
    extracting things like the target label, the text of the sentence, 
    how manny sentences are in the current abstract and what sentence
    number the target line is.
    """
    input_lines = get_lines(filename) # get al lines from fiename
    abstract_lines = '' # create an empty abstract
    abstract_samples = [] # Create an empty list of abstracts
    
    # Loop through each line in the target file
    for line in input_lines:
        if line.startswith("###"): # check to see if the line is ID line
            abstract_id = line
            abstract_lines = '' # reset the abstract string if the line is an ID line
        elif line.isspace(): # check to see if the line is a new line
            abstract_line_split = abstract_lines.splitlines() #split abstract into seperate lines
            
            # iterate through each line in a single abstract and count them at the same time
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}
                target_text_split = abstract_line.split("\t") # split target label from text
                line_data["target"] = target_text_split[0]
                line_data["text"] = target_text_split[1].lower()
                line_data["line_number"] = abstract_line_number
                line_data["total_lines"] = len(abstract_line_split) -1
                abstract_samples.append(line_data)
        else:
            abstract_lines += line
            
    return abstract_samples

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

# Create the function to split the train and test data
def train_test_split_time_series(split_size, timesteps, prices):
  """
  Split train and test dataset for time series. 

  Parameters
  ------------
  split_size : (Float number) size of train dataset
  timesteps : array of timestep value
  prices : array of price value

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

# MASE implementation
def mean_absolute_scaled_error(y_true, y_pred):
  """
  Implement MASE (mean absolute scaled error) assuming no seasonlity data.
  """
  mae = tf.reduce_mean(tf.abs(y_true - y_pred))

  # Find MAE of naive forcast (no seasonality)
  mean_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))

  return mae / mean_naive_no_season

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

import os

# Create a functon to implement a ModelCheckpoint callback with a specific filename
def create_model_checkpoint(model_name, save_path):
  return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                            verbose=0,
                                            save_best_only=True, 
                                           )

# Function to predict the model result
def make_preds(model, input_data):
  """
  Uses model to make predictions input_data.
  """
  forecast = model.predict(input_data)
  return tf.squeeze(forecast) # return 1D array of predictions

def get_ensemble_models(horizon,
                        train_data,
                        test_data,
                        num_iter=10,
                        num_epochs=1000,
                        loss_fn=["mae", "mse", "mape"]):
  """
  Returns a list of num_iter models each trained on MAE, MSE, an MAPE loss.

  For example: if num_iter=10, a list of 30 trained models will be returned:
    10 * len(["mae", "mse", "mape"]).
  """
  # Making empty list for trained ensemble models
  ensemble_models = []

  # Create num_iter number of models per loss function
  for i in range(num_iter):
    # Build and fit a new model with a different loss function
    for loss_function in loss_fn:
      print(f"Optimizing model by reducing: {loss_function} for {num_epochs} epochs, model number: {i}")

      # Construct a simple model (similar to medel_1)
      model = tf.keras.Sequential([
          # Initialize dense layers with normal distribution for estimating prediction intervals later on
          layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
          layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
          layers.Dense(HORIZON)
      ])

      # Compile simple model with current loss functon
      model.compile(loss=loss_function,
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["mae", "mse"])
      
      # Fit the current model
      model.fit(train_data,
                epochs=num_epochs,
                verbose=0,
                validation_data=test_data,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                            patience=200,
                                                            restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                                patience=100,
                                                                verbose=1)])
      # Append fitted model to list of ensemble models
      ensemble_models.append(model)
  
  return ensemble_models

# Create a function which uses a list of trained models to make and return a list of predictions
def make_ensemble_preds(ensemble_models, data):
  ensemble_preds = []
  for model in ensemble_models:
    preds = model.predict(data)
    ensemble_preds.append(preds)
  return tf.constant(tf.squeeze(ensemble_preds))

# Find upper and lower bounds of ensemble predictions
def get_upper_lower(preds): # 1. Take the predictions from a number of randomly initialized models
  """
  Get lower and upper bounds of the prediction. 

  Set to 95% intervals
  """
  # 2. Measure the standard deviation of the predictions
  std = tf.math.reduce_std(preds, axis=0)

  # 3. Multiply the standard deviation by 1.96
  interval = 1.96 * std

  # 4. Get the prediction interval upper and lower bounds
  preds_mean = tf.reduce_mean(preds, axis=0)
  lower, upper = preds_mean - interval, preds_mean + interval

  return lower, upper

def make_future_forecasts(values, model, into_future, window_size) -> list:
  """
  Make future forcasts into_future steps after values ends. 

  Parameters :
    values: Dataset the model will learn. 
    model: the model you created to train on.
    into_future: how many days into futre you want to predict?
    window_size: input size

  Returns future forecasts as a list of floats.
  """
  # 2. Create an empty list for future forecasts/prepare data to forecast on
  future_forecasts = []
  last_window = values[-window_size:]

  # 3. Make INTO-FUTURE number of predictions, altering the data which gets predicted on each
  for _ in range(into_future):
    """
    Predict on the last window then append it again, again and again 
    (our model will eventually start to make forecasts on its own forecasts)
    """
    future_pred = model.predict(tf.expand_dims(last_window, axis=0))
    print(f"Predicting on:\n{last_window} -> Prediction:{tf.squeeze(future_pred).numpy()}\n")

    # Append predictions to future_forecast
    future_forecasts.append(tf.squeeze(future_pred).numpy())

    # Update last window with new pred and get WINDOW_SIZE most recent preds (model was trained on WINDOW_SIZE windows)
    last_window = np.append(last_window, future_pred)[-window_size:]

  return future_forecasts