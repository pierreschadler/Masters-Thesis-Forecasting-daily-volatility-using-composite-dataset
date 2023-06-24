import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.metrics
from tensorflow.keras import layers
import scipy
import random

tf.compat.v1.reset_default_graph()
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)


# =============================================================================
# Importing data
# =============================================================================

#Importing Refinitiv data
data_raw = pd.read_excel('data/Refinitiv_data.xlsx', sheet_name = "data", index_col=0)  
data_raw = data_raw.drop(["VOLUME",
                          "FCHI_M_HIGH","FCHI_M_LOW","FCHI_M_OPEN","FCHI_M_CLOSE",
                          "DJ_M_HIGH","DJ_M_LOW","DJ_M_OPEN","DJ_M_CLOSE",
                          "EU_M_HIGH","EU_M_LOW","EU_M_OPEN","EU_M_CLOSE"],axis=1)
m_periods = [1,3,6,9]

for factor in data_raw.columns:
    for period in m_periods:
        temp = pd.DataFrame(data_raw[factor].rolling(period).mean())
        temp = temp.rename(columns={factor : factor+str(period)})
        data_raw = data_raw.join(temp, how = "left")


#Importing CSV data
FCHI_csv = pd.read_csv("data/^FCHI.csv",parse_dates=['Date'], index_col=['Date'])
GDAXI_csv = pd.read_csv("data/^GDAXI.csv",parse_dates=['Date'], index_col=['Date'])
IXIC_csv = pd.read_csv("data/^IXIC.csv",parse_dates=['Date'], index_col=['Date'])
EURUSD_csv = pd.read_csv("data/EURUSD=X.csv",parse_dates=['Date'], index_col=['Date'])
EURUSD_csv = EURUSD_csv.pop('Adj Close')

FCHI_csv = FCHI_csv.drop(["Close"],axis=1)
GDAXI_csv = GDAXI_csv.drop(["Close"],axis=1)
IXIC_csv = IXIC_csv.drop(["Close"],axis=1)

FCHI_csv = FCHI_csv.rename(columns={'Open': 'FCHI_D_OPEN',
                                    'High': 'FCHI_D_HIGH',
                                    'Low': 'FCHI_D_LOW',
                                    'Adj Close': 'FCHI_D_CLOSE',
                                    'Volume': 'FCHI_D_VOLUME'})
GDAXI_csv = GDAXI_csv.rename(columns={'Open': 'GDAXI_D_OPEN',
                                    'High': 'GDAXI_D_HIGH',
                                    'Low': 'GDAXI_D_LOW',
                                    'Adj Close': 'GDAXI_D_CLOSE',
                                    'Volume': 'GDAXI_D_VOLUME'})
IXIC_csv = IXIC_csv.rename(columns={'Open': 'IXIC_D_OPEN',
                                    'High': 'IXIC_D_HIGH',
                                    'Low': 'IXIC_D_LOW',
                                    'Adj Close': 'IXIC_D_CLOSE',
                                    'Volume': 'IXIC_D_VOLUME'})



#FCHI_csv = FCHI_csv.dropna(axis=0)
#GDAXI_csv = GDAXI_csv.dropna(axis=0)
#IXIC_csv = IXIC_csv.dropna(axis=0)


TNX_csv = pd.read_csv("data/^TNX.csv",parse_dates=['Date'], index_col=['Date'])
TNX_csv = pd.DataFrame(TNX_csv)
TNX_csv.rename({'Adj Close': 'RF_Y'}, axis=1, inplace=True)
TNX_csv = TNX_csv["RF_Y"]/100
TNX_csv = pd.DataFrame(TNX_csv)
TNX_csv = TNX_csv.dropna(axis=0)
    

# Issue volume column

# Join dataframes
data = FCHI_csv.join(TNX_csv)
data = data.join(IXIC_csv)
data = data.join(GDAXI_csv)
data = data.dropna(axis=0)


data = data.join(data_raw)



# =============================================================================
# Creating factors
# =============================================================================
periods = [1,5,15,30]

data["FCHI_RET"] = data["FCHI_D_CLOSE"].pct_change()
data["GDAXI_RET"] = data["GDAXI_D_CLOSE"].pct_change()
data["IXIC_RET"] = data["IXIC_D_CLOSE"].pct_change()

data["FCHI_RV"] = data["FCHI_RET"].rolling(2).std()*(252**0.5)
data["GDAXI_RV"] = data["GDAXI_RET"].rolling(2).std()*(252**0.5)
data["IXIC_RV"] = data["IXIC_RET"].rolling(2).std()*(252**0.5)



data["FCHI_SHARPE"] = (data["FCHI_RET"]-(data["RF_Y"]/252))/data["FCHI_RV"]
data["GDAXI_SHARPE"] = (data["GDAXI_RET"]-(data["RF_Y"]/252))/data["GDAXI_RV"]
data["IXIC_SHARPE"] = (data["IXIC_RET"]-(data["RF_Y"]/252))/data["IXIC_RV"]

data["FCHI_SHARPE_PCT"] = data["FCHI_SHARPE"].pct_change()  
data["GDAXI_SHARPE_PCT"] = data["GDAXI_SHARPE"].pct_change()
data["IXIC_SHARPE_PCT"] = data["IXIC_SHARPE"].pct_change()

#Bollinger bands

data["FCHI_BOLLINGER_RANGE"] = np.subtract(
    data["FCHI_D_CLOSE"].rolling(20).mean()+data["FCHI_D_CLOSE"].rolling(20).std()*2,
    data["FCHI_D_CLOSE"].rolling(20).mean()-data["FCHI_D_CLOSE"].rolling(20).std()*2)

data["GDAXI_BOLLINGER_RANGE"] = np.subtract(
    data["GDAXI_D_CLOSE"].rolling(20).mean()+data["GDAXI_D_CLOSE"].rolling(20).std()*2,
    data["GDAXI_D_CLOSE"].rolling(20).mean()-data["GDAXI_D_CLOSE"].rolling(20).std()*2)

data["IXIC_BOLLINGER_RANGE"] = np.subtract(
    data["IXIC_D_CLOSE"].rolling(20).mean()+data["IXIC_D_CLOSE"].rolling(20).std()*2,
    data["IXIC_D_CLOSE"].rolling(20).mean()-data["IXIC_D_CLOSE"].rolling(20).std()*2)

data["FCHI_BOLLINGER_RANGE_PCT"] = data["FCHI_BOLLINGER_RANGE"].pct_change()
data["GDAXI_BOLLINGER_RANGE_PCT"] = data["GDAXI_BOLLINGER_RANGE"].pct_change()
data["IXIC_BOLLINGER_RANGE_PCT"] = data["IXIC_BOLLINGER_RANGE"].pct_change()

#True range

data["FCHI_TRUE_RANGE"] = np.max(np.transpose(np.array([(data["FCHI_D_HIGH"]-data["FCHI_D_LOW"]),
                                 np.abs((data["FCHI_D_HIGH"]-data["FCHI_D_CLOSE"].shift())),
                                 np.abs((data["FCHI_D_LOW"]-data["FCHI_D_CLOSE"].shift()))]
                                 )), axis=1)
data["GDAXI_TRUE_RANGE"] = np.max(np.transpose(np.array([(data["GDAXI_D_HIGH"]-data["GDAXI_D_LOW"]),
                                 np.abs((data["GDAXI_D_HIGH"]-data["GDAXI_D_CLOSE"].shift())),
                                 np.abs((data["GDAXI_D_LOW"]-data["GDAXI_D_CLOSE"].shift()))]
                                 )), axis=1)
data["IXIC_TRUE_RANGE"] = np.max(np.transpose(np.array([(data["IXIC_D_HIGH"]-data["IXIC_D_LOW"]),
                                 np.abs((data["IXIC_D_HIGH"]-data["IXIC_D_CLOSE"].shift())),
                                 np.abs((data["IXIC_D_LOW"]-data["IXIC_D_CLOSE"].shift()))]
                                 )), axis=1)


data["FCHI_TRUE_RANGE_PCT"] = data["FCHI_TRUE_RANGE"].pct_change()
data["GDAXI_TRUE_RANGE_PCT"] = data["GDAXI_TRUE_RANGE"].pct_change()
data["IXIC_TRUE_RANGE_PCT"] = data["IXIC_TRUE_RANGE"].pct_change()

#Create listsR

periods = [1,5,15,30]
indices = ["FCHI","GDAXI", "IXIC"]
technical_factors = ["_RET","_RV","_SHARPE",
                     "_SHARPE_PCT","_BOLLINGER_RANGE","_BOLLINGER_RANGE_PCT",
                     "_TRUE_RANGE","_TRUE_RANGE_PCT"]



indices_factors = []
for factor in technical_factors:
    for indice in indices:
        indices_factors.append(indice+factor)

#Rolling the factors to capture trends

for factor in indices_factors:
    for period in periods:
        temp = pd.DataFrame(data[factor].rolling(period).mean())
        temp = temp.rename(columns={factor : factor+str(period)})
        data = data.join(temp, how = "left")

# =============================================================================
# Final formatting
# =============================================================================
        
factors = list(data.columns)
factors.remove("FCHI_RV")

data[factors] = data[factors].shift(-1)

data = data.reindex(index=data.index[::-1])

        
# =============================================================================
# Nettoyer
# =============================================================================

data.isnull().sum()
np.isinf(data).values.sum()
data.replace([np.inf, -np.inf], 0, inplace=True)
data.isnull().sum().sum()
data = data.fillna(0)
data.isnull().sum().sum()
data = data.dropna()

list(data_raw.columns)


(data.shape[1] * data.shape[0])

(data == 0).sum().sum() / (data.shape[1] * data.shape[0])
yo = (data == 0).sum().sort_values(ascending=False)
yo.sum()
1-yo[list(data_raw.columns)].sum() / yo.sum()


#data.replace(0, np.nan, inplace=True)


for col in data.columns:
    
    mean = data[col].mean()
    sd = data[col].std()
    print(col, "mean,",mean," sd,",sd)
    
    data = data.drop(data[(data[col] > (mean + (10*sd))) | (data[col] < (mean - (10*sd)))].index)
        

# =============================================================================
# Descriptive statistics
# =============================================================================

data[['FCHI_RET', 'GDAXI_RET', 'IXIC_RET', 'FCHI_RV', 'GDAXI_RV', 'IXIC_RV',
       'FCHI_SHARPE', 'GDAXI_SHARPE', 'IXIC_SHARPE', 'FCHI_BOLLINGER_RANGE',
       'GDAXI_BOLLINGER_RANGE', 'IXIC_BOLLINGER_RANGE',
       'FCHI_TRUE_RANGE', 'GDAXI_TRUE_RANGE',
       'IXIC_TRUE_RANGE']].describe().transpose()

data["FCHI_RV"].plot(color = "black")

# =============================================================================
# Divide
# =============================================================================

train_test_ratio = round(data.shape[0]*0.75)

train_dataset = data[-train_test_ratio:].copy()
test_dataset = data[:(len(data)-train_test_ratio)].copy()


train_features = train_dataset.drop(["FCHI_RV"], axis=1)
test_features = test_dataset.drop(["FCHI_RV"], axis=1)

train_labels = train_dataset.pop('FCHI_RV')
test_labels = test_dataset.pop('FCHI_RV')

# =============================================================================
# Normalization
# =============================================================================

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

# =============================================================================
# Normalization 3D
# =============================================================================

def get3D_create_dataset(dataX, dataY, seq_length, step):
    Xs, ys = [], []
    for i in range(0, len(dataX) - seq_length + 1, step):
        v = dataX.iloc[i:(i + seq_length)].values
        Xs.append(v)
        ys.append(dataY.iloc[i + seq_length - 1])
    return np.array(Xs), np.array(ys)

datax = data.loc[:, data.columns != 'FCHI_RV']
datay = data['FCHI_RV']

Xs, Ys = get3D_create_dataset(datax,datay,15,1)

X_train = Xs[-train_test_ratio:,:,:]
X_test = Xs[:(Xs.shape[0]-train_test_ratio),:,:]

Y_train = Ys[-train_test_ratio:]
Y_test = Ys[:(Ys.shape[0]-train_test_ratio)]

def get3D_minmaxscaleX(X_train):
    train_max = np.max(X_train, axis = 0)
    train_max = np.max(train_max, axis = 0)
    train_min = np.min(X_train, axis = 0)
    train_min = np.min(train_min, axis = 0)
    train_minmax = (train_max - train_min)
    X_train = X_train - train_min.T
    X_train = X_train / train_minmax.T
    return (X_train, train_min, train_minmax)

#X_train, train_min, train_minmax = get3D_minmaxscaleX(X_train)

def get_Xtest_scaleX(X_test, train_min, train_minmax):
    X_test = (X_test - train_min) / train_minmax
    return X_test

#X_test = get_Xtest_scaleX(X_test, train_min, train_minmax)

#X_train[np.isnan(X_train)] = 0
#X_test[np.isnan(X_test)] = 0


# =============================================================================
# Definitions
# =============================================================================

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.ylim()
  plt.xlabel('Epoch')
  plt.ylabel('Mean squared error')
  plt.legend()
  plt.grid(True)
  
def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

def plot_fit(labels, pred):
  plt.plot(labels, label='FCHI_RV')
  plt.plot(pred, label='Predicted')
  plt.ylim()
  plt.xlabel('Time (in days)')
  plt.ylabel('FCHI_RV')
  plt.legend()
  plt.grid(True)

#Add performance results
Results=pd.DataFrame()
Results.index = ["MSE","RMSE","MAE","MPE"]

# =============================================================================
# Linear regression
# =============================================================================

tf.keras.backend.clear_session()

init0 = tf.keras.initializers.RandomUniform(seed=2)

linear_model = tf.keras.Sequential([
    layers.Masking(mask_value=0.),
    normalizer,
    layers.Dense(units=1, activation=None,kernel_initializer=init0)])


linear_model.compile(loss='MeanSquaredError',
              optimizer=tf.keras.optimizers.Adam(0.0001),
              metrics=[
                  tf.keras.metrics.RootMeanSquaredError(name='RMSE'),
                  tf.keras.metrics.MeanAbsoluteError(name='MAE'),
                  tf.keras.metrics.MeanAbsolutePercentageError(name='MPE')])





history_linear_model = linear_model.fit(
    train_features,
    train_labels,
    epochs=50,
    verbose=2)

linear_model.summary()

hist = pd.DataFrame(history_linear_model.history)


plot_loss(history_linear_model)

    
test_predictions_linear_model = linear_model.predict(test_features).flatten()

r_linear_model = rsquared(test_labels, test_predictions_linear_model)
print("ols Rsquared : ", r_linear_model)


test= pd.DataFrame()
test["test_labels"] = test_labels
test = test.reset_index()
test["ols_pred"] = test_predictions_linear_model
test = test[:]
plot_fit(test["test_labels"],test["ols_pred"])



Results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=1)


# =============================================================================
# DNN
# =============================================================================


dnn_model = keras.Sequential([
    layers.Masking(mask_value=0.),
    normalizer,
    layers.Dense(8, activation='sigmoid',kernel_initializer=init0),
    layers.Dense(16, activation='relu',kernel_initializer=init0),
    layers.Dense(16, activation='relu',kernel_initializer=init0),
    layers.Dense(8, activation='relu',kernel_initializer=init0),
    layers.Dense(1, activation=None,kernel_initializer=init0)])
  
dnn_model.compile(loss='MeanSquaredError',
              optimizer=tf.keras.optimizers.Adam(0.0001),
              metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='RMSE'),
                tf.keras.metrics.MeanAbsoluteError(name='MAE'),
                tf.keras.metrics.MeanAbsolutePercentageError(name='MPE')])



history_dnn_model = dnn_model.fit(
    train_features,
    train_labels,
    epochs=150,
    verbose=2)
dnn_model.summary()


plot_loss(history_dnn_model)

test_predictions_dnn_model = dnn_model.predict(test_features).flatten()

r_dnn_model = rsquared(test_labels, test_predictions_dnn_model)
print("ols Rsquared : ", r_dnn_model)


pred_dnn= pd.DataFrame()
pred_dnn["test_labels"] = test_labels
pred_dnn = pred_dnn.reset_index()
pred_dnn["dnn_pred"] = test_predictions_dnn_model
pred_dnn = pred_dnn[:]
plot_fit(test["test_labels"],pred_dnn["dnn_pred"])


Results['dnn_model'] = dnn_model.evaluate(
    test_features, test_labels, verbose=2)
Results.transpose()

# =============================================================================
# LSTM
# =============================================================================

normalizer_lstm = tf.keras.layers.Normalization(axis=-1)
normalizer_lstm.adapt(np.array(X_train))


lstm_model = tf.keras.models.Sequential([
    layers.Masking(mask_value=0.),
    normalizer_lstm,
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True,kernel_initializer=init0)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True,kernel_initializer=init0)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True,kernel_initializer=init0)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,kernel_initializer=init0)),
    tf.keras.layers.Dense(1,kernel_initializer=init0),])

lstm_model.compile(loss='MeanSquaredError',
              optimizer=tf.keras.optimizers.RMSprop(centered=True),
              metrics=[
                tf.keras.metrics.RootMeanSquaredError(name='RMSE'),
                tf.keras.metrics.MeanAbsoluteError(name='MAE'),
                tf.keras.metrics.MeanAbsolutePercentageError(name='MPE')])



history_lstm_model = lstm_model.fit(
    X_train,
    Y_train,
    verbose=2, epochs=75)
lstm_model.summary()



plot_loss(history_lstm_model)

test_predictions_lstm_model = lstm_model.predict(X_test).flatten()

r_lstm_model = rsquared(Y_test, test_predictions_lstm_model)
print("ols Rsquared : ", r_lstm_model)


pred_lstm= pd.DataFrame()
pred_lstm["test_labels"] = Y_test
pred_lstm = pred_lstm.reset_index()
pred_lstm["lstm_pred"] = test_predictions_lstm_model
pred_lstm = pred_lstm[:]
plot_fit(pred_lstm["test_labels"],pred_lstm["lstm_pred"])


Results=pd.DataFrame()
Results.index = ["MSE","RMSE","MAE","MPE"]

Results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=1)

Results['dnn_model'] = dnn_model.evaluate(
    test_features, test_labels, verbose=2)

Results['lstm_model'] = lstm_model.evaluate(
    X_test, Y_test, verbose=2)
Results.transpose()


Results.loc[len(Results.index)] = [r_linear_model, r_dnn_model, r_lstm_model]

Results = Results.transpose()

Results = Results.rename(columns={4: "RSquared"})



Results["MSE"].plot.bar(title = "MSE per model",color=["black"])
Results["RMSE"].plot.bar(title = "RMSE per model",color=["black"])
Results["MAE"].plot.bar(title = "MAE per model",color=["black"])
Results["MPE"].plot.bar(title = "MPE per model",color=["black"])
Results["RSquared"].plot.bar(title = "RSquared per model",color=["black"])
