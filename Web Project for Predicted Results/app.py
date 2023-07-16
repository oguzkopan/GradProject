import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas_ta as ta

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
start_input = st.text_input('Enter Start Date', '2020-01-01')
end_input = st.text_input('Enter End Date', '2023-01-01')
df = yf.download(user_input, start_input, end_input)
df = df.reset_index()   
df['RSI'] = ta.rsi(df.Close, length=12)

#Describing Data
st.subheader(user_input + ' Data from ' + start_input + ' to ' + end_input)
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time Chart in Candlestick')
fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
st.plotly_chart(fig,use_container_width=True)


st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 50MA & 100MA')
ma100 = df.Close.rolling(50).mean()
ma200 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close, 'b')
st.pyplot(fig)


#Splitting data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Splitting data into x_train and y_train
x_train = []
y_train = []

"""
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
"""
#Load My model
model = load_model('keras_model.h5')

#Testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scale_factor = scaler.scale_
y_predicted = y_predicted * scale_factor[0]
y_test = y_test * scale_factor[0]


#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label= 'Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

########Facebook Prophet#########
from prophet import Prophet
import copy

#Check if NA values are in data
df=df[df['Volume']!=0]
df.reset_index(drop=True, inplace=True)
df.isna().sum()
df['y'] = df['Close']
df['ds']=pd.to_datetime(df['Date'])

def prophet_signal(df, l, backcandles, frontpredictions, diff_limit, signal):
    dfsplit = copy.deepcopy(df[l-backcandles:l+1])
    model=Prophet()
    model.fit(dfsplit) #the algo runs at the closing time of the current candle which is included in the fit
    future = model.make_future_dataframe(periods=frontpredictions, include_history=False)
    forecast = model.predict(future)
    if signal:
        if(forecast.yhat.mean()-dfsplit.close.iat[-1]<diff_limit):
            return 1
        elif(forecast.yhat.mean()-dfsplit.close.iat[-1]>diff_limit):
            return 2
        else:
            return 0
    else:
        forecast["y_target"] = df['y'].iloc[l+1:l+frontpredictions+1]
        forecast = forecast[['yhat', 'yhat_lower', 'yhat_upper']].values[0]
        return forecast[0],forecast[1],forecast[2] 

prophet_signal(df, 200, 100, frontpredictions=1, diff_limit=0.008, signal=False)


from tqdm import tqdm
backcandles=100
frontpredictions=1

yhatlow = [0 for i in range(len(df))]
yhat = [0 for i in range(len(df))]
yhathigh = [0 for i in range(len(df))]

for row in tqdm(range(backcandles, len(df)-frontpredictions)):
    prophet_pred = prophet_signal(df, row, backcandles, frontpredictions=1, diff_limit=0.005, signal=False)
    yhat[row] = prophet_pred[0]
    yhatlow[row] = prophet_pred[1]
    yhathigh[row] = prophet_pred[2]

df["yhat"] = yhat
df["yhatlow"] = yhatlow
df["yhathigh"] = yhathigh

df['yhatlow'] = df['yhatlow'].shift(+1)
df['yhathigh'] = df['yhathigh'].shift(+1)
df['yhat'] = df['yhat'].shift(+1)

from matplotlib import pyplot as plt

x1 = df.shape[0] - 100
x2 = df.shape[0]
st.subheader('Facebook Prophet Predictions')
fig3 = plt.figure(figsize=(18,6))
plt.plot(df['ds'].iloc[x1:x2],df['yhat'].iloc[x1:x2], label="Prediction", marker='o')
plt.fill_between(df['ds'].iloc[x1:x2], df['yhatlow'].iloc[x1:x2], df['yhathigh'].iloc[x1:x2], color='b', alpha=.1)
plt.plot(df['ds'].iloc[x1:x2], df['y'].iloc[x1:x2], label = "Market", marker='o')
plt.legend(loc="upper left")
st.pyplot(fig3)