import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas_ta as ta
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
start_input = st.text_input('Enter Start Date', '2015-01-01')
end_input = st.text_input('Enter End Date', '2023-12-12')
df = yf.download(user_input, start_input, end_input)
df = df.reset_index()  

#Describing Data
st.subheader(user_input + ' Data from ' + start_input + ' to ' + end_input)
st.dataframe(df.describe(), use_container_width=True)
st.dataframe(df, use_container_width=True)

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
plt.plot(df.Date, df.Close, 'b')
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

y_test = np.reshape(y_test, (-1,))
y_predicted = np.reshape(y_predicted, (-1,))
##
lstm_predictions = pd.DataFrame(columns=['Date', 'Original Price', 'Predicted Price'])  # DataFrame to be updated

start_index = len(df) - len(y_predicted)
end_index = len(df) - 1  # Subtract 1 to include the last date

# Adjust the length of input_data to match the desired range
input_data_adjusted = input_data[-(end_index-start_index+1):]

# Assign values from the 'Date' column of df to the 'date' column of df2
lstm_predictions['Date'] = df['Date'].values[start_index:end_index+1]

lstm_predictions['Original Price'] = pd.DataFrame(y_test)
lstm_predictions['Predicted Price'] = pd.DataFrame(y_predicted)

# Convert the 'date' column to datetime type
lstm_predictions['Date'] = pd.to_datetime(lstm_predictions['Date'])

lstm_predictions.set_index('Date')

###

#Final Graph
st.subheader('[LSTM] Predictions vs Original')
#lstm_predictions = pd.DataFrame({'Original Price': y_test, 'Predicted Price': y_predicted})
st.dataframe(lstm_predictions, use_container_width=True)

fig4 = plt.figure(figsize=(18,6))
plt.plot(lstm_predictions.Date, lstm_predictions['Predicted Price'], label="Prediction", marker='o')
plt.plot(lstm_predictions.Date, lstm_predictions['Original Price'], label = "Real", marker='o')
plt.legend(loc="upper left")
st.pyplot(fig4)

fig2 = plt.figure(figsize=(12,6))
plt.plot(lstm_predictions.Date, y_test, 'b', label= 'Original Price')
plt.plot(lstm_predictions.Date, y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



#################  PECNET    #################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
#from keras.models import load_model
import plotly.graph_objects as go
from datetime import datetime
import pandas_ta as ta

import pandas as pd
from datetime import timedelta
import csv

# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
from keras import backend as K
from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import plot_model 
from keras import optimizers
from keras.layers.core import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM

import math
import pywt

import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import datetime

# computing averages
def yy5(input_data):
    A=0
    B=0
    C=0
    D=0
    outputavg = []    
    for X in input_data:
        Y=(X+A+B+C+D)/5
        outputavg.append(Y)
        D=C
        C=B
        B=A
        A=X

    return outputavg


#construction of outputs
def output(inputdata):
    out=[]
    for i in range(7, len(inputdata)-1):
        out.append(inputdata[i+1])
    out = np.append(out, [np.nan])
    return out

#successive values 
def successive(successive):

    input_data=[]
    for i in range(7, len(successive)):

        input_data.append([successive[i-3]]+[successive[i-2]]+[successive[i-1]]+[successive[i]])
    return input_data  

#wavelet transform
def four_wavelets(training):
    input_data=np.array(training)
    days = input_data[:,0:4]


    for row in input_data:
            (a, d) = pywt.dwt(days, 'haar')
            (a2,d2)=pywt.dwt(a, 'haar') 
            l3=np.append(a2,d2, axis=1)
            l2_3=np.append(l3,d, axis=1)
            transformed_df=l2_3

    training=transformed_df


    return training

def pecnet():
    #network configurations
    hidden1=32
    second_layer1=32
    third_layer1=32
    forth_layer1=16
    hidden2=32
    second_layer2=32
    third_layer2=32
    forth_layer2=16
    hidden3=32
    second_layer3=32
    third_layer3=32
    forth_layer3=16
    hidden4=32
    second_layer4=32
    third_layer4=32
    forth_layer4=16
    hidden5=32
    second_layer5=32
    third_layer5=32
    forth_layer5=16



    #calling input files
    #input_data=pd.read_csv('bistclose.csv')
    #input_data=pd.read_csv('close_is.csv')
    #input_data=pd.read_csv('open.csv')
    #input_data=pd.read_csv('close.csv')
    #input_data=pd.read_csv('min.csv')
    #input_data=pd.read_csv('max.csv')
    input_data=pd.read_csv('/home/oguz/Desktop/GradProject-main/Web Project for Predicted Results/file1.csv')


    #construction of input arrays
    input_data=np.array(input_data)
    input_data=input_data.reshape(input_data.shape[0])
    input_data=list(input_data)
    input_data=np.array(input_data)


    average=yy5(input_data)
    input_data_average=successive(average)
    input_data_successive=successive(input_data)
    out=output(input_data)
    
    #%matplotlib notebook

    #fig,ax= plt.subplots()
    #ax.plot(input_data, label='daily_input')
    #ax.plot(average, label='average_input')
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
    #plt.title('Input data')
    #plt.show();


    #division of data set into training and test data set
    N=len(input_data)
    division_of_training=0.95
    input_train=input_data_average[:int(N*division_of_training)]
    input_test=input_data_average[int(N*division_of_training):int(N*1)]

    successive_train=input_data_successive[:int(N*division_of_training)]
    successive_test=input_data_successive[int(N*division_of_training):int(N*1)]

    second_input_train=successive_train 
    second_input_test=successive_test 

    output_train= out[:int(N*division_of_training)]
    output_test=out[int(N*division_of_training):int(N*1)]



    #normalization
    inputiavg=np.array(input_train)
    inputiavgt=np.array(input_test)

    inputsuc=np.array(second_input_train)
    inputsuct=np.array(second_input_test)

    subtraction_average_train=inputiavg
    subtraction_average_test=inputiavgt

    subtraction_successive_train=inputsuc
    subtraction_successive_test=inputsuct

    subtraction_average_train=subtraction_average_train.sum(axis=1)/4
    subtraction_average_test=subtraction_average_test.sum(axis=1)/4

    subtraction_successive_train=subtraction_successive_train.sum(axis=1)/4
    subtraction_successive_test=subtraction_successive_test.sum(axis=1)/4

    #normalization of inputs
    first_input_train=input_train-subtraction_average_train[:, None]
    first_input_test=input_test-subtraction_average_test[:,None]

    output_train=output_train-subtraction_successive_train
    output_test=output_test-subtraction_successive_test

    second_input_train=second_input_train-subtraction_successive_train[:,None]
    second_input_test=second_input_test-subtraction_successive_test[:,None]


    #4inputs WT
    final_first_w_input_train=four_wavelets(first_input_train)
    final_first_w_input_test=four_wavelets(first_input_test)

    X_train=np.array(final_first_w_input_train[:, 1:])
    y_train=np.array(output_train)

    X_test=np.array(final_first_w_input_test[:,1:])
    y_test=np.array(output_test)

    m_primary=len(X_train[0,:])
    p_primary=np.size(y_train[0])
    N_primary=len(X_train)

    model= Sequential ([
        Dense(hidden1, input_dim=m_primary, activation='relu'), 
        Dropout(0.1),
        Dense(second_layer1), #,activation='relu'),
        Dropout(0.1),
        Dense(third_layer1), #,activation='relu'),
        Dropout(0.1),
        Dense(forth_layer1), #,activation='relu'),
        Dropout(0.1),
        Dense(p_primary)
        ])

    model.summary()

    sgd=SGD(learning_rate=0.05,momentum=0.75, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_absolute_error','mean_squared_logarithmic_error','cosine_similarity','logcosh'])
    history1=model.fit(X_train, y_train, batch_size=N_primary, epochs=300, shuffle=False, verbose=0)  

    predicted_train = model.predict(X_train) 
    predicted_train = np.reshape(predicted_train, (predicted_train.size,))
    error_train1=predicted_train-y_train

    predicted_test = model.predict(X_test) 
    predicted_test = np.reshape(predicted_test, (predicted_test.size,))
    error_test1=predicted_test-y_test


    # Second NN, error forecasting network 
    error_train=pd.DataFrame(error_train1)
    add_train=four_wavelets(second_input_train) 

    X_error_train1=np.array(add_train[:, 1:])
    y_error_train1=np.array(error_train)

    error_test=pd.DataFrame(error_test1)
    add_test=four_wavelets(second_input_test) 

    X_error_test1=np.array(add_test[:, 1:])

    m_second=len(X_error_train1[0,:])
    p_second=np.size(y_train[0])
    N_second=len(X_error_train1)

    error_model1= Sequential ([
        Dense(hidden2, input_dim=m_second, activation='relu'), 
        Dropout(0.1),
        Dense(second_layer2), #,activation='relu'),
        Dropout(0.1),
        Dense(third_layer2), #,activation='relu'),
        Dropout(0.1),
        Dense(forth_layer2), #,activation='relu'),
        Dropout(0.1),
        Dense(p_second)
    ])

    error_model1.summary()

    sgd=SGD(learning_rate=0.05, momentum=0.75, decay=0.0, nesterov=False)
    error_model1.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse','mae','accuracy'])
    history3=error_model1.fit(X_error_train1, y_error_train1, batch_size=N_second, epochs=300, shuffle=False, verbose=0)

    error_predicted_tr = error_model1.predict(X_error_train1)
    error_predicted_tr = np.reshape(error_predicted_tr, (error_predicted_tr.size,))
    error_predicted_tes = error_model1.predict(X_error_test1)
    error_predicted_tes = np.reshape(error_predicted_tes, (error_predicted_tes.size,))

    compensated1_train=(predicted_train+subtraction_successive_train)-(error_predicted_tr)
    compensated1_test=(predicted_test+subtraction_successive_test)-(error_predicted_tes)


    # Third NN, error network 
    error_train2a=compensated1_train-(y_train+subtraction_successive_train)
    error_test2a=compensated1_test-(y_test+subtraction_successive_test)

    error_train2=pd.DataFrame(error_train2a)
    error_train2 [1]= error_train2[0].shift(1)
    error_train2 [2]=error_train2[1].shift(1)
    error_train2 [3]=error_train2[2].shift(1)
    error_train2[4]=error_train2[3].shift(1)
    error_train2 = error_train2.replace(np.nan, 0)

    ##error normalization
    subtraction_error_train2=np.array(error_train2)
    subtraction_error_train2=subtraction_error_train2[:,:-1]
    subtraction_error_train2=subtraction_error_train2.sum(axis=1)/4

    error_train2=error_train2-subtraction_error_train2[:, None]


    error_train2=np.array(error_train2)
    days_train = error_train2[:,1:5]
    input3_train=four_wavelets(days_train)
    output3_train=error_train2[:,0:1]

    X_error_train2=np.array(input3_train[:, 1:])
    y_error_train2=np.array(output3_train)

    error_test2=pd.DataFrame(error_test2a)
    error_test2 [1]= error_test2[0].shift(1)
    error_test2 [2]=error_test2[1].shift(1)
    error_test2 [3]=error_test2[2].shift(1)
    error_test2[4]=error_test2[3].shift(1)
    error_test2 = error_test2.replace(np.nan, 0)

    subtraction_error_test2=np.array(error_test2)
    subtraction_error_test2=subtraction_error_test2[:,:-1]
    subtraction_error_test2=subtraction_error_test2.sum(axis=1)/4

    error_test2=error_test2-subtraction_error_test2[:,None]

    error_test2=np.array(error_test2)
    days_test = error_test2[:,1:5]
    input3_test=four_wavelets(days_test)
    output3_test=error_test2[:,0:1]

    X_error_test2=np.array(input3_test[:, 1:])


    #####3rd NN
    m_error=len(X_error_train2[0,:])
    p_error=np.size(y_error_train2[0])
    N_error=len(X_error_train2)



    error_model2= Sequential ([
        Dense(hidden3, input_dim=m_error, activation='relu'), 
        Dropout(0.1),
        Dense(second_layer3), #,activation='relu'),
        Dropout(0.1),
        Dense(third_layer3), #,activation='relu'),
        Dropout(0.1),
        Dense(forth_layer3), #,activation='relu'),
        Dropout(0.1),
        Dense(p_error)
    ])

    error_model2.summary()

    sgd=SGD(learning_rate=0.05, momentum=0.75, decay=0.0, nesterov=False)
    error_model2.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse','mae','accuracy'])
    history4=error_model2.fit(X_error_train2, y_error_train2, batch_size=N_error, epochs=300, shuffle=False, verbose=0)


    error_predicted_tr2 = error_model2.predict(X_error_train2)
    error_predicted_tr2 = np.reshape(error_predicted_tr2, (error_predicted_tr2.size,))
    error_predicted_tes2 = error_model2.predict( X_error_test2)
    error_predicted_tes2= np.reshape(error_predicted_tes2, (error_predicted_tes2.size,))

    compensated_y_train=compensated1_train-(error_predicted_tr2+subtraction_error_train2)
    compensated_y_test=compensated1_test-(error_predicted_tes2+subtraction_error_test2)


    # Final NN 
    error_predicted_tr3=error_predicted_tr2+subtraction_error_train2
    error_predicted_tes3=error_predicted_tes2+subtraction_error_test2

    training_final_add=np.column_stack((predicted_train, error_predicted_tr))
    training_final_add=np.column_stack((training_final_add,error_predicted_tr3))

    test_final_add=np.column_stack((predicted_test, error_predicted_tes))
    test_final_add=np.column_stack((test_final_add,error_predicted_tes3))

    ####final NN
    m_final=len(training_final_add[0,:])
    p_final=np.size(y_train[0])
    N_final=len(training_final_add)

    final_model= Sequential ([
        Dense(hidden4, input_dim=m_final, activation='relu'), 
    #    Dropout(0.1),
    #    Dense(second_layer4), #,activation='relu'),
    #    Dropout(0.1),
    #    Dense(third_layer4), #,activation='relu'),
    #    Dropout(0.1),
    #    Dense(forth_layer4), #,activation='relu'),
    #    Dropout(0.1),
        Dense(p_final)
    ])

    final_model.summary()

    sgd=SGD(learning_rate=0.05, momentum=0.75, decay=0.0, nesterov=False)
    final_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse','mae','accuracy'])
    final_history=final_model.fit(training_final_add, y_train, batch_size=N_final, epochs=300, shuffle=False, verbose=0)


    final_predicted_tr =final_model.predict(training_final_add)
    final_predicted_tr = np.reshape(final_predicted_tr, (final_predicted_tr.size,))
    final_predicted_tes = final_model.predict(test_final_add)
    final_predicted_tes = np.reshape(final_predicted_tes, (final_predicted_tes.size,))



    # errors
    EPSILON =  1e-10

    y_train=y_train+subtraction_successive_train
    final_y_train=final_predicted_tr+subtraction_successive_train
    final_y_train = np.reshape(final_y_train, (final_y_train.size,))

    final_error_train=final_y_train-y_train
    final_rmse_error_train=np.sqrt(sum(final_error_train*final_error_train)/len(final_error_train))
    final_mse_train=(sum(final_error_train*final_error_train)/len(final_error_train))
    final_mape_train=100*sum(abs(final_error_train/y_train))/len(y_train)
    final_mae_train=sum(abs(final_error_train-y_train))/len(y_train)
    final_rmspe_train=100*np.sqrt(np.nanmean(np.square(((y_train - final_y_train) / (y_train+ EPSILON)))))


    y_test=y_test+subtraction_successive_test

    final_y_test=final_predicted_tes+subtraction_successive_test
    y_test = np.reshape(y_test, (y_test.size,))
    final_y_test = np.reshape(final_y_test, (final_y_test.size,))


    #final_error_test=y_test[:-1]-final_predicted_tes[:-1]
    final_error_test=final_y_test[:-1]-y_test[:-1] 
    final_rmse_error_test=np.sqrt(sum(final_error_test*final_error_test)/len(final_error_test))
    final_mse_test=(sum(final_error_test*final_error_test)/len(final_error_test))
    final_mape_test=100*sum(abs(final_error_test/y_test[:-1]))/len(y_test-1)
    final_mae_test=sum(abs(final_error_test-y_test[:-1]))/len(y_test-1)
    final_rmspe_test=100*np.sqrt(np.nanmean(np.square(((y_test[:-1] - final_y_test[:-1]) / (y_test[:-1]+ EPSILON)))))

    #errors of the first nn
    predicted_train=predicted_train+subtraction_successive_train
    predicted_test=predicted_test+subtraction_successive_test

    predicted_error_train=predicted_train-y_train
    predicted_rmse_error_train=np.sqrt(sum(predicted_error_train*predicted_error_train)/len(predicted_error_train))
    predicted_mse_train=(sum(predicted_error_train*predicted_error_train)/len(predicted_error_train))
    predicted_mape_train=100*sum(abs(predicted_error_train/y_train))/len(y_train)
    predicted_mae_train=sum(abs(predicted_error_train-y_train))/len(y_train)
    predicted_rmspe_train=100*np.sqrt(np.nanmean(np.square(((y_train - predicted_train) /(y_train+ EPSILON)))))

    predicted_error_test=predicted_test[:-1]-y_test[:-1]
    predicted_rmse_error_test=np.sqrt(sum(predicted_error_test*predicted_error_test)/len(predicted_error_test))
    predicted_mse_test=(sum(predicted_error_test*predicted_error_test)/len(predicted_error_test))
    predicted_mape_test=100*sum(abs(predicted_error_test/y_test[:-1]))/len(y_test-1)
    predicted_mae_test=sum(abs(predicted_error_test-y_test[:-1]))/len(y_test-1)
    predicted_rmspe_test=100*np.sqrt(np.nanmean(np.square(((y_test[:-1] - predicted_test[:-1]) / (y_test[:-1]+ EPSILON)))))

    #errors of the second nn
    compensated1_train_error=compensated1_train-y_train

    compensated1_train_rmse_error_train=np.sqrt(sum(compensated1_train_error*compensated1_train_error)/len(compensated1_train_error))
    compensated1_train_mse_train=(sum(compensated1_train_error*compensated1_train_error)/len(compensated1_train_error))
    compensated1_train_mape_train=100*sum(abs(compensated1_train_error/y_train))/len(y_train)
    compensated1_train_mae_train=sum(abs(compensated1_train_error-y_train))/len(y_train)
    compensated1_train_rmspe_train=np.sqrt(np.nanmean(np.square(((y_train - compensated1_train) /(y_train+ EPSILON)))))*100

    compensated1_test_error=compensated1_test[:-1]-y_test[:-1]

    compensated1_test_rmse_error_test=np.sqrt(sum(compensated1_test_error*compensated1_test_error)/len(compensated1_test_error))
    compensated1_test_mse_test=(sum(compensated1_test_error*compensated1_test_error)/len(compensated1_test_error))
    compensated1_test_mape_test=100*sum(abs(compensated1_test_error/y_test[:-1]))/len(y_test-1)
    compensated1_test_mae_test=sum(abs(compensated1_test_error-y_test[:-1]))/len(y_test-1)
    compensated1_test_rmspe_test=np.sqrt(np.nanmean(np.square(((y_test[:-1] - compensated1_test[:-1]) / (y_test[:-1]+ EPSILON)))))*100

    #errors of the third nn
    compensated_error_train=compensated_y_train-y_train

    comp_rmse_error_train=np.sqrt(sum(compensated_error_train*compensated_error_train)/len(compensated_error_train))
    comp_mse_train=(sum(compensated_error_train*compensated_error_train)/len(compensated_error_train))
    comp_mape_train=100*sum(abs(compensated_error_train/y_train))/len(y_train)
    comp_mae_train=sum(abs(compensated_error_train-y_train))/len(y_train)
    comp_rmspe_train=np.sqrt(np.nanmean(np.square(((y_train - compensated_y_train) / (y_train+ EPSILON)))))*100

    compensated_error_test=compensated_y_test[:-1]-y_test[:-1]

    comp_rmse_error_test=np.sqrt(sum(compensated_error_test*compensated_error_test)/len(compensated_error_test))
    comp_mse_test=(sum(compensated_error_test*compensated_error_test)/len(compensated_error_test))
    comp_mape_test=100*sum(abs(compensated_error_test/y_test[:-1]))/len(y_test-1)
    comp_mae_test=sum(abs(compensated_error_test-y_test[:-1]))/len(y_test-1)
    comp_rmspe_test=np.sqrt(np.nanmean(np.square(((y_test[:-1] - compensated_y_test[:-1]) / (y_test[:-1]+ EPSILON)))))*100

    zz_rmse_errors_ttrain=(predicted_rmse_error_train,compensated1_train_rmse_error_train, comp_rmse_error_train,final_rmse_error_train)
    zz_rmse_errors_test=(predicted_rmse_error_test,compensated1_test_rmse_error_test, comp_rmse_error_test,final_rmse_error_test)

    zz_rmspe_errors_ttrain=(predicted_rmspe_train,compensated1_train_rmspe_train, comp_rmspe_train,final_rmspe_train)
    zz_rmspe_errors_test=(predicted_rmspe_test,compensated1_test_rmspe_test, comp_rmspe_test,final_rmspe_test)

    zz_mape_errors_ttrain=(predicted_mape_train,compensated1_train_mape_train, comp_mape_train,final_mape_train)
    zz_mape_errors_test=(predicted_mape_test,compensated1_test_mape_test, comp_mape_test,final_mape_test)

    zz_mae_errors_ttrain=(predicted_mae_train,compensated1_train_mae_train, comp_mae_train,final_mae_train)
    zz_mae_errors_test=(predicted_mae_test,compensated1_test_mae_test, comp_mae_test,final_mae_test)

    zz_predictions_train = (y_train, predicted_train,compensated1_train,  compensated_y_train, final_y_train)
    zz_predictions_test = (y_test,predicted_test,compensated1_test, compensated_y_test, final_y_test)
    
    return final_y_test, final_y_train, input_data



df = yf.download('AAPL', '2015-01-01', '2023-12-12')
df = df.reset_index()

frame = df
frame = frame.iloc[:,4:5]
#frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
#frame.set_index('Time', inplace=True)
#frame.index = pd.to_datetime(frame.index, unit='ms')

frame.to_csv('/home/oguz/Desktop/GradProject-main/Web Project for Predicted Results/file1.csv', float_format='%.6f', header=False, index=False)
final_y_test, final_y_train, input_data = pecnet()
print("Prediction is:",final_y_test[-1])

df2 = pd.DataFrame(columns=['date', 'real', 'prediction'])  # DataFrame to be updated

start_index = len(df) - len(final_y_test)
end_index = len(df) - 1  # Subtract 1 to include the last date

# Adjust the length of input_data to match the desired range
input_data_adjusted = input_data[-(end_index-start_index+1):]

# Assign values from the 'Date' column of df to the 'date' column of df2
df2['date'] = df['Date'].values[start_index:end_index+1]

# Assign values to the 'real' column of df2
df2['real'] = input_data_adjusted

# Assign values to the 'prediction' column of df2
df2['prediction'] = final_y_test

df2.loc[max(df2.index)+1, :] = None

df2['prediction'] = df2.prediction.shift(1)
# Convert the 'date' column to datetime type
df2['date'] = pd.to_datetime(df2['date'])

# Find the last valid date in the 'date' column
last_valid_date = df2['date'].dropna().iloc[-1]

# Fill the NaN value in the 'date' column with the next day after the last valid date
df2['date'].fillna(last_valid_date + pd.DateOffset(days=1), inplace=True)

#Visualizations

st.subheader('[PECNET] Real vs Predicted in Test Data')
fig4 = plt.figure(figsize=(18,6))
plt.plot(df2.date, df2['prediction'], label="Prediction", marker='o')
plt.plot(df2.date, df2['real'], label = "Real", marker='o')
plt.legend(loc="upper left")
st.pyplot(fig4)


#####################BACKTEST##############################
dict = {'date': 'date', 'real': 'price', 'prediction': 'pred'}

df2.rename(columns=dict, inplace=True)
df2.drop(index=df2.index[0], axis=0, inplace=True)
df2.to_csv('backtest.csv', index=False, encoding='utf-8')





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta
plt.style.use("seaborn")

class IterativeBase():

    def __init__(self, symbol, start, end, amount, use_prediction = True):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_balance = amount
        self.current_balance = amount
        self.units = 0
        self.trades = 0
        self.use_prediction = use_prediction
        self.get_data()
        self.create_positions()

    def create_positions(self):
        pos = pd.DataFrame(columns=["date","price","pos_changed","current_balance"])
        pos.set_index("date")
        self.positions = pos
        
    def add_position(self, date, price, pos_changed, current_balance):
        df_new_row = pd.DataFrame([{ 'date':date,'price':price, 'pos_changed':pos_changed, 'current_balance':current_balance }])
        self.positions = pd.concat([self.positions, df_new_row], ignore_index=True)

    def get_data(self):
        raw = pd.read_csv("backtest.csv", parse_dates = ["date"], index_col = "date", engine='python').dropna()
        raw = raw.loc[self.start:self.end]
        #raw["returns"] = np.log(raw.price / raw.price.shift(1))
        self.data = raw

    def plot_data(self, cols = None):  
        if cols is None:
            cols = "price"
        self.data[cols].plot(figsize = (15, 8), title = self.symbol + "PRICE")
        #fig = px.line(self.data, x=self.data.index, y=cols, title="ISCTR - Close Prices")  # creating a figure using px.line
        #fig.show()
    
    def get_values(self, bar):
        date = str(self.data.index[bar].date())
        price = round(self.data.price.iloc[bar], 5)
        prediction = round(self.data.pred.iloc[bar], 5)
        return date, price, prediction
    
    def print_current_balance(self, bar):
        date, price, prediction = self.get_values(bar)
        print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))
        
        
    def buy_instrument(self, bar, units = None, amount = None):
        date, price, prediction = self.get_values(bar)
        #if self.use_spread:
        #    price += spread/2 # ask price
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance -= units * price # reduce cash balance by "purchase price"
        self.units += units
        self.trades += 1
        print("{} |  Buying {} for {}".format(date, units, round(price, 5)))
    
    def sell_instrument(self, bar, units = None, amount = None):
        date, price, prediction = self.get_values(bar)
        #if self.use_spread:
        #    price -= spread/2 # bid price
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance += units * price # increases cash balance by "purchase price"
        self.units -= units
        self.trades += 1
        print("{} |  Selling {} for {}".format(date, units, round(price, 5)))
    
    def print_current_position_value(self, bar):
        date, price, prediction = self.get_values(bar)
        cpv = self.units * price
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))
    
    def print_current_nav(self, bar):
        date, price, prediction = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))
        
    def get_current_nav(self, bar):
        date, price, prediction = self.get_values(bar)
        nav = self.current_balance + self.units * price
        return round(nav, 2)
        
    def close_pos(self, bar):
        date, price, prediction = self.get_values(bar)
        print(75 * "-")
        print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price # closing final position (works with short and long!)
        #self.current_balance -= (abs(self.units) * spread/2 * self.use_spread) # substract half-spread costs
        print("{} | closing position of {} for {}".format(date, self.units, price))
        self.units = 0 # setting position to neutral
        self.trades += 1
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        self.print_current_balance(bar)
        print("{} | net performance (%) = {}".format(date, round(perf, 2) ))
        print("{} | number of trades executed = {}".format(date, self.trades))
        print(75 * "-")
        
    def plot_pnl_price(self):
        subfig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # plotting close prices with bollinger bands
        fig = px.line(self.positions, x='date', y='current_balance')
        fig2 = px.line(self.positions, x='date', y='price')
        
        fig2.update_traces(yaxis="y2")

        # adding trades to plots
        for bar in range(len(self.positions)-1):
            color = "green" if self.positions["current_balance"].iloc[bar+1] >= self.positions["current_balance"].iloc[bar] else "red"
            fig.add_shape(type="line",
                        x0=self.positions["date"].iloc[bar], y0=self.positions["current_balance"].iloc[bar],
                        x1=self.positions["date"].iloc[bar+1], y1=self.positions["current_balance"].iloc[bar+1],
                        line_color=color, line_width=3)
        
        subfig.layout.xaxis.title = "Time"
        subfig.layout.yaxis.title = "PNL(Balance)"
        subfig.layout.yaxis2.type = "log"
        subfig.layout.yaxis2.title = "Price"
        
        subfig.add_traces(fig.data + fig2.data)
        subfig.for_each_trace(lambda t: t.update(line_color=t.marker.color))
        
        st.plotly_chart(subfig)
        
    def plot_pretty_balance(self):
        # plotting close prices with bollinger bands
        fig = px.line(self.positions, x='date', y='current_balance')

        # adding trades to plots
        for bar in range(len(self.positions)-1):
            if self.positions["current_balance"].iloc[bar+1] >= self.positions["current_balance"].iloc[bar]:
                fig.add_shape(type="line",
                    x0=self.positions["date"].iloc[bar], y0=self.positions["current_balance"].iloc[bar], x1=self.positions["date"].iloc[bar+1], y1=self.positions["current_balance"].iloc[bar+1],
                    line_color="green", line_width=3)
            else:
                fig.add_shape(type="line",
                    x0=self.positions["date"].iloc[bar], y0=self.positions["current_balance"].iloc[bar], x1=self.positions["date"].iloc[bar+1], y1=self.positions["current_balance"].iloc[bar+1],
                    line_color="red", line_width=3)
        
        st.plotly_chart(fig)
    
    def plot_pretty_price(self):
        # plotting close prices with bollinger bands
        fig = px.line(self.positions, x='date', y='price')

        # adding trades to plots
        for bar in range(len(self.positions)-1):
            if self.positions["current_balance"].iloc[bar+1] >= self.positions["current_balance"].iloc[bar]:
                fig.add_shape(type="line",
                    x0=self.positions["date"].iloc[bar], y0=self.positions["price"].iloc[bar], x1=self.positions["date"].iloc[bar+1], y1=self.positions["price"].iloc[bar+1],
                    line_color="green", line_width=3)
            else:
                fig.add_shape(type="line",
                    x0=self.positions["date"].iloc[bar], y0=self.positions["price"].iloc[bar], x1=self.positions["date"].iloc[bar+1], y1=self.positions["price"].iloc[bar+1],
                    line_color="red", line_width=3)
        
        st.plotly_chart(fig)
    
    def plot_prettier(self):
        df = pd.read_csv('backtest.csv')
        # plotting close prices with bollinger bands
        #print(df)
        fig = px.line(df, x='date', y='price')
        
        # adding trades to plots
        for bar in range(len(self.positions)-1):
            if self.positions["current_balance"].iloc[bar+1] >= self.positions["current_balance"].iloc[bar]:
                fig.add_shape(type="line",
                    x0=self.positions["date"].iloc[bar], y0=self.positions["price"].iloc[bar], x1=self.positions["date"].iloc[bar+1], y1=self.positions["price"].iloc[bar+1],
                    line_color="green", line_width=3)
            else:
                fig.add_shape(type="line",
                    x0=self.positions["date"].iloc[bar], y0=self.positions["price"].iloc[bar], x1=self.positions["date"].iloc[bar+1], y1=self.positions["price"].iloc[bar+1],
                    line_color="red", line_width=3)
                
        st.plotly_chart(fig)
    

class IterativeBacktest(IterativeBase):

    # helper method
    def go_long(self, bar, units = None, amount = None):
        if self.position == -1:
            self.buy_instrument(bar, units = -self.units) # if short position, go neutral first
        if units:
            self.buy_instrument(bar, units = units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.buy_instrument(bar, amount = amount) # go long
        date, price, prediction = self.get_values(bar)
        nav = self.get_current_nav(bar)
        self.add_position(date, price, 1, nav)

    # helper method
    def go_short(self, bar, units = None, amount = None):
        if self.position == 1:
            self.sell_instrument(bar, units = self.units) # if long position, go neutral first
        if units:
            self.sell_instrument(bar, units = units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.sell_instrument(bar, amount = amount) # go short
        date, price, prediction = self.get_values(bar)
        nav = self.get_current_nav(bar)
        self.add_position(date, price, -1, nav)

    def test_sma_strategy(self, SMA_S, SMA_L):
        
        # nice printout
        stm = "Testing SMA strategy | {} | SMA_S = {} & SMA_L = {}".format(self.symbol, SMA_S, SMA_L)
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data() # reset dataset
        
        # prepare data
        self.data["SMA_S"] = self.data["price"].rolling(SMA_S).mean()
        self.data["SMA_L"] = self.data["price"].rolling(SMA_L).mean()
        self.data.dropna(inplace = True)

        # sma crossover strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            if self.data["SMA_S"].iloc[bar] > self.data["SMA_L"].iloc[bar]: # signal to go long
                if self.position in [0, -1]:
                    self.go_long(bar, amount = "all") # go long with full amount
                    self.position = 1  # long position
                    #self.trades +=1
            elif self.data["SMA_S"].iloc[bar] < self.data["SMA_L"].iloc[bar]: # signal to go short
                if self.position in [0, 1]:
                    self.go_short(bar, amount = "all") # go short with full amount
                    self.position = -1 # short position
                    #self.trades +=1
        self.close_pos(bar+1) # close position at the last bar
        

    def test_my_strategy(self):
        
        # nice printout
        ma = "Testing PECNET Results on | {} ".format(self.symbol)
        st.write(ma)
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.get_data() # reset dataset
        
        # my strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            print("*" * 75)
            self.print_current_nav(bar)
            if self.data["pred"].iloc[bar+1] > self.data["price"].iloc[bar]: # signal to go long
                if self.position in [0, -1]:
                    self.go_long(bar, amount = "all") # go long with full amount
                    self.position = 1  # long position
                    self.print_current_position_value(bar)
                    #self.trades +=1
            elif self.data["pred"].iloc[bar+1] < self.data["price"].iloc[bar]: # signal to go short
                if self.position in [0, 1]:
                    self.go_short(bar, amount = "all") # go short with full amount
                    self.position = -1 # short position
                    self.print_current_position_value(bar)
                    #self.trades +=1
        #self.close_pos(bar+1) # close position at the last bar
        #self.positions["returns"] = positions.current_balance - positions.current_balance.shift(1)

st.subheader('Backtest Results')

bc = IterativeBacktest('APPL', "2023-02-23", "2023-07-12", 10000, use_prediction = True)

bc.test_my_strategy()

st.dataframe(bc.data, use_container_width=True)

st.dataframe(bc.positions, use_container_width=True)

bc.plot_data()

px.line(bc.positions, x='date', y=['current_balance'])

bc.plot_pretty_balance()

bc.plot_pnl_price()

bc.plot_prettier()


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



