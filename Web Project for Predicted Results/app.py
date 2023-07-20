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
from datetime import date, timedelta
from pecnet import *
from backtester import *

def home_page(df):
    st.title("Stock Details")
    # Your content for the home page goes here...
    df=df
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

def lstm_page(df):
    # Your content for the data page goes here...
    st.title('Stock Trend Prediction with LSTM')

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



def facebook_prophet_page(df):
    st.title("Facebook Prophet")
    # Your content for the settings page goes here...
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




def pecnet_page(df):
    st.title("PECNET")
    # Your content for the settings page goes here...
    #################  PECNET    #################
    
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

    dict = {'date': 'date', 'real': 'price', 'prediction': 'pred'}
    df2.rename(columns=dict, inplace=True)
    df2.drop(index=df2.index[0], axis=0, inplace=True)
    df2.to_csv('backtest_pecnet.csv', index=False, encoding='utf-8')



def backtest_page(df):
    st.header('Backtest Results')
    st.subheader('Pecnet Test Results')

    bc = IterativeBacktest('APPL', "2020-01-01", "2023-12-12", 10000, use_prediction = True, csv_file_path = "backtest_pecnet.csv")

    bc.test_my_strategy()

    st.dataframe(bc.data, use_container_width=True)

    st.dataframe(bc.positions, use_container_width=True)

    bc.plot_data()

    px.line(bc.positions, x='date', y=['current_balance'])

    bc.plot_pretty_balance()

    bc.plot_pnl_price()

    bc.plot_prettier()



def main():
    st.sidebar.title("AI Based Trading Application")
    page = st.sidebar.radio("Select a page:", ("Stock Details", "LSTM Results", "Facebook Prophet", "PECNET", "Backtest Results"))

    # Use st.sidebar.text_input for the Stock Ticker
    user_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')

    # Use st.sidebar.date_input for the Start Date
    start_date = date(2015, 1, 1)
    start_input = st.sidebar.date_input('Enter Start Date', start_date)

    # Use st.sidebar.date_input for the End Date
    end_date = date(2023, 12, 12)
    end_input = st.sidebar.date_input('Enter End Date', end_date)

    # Convert date objects to strings before concatenating with other strings
    start_input = start_input.strftime("%Y-%m-%d")
    end_input = end_input.strftime("%Y-%m-%d")

    df = yf.download(user_input, start_input, end_input)
    df = df.reset_index()  

    #Describing Data
    st.subheader(user_input + ' Data from ' + start_input + ' to ' + end_input)
    st.dataframe(df.describe(), use_container_width=True)
    st.dataframe(df, use_container_width=True)

    if page == "Stock Details":
        st.title("Stock Details Page")
        # Call the respective page function and display a "Processing" notification
        with st.spinner("Processing..."):
            home_page(df)
        st.success("Processing complete!")

    elif page == "LSTM Results":
        st.title("LSTM Results Page")
        # Call the respective page function and display a "Processing" notification
        with st.spinner("Processing..."):
            lstm_page(df)
        st.success("Processing complete!")

    elif page == "Facebook Prophet":
        st.title("Facebook Prophet Page")
        # Call the respective page function and display a "Processing" notification
        with st.spinner("Processing..."):
            facebook_prophet_page(df)
        st.success("Processing complete!")

    elif page == "PECNET":
        st.title("PECNET Page")
        # Call the respective page function and display a "Processing" notification
        with st.spinner("Processing..."):
            pecnet_page(df)
        st.success("Processing complete!")

    elif page == "Backtest Results":
        st.title("Backtest Results Page")
        # Call the respective page function and display a "Processing" notification
        with st.spinner("Processing..."):
            backtest_page(df)
        st.success("Processing complete!")

if __name__ == "__main__":
    main()