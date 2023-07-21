#####################BACKTEST##############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st
plt.style.use("seaborn")

class IterativeBase():

    def __init__(self, symbol, start, end, amount, use_prediction = True, csv_file_path= "backtest_pecnet.csv"):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.csv_file_path = csv_file_path
        self.initial_balance = amount
        self.current_balance = amount
        self.units = 0
        self.trades = 0
        self.use_prediction = use_prediction
        self.get_data(csv_file_path)
        self.create_positions()

    def create_positions(self):
        pos = pd.DataFrame(columns=["date","price","pos_changed","current_balance"])
        pos.set_index("date")
        self.positions = pos
        
    def add_position(self, date, price, pos_changed, current_balance):
        df_new_row = pd.DataFrame([{ 'date':date,'price':price, 'pos_changed':pos_changed, 'current_balance':current_balance }])
        self.positions = pd.concat([self.positions, df_new_row], ignore_index=True)

    def get_data(self, csv_file_path):
        raw = pd.read_csv(csv_file_path, parse_dates = ["date"], index_col = "date", engine='python').dropna()
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
        st.write(75 * "-")
        print("{} | net performance (%) = {}".format(date, round(perf, 2) ))

        text = "Net Performance (%) " + str(perf)
        color = "blue"
        # Create a box with custom colored text using Markdown
        styled_text = f'<div style="color: {color}; font-size: 24px; padding: 10px; text-align: center; border: 1px solid {color}; border-radius: 5px;">{text}</div>'
        st.markdown(styled_text, unsafe_allow_html=True)

        text = "Number of Trades Executed = " + str(self.trades)
        color = "blue"
        # Create a box with custom colored text using Markdown
        styled_text = f'<div style="color: {color}; font-size: 24px; padding: 10px; text-align: center; border: 1px solid {color}; border-radius: 5px;">{text}</div>'
        st.markdown(styled_text, unsafe_allow_html=True)
        
        st.write(75 * "-")
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
        df = pd.read_csv(self.csv_file_path)
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
        self.get_data(csv_file_path=self.csv_file_path) # reset dataset
        
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
        self.get_data(csv_file_path=self.csv_file_path) # reset dataset
        
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
        self.close_pos(bar+1) # close position at the last bar
        #self.positions["returns"] = positions.current_balance - positions.current_balance.shift(1)

