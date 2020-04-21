import os
from datetime import datetime

import requests
from bs4 import BeautifulSoup

import zipfile
import pandas as pd
import numpy as np
import calendar
#import matplotlib.pyplot as plt

import math

class TimeFrame:
    ONE_MINUTE = 'M1'
    TICK_DATA = 'T'
    TICK_DATA_LAST = 'T_LAST'
    TICK_DATA_BID = 'T_BID'
    TICK_DATA_ASK = 'T_ASK'


class Platform:
    META_TRADER = 'MT'
    GENERIC_ASCII = 'ASCII'
    EXCEL = 'XLSX'
    NINJA_TRADER = 'NT'
    META_STOCK = 'MS'


class URL:
    META_TRADER = 'https://www.histdata.com/download-free-forex-historical-data/?/metatrader/1-minute-bar-quotes/'
    ASCII_1M = 'https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/'
    ASCII_TICK_DATA = 'https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/'
    EXCEL = 'https://www.histdata.com/download-free-forex-historical-data/?/excel/1-minute-bar-quotes/'
    NINJA_TRADER = 'https://www.histdata.com/download-free-forex-historical-data/?/ninjatrader/1-minute-bar-quotes/'
    NINJA_TRADER_LAST_QUOTES = 'https://www.histdata.com/download-free-forex-historical-data/?/ninjatrader/tick-last-quotes/'
    NINJA_TRADER_BID_QUOTES = 'https://www.histdata.com/download-free-forex-historical-data/?/ninjatrader/tick-bid-quotes/'
    NINJA_TRADER_ASK_QUOTES = 'https://www.histdata.com/download-free-forex-historical-data/?/ninjatrader/tick-ask-quotes/'
    META_STOCK = 'https://www.histdata.com/download-free-forex-historical-data/?/metastock/1-minute-bar-quotes/'


def get_prefix_referer(time_frame, platform):
    if time_frame == TimeFrame.TICK_DATA and platform == Platform.GENERIC_ASCII:
        return URL.ASCII_TICK_DATA
    elif time_frame == TimeFrame.TICK_DATA_LAST and platform == Platform.NINJA_TRADER:
        return URL.NINJA_TRADER_LAST_QUOTES
    elif time_frame == TimeFrame.TICK_DATA_BID and platform == Platform.NINJA_TRADER:
        return URL.NINJA_TRADER_BID_QUOTES
    elif time_frame == TimeFrame.TICK_DATA_ASK and platform == Platform.NINJA_TRADER:
        return URL.NINJA_TRADER_ASK_QUOTES
    elif time_frame == TimeFrame.ONE_MINUTE and platform == Platform.GENERIC_ASCII:
        return URL.ASCII_1M
    elif time_frame == TimeFrame.ONE_MINUTE and platform == Platform.META_TRADER:
        return URL.META_TRADER
    elif time_frame == TimeFrame.ONE_MINUTE and platform == Platform.EXCEL:
        return URL.EXCEL
    elif time_frame == TimeFrame.ONE_MINUTE and platform == Platform.NINJA_TRADER:
        return URL.NINJA_TRADER
    elif time_frame == TimeFrame.ONE_MINUTE and platform == Platform.META_STOCK:
        return URL.META_STOCK
    else:
        raise Exception('Invalid combination of time_frame and platform.')


def get_referer(referer_prefix, pair, year, month):
    if month is not None:
        return referer_prefix + '{}/{}/{}'.format(pair.lower(), year, month)
    return referer_prefix + '{}/{}'.format(pair.lower(), year)


def download_hist_data(year='2016',
                       month=None,
                       pair='eurusd',
                       time_frame=TimeFrame.ONE_MINUTE,
                       platform=Platform.GENERIC_ASCII,
                       output_directory='.',
                       verbose=True):
    """
    Download 1-Minute FX data per month.
    :param year: Trading year. Format is 2016.
    :param month: Trading month. Format is 7 or 12.
    :param pair: Currency pair. Example: eurgbp.
    :param time_frame: M1 (one minute) or T (tick data)
    :param platform: MT, ASCII, XLSX, NT, MS
    :param output_directory: Where to dump the data.
    :return: ZIP Filename.
    """

    tick_data = time_frame.startswith('T')
    if (not tick_data) and ((int(year) >= datetime.now().year and month is None) or
                            (int(year) <= datetime.now().year - 1 and month is not None)):
        msg = 'For the current year, please specify month=7 for example.\n'
        msg += 'For the past years, please query per year with month=None.'
        raise AssertionError(msg)

    prefix_referer = get_prefix_referer(time_frame, platform)
    referer = get_referer(prefix_referer, pair.lower(), year, month)

    # Referer is the most important thing here.
    headers = {'Host': 'www.histdata.com',
               'Connection': 'keep-alive',
               'Content-Length': '104',
               'Cache-Control': 'max-age=0',
               'Origin': 'https://www.histdata.com',
               'Upgrade-Insecure-Requests': '1',
               'Content-Type': 'application/x-www-form-urlencoded',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
               'Referer': referer}

    if verbose:
        print(referer)
    r1 = requests.get(referer, allow_redirects=True)
    assert r1.status_code == 200, 'Make sure the website www.histdata.com is up.'

    soup = BeautifulSoup(r1.content, 'html.parser')
    try:
        token = soup.find('input', {'id': 'tk'}).attrs['value']
        assert len(token) > 0
    except:
        raise AssertionError('There is no token. Please make sure your year/month/pair is correct.'
                             'Example is year=2016, month=7, pair=eurgbp')

    data = {'tk': token,
            'date': str(year),
            'datemonth': '{}{}'.format(year, str(month).zfill(2)) if month is not None else str(year),
            'platform': platform,
            'timeframe': time_frame,
            'fxpair': pair.upper()}
    r = requests.post(url='https://www.histdata.com/get.php',
                      data=data,
                      headers=headers)

    assert len(r.content) > 0, 'No data could be found here.'
    if verbose:
        print(data)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if month is None:
        output_filename = 'HISTDATA_COM_{}_{}_{}{}.zip'.format(platform, pair.upper(), time_frame, str(year))
    else:
        output_filename = 'HISTDATA_COM_{}_{}_{}{}.zip'.format(platform, pair.upper(), time_frame,
                                                       '{}{}'.format(year, str(month).zfill(2)))
    output_filename = os.path.join(output_directory, output_filename)
    with open(output_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    if verbose:
        print('Wrote to {}'.format(output_filename))
    return output_filename


if __name__ == '__main__':
    # print(download_hist_data(year='2019', month='6', platform=Platform.NINJA_TRADER, time_frame=TimeFrame.TICK_DATA_LAST))
    # print(download_hist_data(year='2019', month='6', platform=Platform.NINJA_TRADER, time_frame=TimeFrame.TICK_DATA_ASK))
    # print(download_hist_data(year='2019', month='6', platform=Platform.NINJA_TRADER, time_frame=TimeFrame.TICK_DATA_BID))
    # print(download_hist_data(year='2019', month='6', platform=Platform.NINJA_TRADER, time_frame=TimeFrame.ONE_MINUTE))
    # print(download_hist_data(year='2019', month='6', platform=Platform.GENERIC_ASCII, time_frame=TimeFrame.TICK_DATA))
    # print(download_hist_data(year='2019', month='6', platform=Platform.EXCEL, time_frame=TimeFrame.ONE_MINUTE))
    # print(download_hist_data(year='2019', month='6', platform=Platform.META_TRADER, time_frame=TimeFrame.ONE_MINUTE))
    # print(download_hist_data(year='2019', month='6', platform=Platform.META_STOCK, time_frame=TimeFrame.ONE_MINUTE))

    # print(
    #     download_hist_data(year='2018', month='6', platform=Platform.NINJA_TRADER, time_frame=TimeFrame.TICK_DATA_LAST))
    # print(
    #     download_hist_data(year='2018', month='6', platform=Platform.NINJA_TRADER, time_frame=TimeFrame.TICK_DATA_ASK))
    # print(
    #     download_hist_data(year='2018', month='6', platform=Platform.NINJA_TRADER, time_frame=TimeFrame.TICK_DATA_BID))
    # print(download_hist_data(year='2018', month=None, platform=Platform.NINJA_TRADER, time_frame=TimeFrame.ONE_MINUTE))
    # print(download_hist_data(year='2018', month='2', platform=Platform.GENERIC_ASCII, time_frame=TimeFrame.TICK_DATA))
    # print(download_hist_data(year='2018', month=None, platform=Platform.EXCEL, time_frame=TimeFrame.ONE_MINUTE))
    # print(download_hist_data(year='2018', month=None, platform=Platform.META_TRADER, time_frame=TimeFrame.ONE_MINUTE))
    # print(download_hist_data(year='2018', month=None, platform=Platform.META_STOCK, time_frame=TimeFrame.ONE_MINUTE))
    pass

def clean_data(data_bid, first):

    N = data_bid.shape[0]
    
    Hour = []
    Min = []
    Time = []
    Date = []
    Open = []
    High = []
    Low = []
    Close = []
    Ask = []
    Volume = []
    
    Hour.append(int(data_bid["Time"][first].split(':')[0]))
    Min.append(int(data_bid["Time"][first].split(':')[1]))
    Time.append(data_bid["Time"][first])
    Date.append(data_bid["Date"][first])
    Open.append(data_bid["Open"][first])
    Close.append(data_bid["Close"][first])
    High.append(data_bid["High"][first])
    Low.append(data_bid["Low"][first])
    Volume.append(data_bid["Volume"][first])
    
    
    for i in range(first+1,N):
        H = int(data_bid["Time"][i].split(':')[0])
        M = int(data_bid["Time"][i].split(':')[1])
        if Min[-1] == 59:
            M = M+60
        dis = M - Min[-1]
        for j in range(dis):
            Hour.append(int(data_bid["Time"][i].split(':')[0]))
            Min.append(int(data_bid["Time"][i].split(':')[1]))
            Time.append(data_bid["Time"][i])
            Date.append(data_bid["Date"][i])
            Open.append(data_bid["Open"][i])
            Close.append(data_bid["Close"][i])
            High.append(data_bid["High"][i])
            Low.append(data_bid["Low"][i])
            Volume.append(data_bid["Volume"][i])
            
    d = pd.DataFrame(list(zip(Date, Time, Open, High, Low, Close, Volume)),
            columns=['Date','Time','Open', 'High', 'Low', 'Close', 'Volume'])
    
    return(d)

def time_convertor(data_bid, period):
    N = data_bid.shape[0]
    Time = []
    Date = []
    Open = []
    High = []
    Low = []
    Close = []
    #Ask = []
    Volume = []
    for i in range(0,N - period,period):
        Time.append(data_bid["Date"][i] + ' ' + data_bid["Time"][i])
        Date.append(data_bid["Date"][i])
        Open.append(data_bid["Open"][i])
        Close.append(data_bid["Close"][i+period-1])
        High.append(max(data_bid["High"][i:i+period]))
        Low.append(min(data_bid["Low"][i:i+period]))
        Volume.append(sum(data_bid["Volume"][i:i+period]))
    return(Time, Date, Open, High, Low, Close, Volume)

def preprocess(data,period):    
    Time, Date, Open, High, Low, Close, Volume = time_convertor(data,period)

    week_day = []
    for i in range(len(Time)):
        day = Date[i].split('.')[2]
        month = Date[i].split('.')[1]
        year = Date[i].split('.')[0]
        week_day.append(calendar.weekday(int(year), int(month), int(day)))

    Time = np.array(Time)
    Open = np.array(Open)  
    High =  np.array (High)
    Low =  np.array(Low)
    Close =  np.array(Close)
    #Ask = np.array(Ask)
    week_day = np.array(week_day)

    l = np.where(np.logical_or(week_day == 5, week_day == 6))

    Time = np.delete(Time, l) 
    Date = np.delete(Date, l)  
    Open = np.delete(Open, l)  
    High =  np.delete (High, l)
    Low =  np.delete(Low, l)
    Close =  np.delete(Close, l)
    Volume = np.delete(Volume, l)
    #Ask = np.delete(Ask, l)

    data = pd.DataFrame(list(zip(Time, Open, High, Low, Close, Volume)),
                columns=['Time','Open', 'High', 'Low', 'Close', 'Volume'])
    
    #return(data, Time, Date, Open, High, Low, Close, Volume)
    return(data)

def construct_data(pair, start_year, end_year, period):
    first = 0
    time_frame = "M1"
    years = np.arange(start_year,end_year + 1)
    df = pd.DataFrame(columns=['Time','Open', 'High', 'Low', 'Close', 'Volume'])
    now = datetime.now()
    curr = os.getcwd()
    if not os.path.split(curr)[1]=='FX_data': os.chdir("./FX_data/")   
    for year in years:
        if year == now.year:
            for month in np.arange(1, now.month):
                yearmonth = str(year) + str(month) if month > 9 else str(year) + '0' + str(month)
               
                file_name ='HISTDATA_COM_MT_{}_{}{}.zip'.format(pair.upper(), time_frame, yearmonth)
                file_name_csv = 'DAT_MT_{}_{}_{}.csv'.format(pair.upper(), time_frame, yearmonth)
                if not os.path.exists(file_name):
                    download_hist_data(year=year, month=month,pair=pair, platform=Platform.META_TRADER, time_frame=TimeFrame.ONE_MINUTE, verbose=False)
                archive = zipfile.ZipFile(file_name, 'r')
                data = pd.read_csv(archive.open(file_name_csv), header=None)
                data.columns = ['Date','Time','Open','High','Low','Close','Volume']
                if year == start_year and month == 1:
                    while not int(data["Time"][first].split(':')[1]) == 0:
                        first+=1
                data = clean_data(data,first)
                data = preprocess(data,period)
                df = pd.concat([df,data],ignore_index=True,sort=False, axis=0)            
        else:
            file_name = 'HISTDATA_COM_MT_{}_{}{}.zip'.format(pair.upper(), time_frame,year)
            file_name_csv = 'DAT_MT_{}_{}_{}.csv'.format(pair.upper(), time_frame,year)
            if not os.path.exists(file_name):
                download_hist_data(year=year,pair=pair, platform=Platform.META_TRADER, time_frame=TimeFrame.ONE_MINUTE, verbose=False)
            archive = zipfile.ZipFile(file_name, 'r')
            data = pd.read_csv(archive.open(file_name_csv), header=None)
            data.columns = ['Date','Time','Open','High','Low','Close','Volume']
            if year == start_year:
                while not int(data["Time"][first].split(':')[1]) == 0:
                    first+=1
            data = clean_data(data,first)
            data = preprocess(data,period)
            df = pd.concat([df,data],ignore_index=True,sort=False, axis=0)
    return(df)
    
#PIP_RATIO = math.pow(10,-len(str(Close[0]).split('.')[1])+1)
#self.INVERSE_PIP_RATIO = 1/PIP_RATIO
MARGIN = 4


def sma(series, window=50, min_periods=0):
    # Center must always be False to circumvent look-ahead bias
    sma = series.rolling(window=window, min_periods=min_periods,
                         center=False).mean()
    sma.rename(index='SMA', inplace=True)
    return(sma)

def stoc(df, col_labels=('Low', 'High', 'Close'),
         k_smooth=5, d_smooth=5, window=15, min_periods=0):
    # Should col_labels be a dictionary? i.e. {'low':'Low', ...}
    low = df[col_labels[0]]
    high = df[col_labels[1]]
    close = df[col_labels[2]]

    # min_periods should always be 0 for this
    lowest_low = low.rolling(window=window, min_periods=window).min()
    highest_high = high.rolling(window=window, min_periods=window).max()

    k = (close - lowest_low) / (highest_high - lowest_low) * 100

    if min_periods > 0:
        k[:min_periods] = np.NaN

    if k_smooth != 0:
        k = sma(k, window=k_smooth)
    d = sma(k, window=d_smooth)
   # return pd.DataFrame({'%K': k, '%D': d})

    return(k)

def sstoc(df, col_labels=('Low', 'High', 'Close'),
          k_smooth=5, d_smooth=5, window=15):
    return stoc(df, col_labels=col_labels, k_smooth=k_smooth,
    d_smooth=d_smooth, window=window, min_periods=0)

def slow_stochastic(data,Kperiod,Dperiod): 
    not_null = 0
    l, h = pd.rolling_min(data["Low"], Kperiod), pd.rolling_max(data["High"], Kperiod)
    k = 100 * (data["Close"] - l) / (h - l) 
    #k = pd.rolling_mean(k, Dperiod)
    #if k_smooth != 0:
    k = sma(k, window=Dperiod)
 #   d = sma(k, window=d_smooth)
    for i in range(19,k.shape[0]):
        if not pd.isnull(k[i]):
            not_null = k[i]
        else:
            k[i] = not_null    
    return(k)

def stochastic_crossover(i, stochastic):
    if stochastic[i]>=50.0 and stochastic[i-1]<50.0:
        return(1)
    elif stochastic[i]<50.0 and stochastic[i-1]>=50.0:
        return(2)
    else:
        return(0)

#def distant(first,second,Distance = 50):
#    if((first-second)> Distance * PIP_RATIO):
#        return True
#    else:
#        return False

class position:
    def __init__(self, pip_ratio):
        self.have_position = False
        self.open_t = []
        self.profitloss = []
        self.max_profit = []
        self.max_loss = []
        self.shooting_d = []
        self.trigger_d = []
        self.returns = []
        self.accumPL = []
        self.volume = []
        self.PIP_RATIO = pip_ratio
        self.INVERSE_PIP_RATIO = 1/pip_ratio
    def take_position(self, get_positopn_type, uptrend, cum_pips, price, shooting_delta, trigger_delta, time, limit, stop, trailing_stop,):
        if (self.have_position):
            if get_positopn_type == "Buy" and self.position_type == "Short":
                profit = (self.open_price - price) * self.INVERSE_PIP_RATIO
                cum_pips += profit
                
                self.accumPL.append(cum_pips)
                self.open_t.append(self.open_time)
                self.profitloss.append(profit)
                self.max_profit.append((self.open_price - self.low)*self.INVERSE_PIP_RATIO/self.open_price)
                self.max_loss.append((self.open_price - self.high)*self.INVERSE_PIP_RATIO/self.open_price)
                self.shooting_d.append(self.shooting_delta)
                self.trigger_d.append(self.trigger_delta)
                #self.returns.append(list((Close[i-31:i-1] - Close[i-30:i])*self.INVERSE_PIP_RATIO/Close[i-31:i-1]))
               # self.volume.append(list((Volume[i-31:i-1] - Volume[i-30:i])*self.INVERSE_PIP_RATIO/Volume[i-31:i-1]))
                
                
                #print("Short position opened %s price %.5f closed %s profit %.5f pips"%(self.open_time, self.open_price, time, profit))
                #print("Maximum Profit: ", (self.open_price - self.low)*self.INVERSE_PIP_RATIO)
                #print("Maximum Loss: ", (self.open_price - self.high)*self.INVERSE_PIP_RATIO)
               # print("Shooting Delta: ", self.shooting_delta)
               # print("Trigger Delta: ", self.trigger_delta)
               # print("Ratio of Deltas: ", self.shooting_delta/self.trigger_delta)
                #print("Cumulative Pips is :",cum_pips)
                #print("*************************************************")
                self.open_price = price
                self.position_type = "Long"
                self.open_time = time
                self.limit_price = limit
                self.stop_loss = stop
                self.trailing_stop = trailing_stop
                self.trailing_price = price
                self.high = 0
                self.low = 10000
                self.shooting_delta = shooting_delta
                self.trigger_delta = trigger_delta
                uptrend = True
                #print("Buy at %s price %.5f, limit order at %.5f, stop loss at %.5f with volume difference of %.2f"%(time, price,self.limit_price, self.stop_loss,(Volume[i] - Volume[i-1])*self.INVERSE_PIP_RATIO/Volume[i-1]))
                #print("Buy at %s price %.5f, limit order at %.5f, stop loss at %.5f"%(time, price,self.limit_price, self.stop_loss))
                #print("*************************************************")
            elif get_positopn_type == "Sell" and self.position_type == "Long":
                profit = (price - self.open_price) * self.INVERSE_PIP_RATIO
                cum_pips += profit
               
                self.accumPL.append(cum_pips)
                self.open_t.append(self.open_time)
                self.profitloss.append(profit)
                self.max_profit.append((self.high - self.open_price)*self.INVERSE_PIP_RATIO/self.open_price)
                self.max_loss.append((self.low - self.open_price)*self.INVERSE_PIP_RATIO/self.open_price)
                self.shooting_d.append(self.shooting_delta)
                self.trigger_d.append(self.trigger_delta)
                #self.returns.append(list((Close[i-31:i-1] - Close[i-30:i])*self.INVERSE_PIP_RATIO/Close[i-31:i-1]))
              #  self.volume.append(list((Volume[i-31:i-1] - Volume[i-30:i])*self.INVERSE_PIP_RATIO/Volume[i-31:i-1]))
                
                #print("Long position opened %s price %.5f closed %s profit %.5f pips"%(self.open_time, self.open_price, time, profit))
                #print("Maximum Profit: ", (self.high - self.open_price)*self.INVERSE_PIP_RATIO)
                #print("Maximum Loss: ", (self.low - self.open_price)*self.INVERSE_PIP_RATIO)
               # print("Shooting Delta: ", self.shooting_delta)
               # print("Trigger Delta: ", self.trigger_delta)
               # print("Ratio of Deltas: ", self.shooting_delta/self.trigger_delta)
                #print("Cumulative Pips is :",cum_pips)
                #print("*************************************************")
                self.open_price = price
                self.position_type = "Short"
                self.open_time = time
                self.limit_price = limit
                self.stop_loss = stop
                self.trailing_stop = trailing_stop
                self.trailing_price = price
                self.high = 0
                self.low = 10000
                self.shooting_delta = shooting_delta
                self.trigger_delta = trigger_delta
                uptrend = False
                #print("Sell at %s price %.5f, limit order at %.5f, stop loss at %.5f with volume difference of %.2f"%(time, price, self.limit_price, self.stop_loss, (Volume[i] - Volume[i-1])*self.INVERSE_PIP_RATIO/Volume[i-1]))
                #print("Sell at %s price %.5f, limit order at %.5f, stop loss at %.5f"%(time, price, self.limit_price, self.stop_loss))
                #print("************************************************")
        else:
            if get_positopn_type == "Buy":
                self.open_price = price
                self.have_position = True;
                self.position_type = "Long"
                self.open_time = time
                self.limit_price = limit
                self.stop_loss = stop
                self.trailing_stop = trailing_stop
                self.trailing_price = price
                self.high = 0
                self.low = 10000
                self.shooting_delta = shooting_delta
                self.trigger_delta = trigger_delta
                #self.returns.append(list((Close[i-31:i-1] - Close[i-30:i])*self.INVERSE_PIP_RATIO/Close[i-31:i-1]))
            #    self.volume.append(list((Volume[i-31:i-1] - Volume[i-30:i])*self.INVERSE_PIP_RATIO/Volume[i-31:i-1]))
                uptrend = True; 
                #print("Buy at %s price %.5f, limit order at %.5f, stop loss at %.5f with volume difference of %.2f"%(time, price, self.limit_price, self.stop_loss, (Volume[i] - Volume[i-1])*self.INVERSE_PIP_RATIO/Volume[i-1]))
                #print("Buy at %s price %.5f, limit order at %.5f, stop loss at %.5f"%(time, price, self.limit_price, self.stop_loss))
                
                #print("*************************************************")
            elif get_positopn_type == "Sell":
                self.open_price = price
                self.have_position=True; 
                self.position_type = "Short"
                self.open_time = time
                self.limit_price = limit
                self.stop_loss = stop
                self.trailing_stop = trailing_stop
                self.trailing_price = price
                self.high = 0
                self.low = 10000
                self.shooting_delta = shooting_delta
                self.trigger_delta = trigger_delta
                #self.returns.append(list((Close[i-31:i-1] - Close[i-30:i])*self.INVERSE_PIP_RATIO/Close[i-31:i-1]))
            #    self.volume.append(list((Volume[i-31:i-1] - Volume[i-30:i])*self.INVERSE_PIP_RATIO/Volume[i-31:i-1]))
                uptrend = False 
                #print("Sell at %s price %.5f, limit order at %.5f, stop loss at %.5f  with volume difference of %.2f"%(time, price, self.limit_price, self.stop_loss, (Volume[i] - Volume[i-1])*self.INVERSE_PIP_RATIO/Volume[i-1]))
                #print("Sell at %s price %.5f, limit order at %.5f, stop loss at %.5f"%(time, price, self.limit_price, self.stop_loss))
                
                #print("*************************************************")
        return uptrend, cum_pips 
    
    def close_position(self, cum_pips, time, Low, High, state):
        if self.position_type == "Short":
            if High > self.high: self.high = High
            if Low < self.low: self.low = Low
            
            if Low + MARGIN * self.PIP_RATIO < self.limit_price:
                profit = (self.open_price - self.limit_price)*self.INVERSE_PIP_RATIO
                cum_pips += profit
                
                self.accumPL.append(cum_pips)
                self.open_t.append(self.open_time)
                self.profitloss.append(profit)
                self.max_profit.append((self.open_price - self.low)*self.INVERSE_PIP_RATIO/self.open_price)
                self.max_loss.append((self.open_price - self.high)*self.INVERSE_PIP_RATIO/self.open_price)
                self.shooting_d.append(self.shooting_delta)
                self.trigger_d.append(self.trigger_delta)
                
                
                #print("Short position opened %s price %.5f closed %s profit %d pips"%(self.open_time, self.open_price, time, profit))
                #print("Maximum Profit: ", (self.open_price - self.low)*self.INVERSE_PIP_RATIO)
                #print("Maximum Loss: ", (self.open_price - self.high)*self.INVERSE_PIP_RATIO)
               # print("Shooting Delta: ", self.shooting_delta)
               # print("Trigger Delta: ", self.trigger_delta)
               # print("Ratio of Deltas: ", self.shooting_delta/self.trigger_delta)
                #print("Cumulative Pips is :",cum_pips)
                #print("*************************************************")
                self.low = 10000
                self.high = 0
                if state == 3:
                    state = 0
                elif state == 13:
                    state = 10
                self.have_position = False
            elif High + MARGIN * self.PIP_RATIO > self.stop_loss:
                profit = (self.open_price - self.stop_loss) * self.INVERSE_PIP_RATIO
                cum_pips += profit
                
                self.accumPL.append(cum_pips)
                self.open_t.append(self.open_time)
                self.profitloss.append(profit)
                self.max_profit.append((self.open_price - self.low)*self.INVERSE_PIP_RATIO/self.open_price)
                self.max_loss.append((self.open_price - self.high)*self.INVERSE_PIP_RATIO/self.open_price)
                self.shooting_d.append(self.shooting_delta)
                self.trigger_d.append(self.trigger_delta)
                
                
                #print("Short position opened %s price %.5f closed %s profit %d pips"%(self.open_time, self.open_price, time, profit))
                #print("Maximum Profit: ", (self.open_price - self.low)*self.INVERSE_PIP_RATIO)
                #print("Maximum Loss: ", (self.open_price - self.high)*self.INVERSE_PIP_RATIO)
               # print("Shooting Delta: ", self.shooting_delta)
               # print("Trigger Delta: ", self.trigger_delta)
               # print("Ratio of Deltas: ", self.shooting_delta/self.trigger_delta)
                #print("Cumulative Pips is :",cum_pips)
                #print("*************************************************")
                self.low = 10000
                self.high = 0
                if state == 3:
                    state = 0
                elif state == 13:
                    state = 10
                self.have_position = False
                
            if self.trailing_stop > 0:
                if self.trailing_price - Low > self.trailing_stop * self.PIP_RATIO:
                    self.stop_loss = self.stop_loss - self.trailing_stop * self.PIP_RATIO
                    self.trailing_price = self.trailing_price - self.trailing_stop * self.PIP_RATIO
                
        else:
            if High > self.high: self.high = High
            if Low < self.low: self.low = Low
            
            if High > self.limit_price:
                profit = (self.limit_price - self.open_price) * self.INVERSE_PIP_RATIO
                cum_pips += profit
                
                self.accumPL.append(cum_pips)
                self.open_t.append(self.open_time)
                self.profitloss.append(profit)
                self.max_profit.append((self.high - self.open_price)*self.INVERSE_PIP_RATIO/self.open_price)
                self.max_loss.append((self.low - self.open_price)*self.INVERSE_PIP_RATIO/self.open_price)
                self.shooting_d.append(self.shooting_delta)
                self.trigger_d.append(self.trigger_delta)
                
                
                #print("Long position opened %s price %.5f closed %s profit %d pips"%(self.open_time, self.open_price, time, profit))
                #print("Maximum Profit: ", (self.high - self.open_price)*self.INVERSE_PIP_RATIO)
                #print("Maximum Loss: ", (self.low - self.open_price)*self.INVERSE_PIP_RATIO)
               # print("Shooting Delta: ", self.shooting_delta)
               # print("Trigger Delta: ", self.trigger_delta)
               # print("Ratio of Deltas: ", self.shooting_delta/self.trigger_delta)
                #print("Cumulative Pips is :",cum_pips)
                #print("************************************************")
                self.low = 10000
                self.high = 0
                if state == 3:
                    state = 0
                elif state == 13:
                    state = 10
                self.have_position = False
            elif Low < self.stop_loss:
                profit = (self.stop_loss - self.open_price) * self.INVERSE_PIP_RATIO
                cum_pips += profit
                
                self.accumPL.append(cum_pips)
                self.open_t.append(self.open_time)
                self.profitloss.append(profit)
                self.max_profit.append((self.high - self.open_price)*self.INVERSE_PIP_RATIO/self.open_price)
                self.max_loss.append((self.low - self.open_price)*self.INVERSE_PIP_RATIO/self.open_price)
                self.shooting_d.append(self.shooting_delta)
                self.trigger_d.append(self.trigger_delta)
                #print("low")
                #print(Low)
                #print("stop loss")
                #print(self.stop_loss)
            
                #print("Long position opened %s price %.5f closed %s profit %d pips"%(self.open_time, self.open_price, time, profit))
                #print("Maximum Profit: ", (self.high - self.open_price)*self.INVERSE_PIP_RATIO)
                #print("Maximum Loss: ", (self.low - self.open_price)*self.INVERSE_PIP_RATIO)
               # print("Shooting Delta: ", self.shooting_delta)
               # print("Trigger Delta: ", self.trigger_delta)
               # print("Ratio of Deltas: ", self.shooting_delta/self.trigger_delta)
                #print("Cumulative Pips is :",cum_pips)
               # print("*************************************************")
                self.low = 10000
                self.high = 0
                if state == 3:
                    state = 0
                elif state == 13:
                    state = 10
                self.have_position = False
                
            if self.trailing_stop > 0:                
                if High - self.trailing_price > self.trailing_stop * self.PIP_RATIO:
                    self.stop_loss =  self.stop_loss + self.trailing_stop * self.PIP_RATIO
                    self.trailing_price = self.trailing_price + self.trailing_stop * self.PIP_RATIO     
        return cum_pips, state