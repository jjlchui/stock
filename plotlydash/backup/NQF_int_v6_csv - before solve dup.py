import schedule
import requests 
import datetime
#from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
from bs4 import BeautifulSoup
import re 
import os
import time
import statsmodels.api as sm
import csv  

def get_price_stock(ticker="NQ=F"):
    x=0
    try:
        url = "https://finance.yahoo.com/quote/"+ticker+"?p="+ticker+"&.tsrc=fin-srch"
        #url = "https://finance.yahoo.com/quote/U?p=U&.tsrc=fin-srch"
        headers = {"User-Agent" : "Chrome/101.0.4951.41"}
        r = requests.get(url, headers=headers)
        page_content = r.content
        #soup = BeautifulSoup(page_content, 'lxml')
        print('beautiful soup ------')
        soup = BeautifulSoup(page_content, "html.parser")
        web_content = soup.find('div', {'class' :'D(ib) Mend(20px)'}) 
        if (web_content == None):
            time.sleep(1)
            return 0,0,0
        else:
            print('web_content :')
            stock_price1 = web_content.find("fin-streamer", {'class' : 'Fw(b) Fz(36px) Mb(-4px) D(ib)'}).text
            print('stock price1 :', stock_price1) 
            if(stock_price1 == ""  ):
                time.sleep(1)
                return 0,0,0
            else:
                stock_price = stock_price1.replace(",", "")
                print('stock price :', stock_price)
                change = web_content.find("fin-streamer", {'data-field' : "regularMarketChangePercent"}).text
                #tabl = soup.find_all("div", {'class' : "D(ib) W(1/2) Bxz(bb) Pend(12px) Va(t) ie-7_D(i) smartphone_D(b) smartphone_W(100%) smartphone_Pend(0px) smartphone_BdY smartphone_Bdc($seperatorColor)"})
                change = change.strip('()')
                change = re.sub('%', "", change)
                print('% change :',change)
                web_content1 = soup.find('table', {'class' :"W(100%) M(0) Bdcl(c)"}).text
                print('web_content1....', web_content1) 
                words = web_content1.split()
                print('words.....', words)
                old, new = words[4].split('e')
                new_vol = new.split('A')
                my_var = new_vol[0]
                print('new_vol', my_var)
                if (my_var == "N/"):
                    volume = 0
                else:
                    volume = my_var
                    volume = volume.replace(",", "")
                print("volume: ",volume)
                return stock_price, change, volume  
    except ConnectionError:
        print("Network Issue !!!")
    
    return stock_price, change, volume  

a = get_price_stock('NQ=F')

def Save_csv(file, filename):  
    cwd = os.getcwd()
    path = os.path.dirname(cwd)

    file_path = path + "\\stock\\data\\"
    
    #time_stamp = datetime.datetime.now() - datetime.timedelta(hours=13)
    time_stamp = datetime.datetime.now() .strftime('%Y-%m-%d %H:%M:%S')
    timefile = os.path.join(file_path + str(time_stamp[0:11]) +"NQ=F USTime" + filename)
    
    file.drop_duplicates(subset='time', keep=False)
    file.to_csv(timefile, mode='a', header=False, index=False, encoding = 'utf-8', chunksize=1)
    return timefile, time_stamp

def Process_Data(file):
    #df = pd.read_csv(file, header=None, usecols=[0,1,2,3], names=['datetime', 'price', 'change', 'volume'],
    #                 index_col = ['datetime'], parse_dates=['datetime'])
    data = file.copy()
    data.columns = ['datetime', 'price', 'change', 'volume']
    print("Hey, Process_Data_test")
    data['first'] = data['volume'].astype(str).str[0]
    data.drop(data[data["first"] == 'e'].index, inplace=True)
    data = data.drop('first', axis=1)

    data.fillna(method='ffill', inplace=True)
    data.set_index("datetime", inplace=True)
    data.index = pd.DatetimeIndex(data.index)
    
    
    try: 
        #data['price'] = data['price'].astype(float)
        #data['volume'] = data['volume'].astype(float)
        #data['change'] = data['change'].astype(float)
        data['price'] = pd.to_numeric(data['price'])
        data['volume'] = pd.to_numeric(data['volume'])
        data['change'] = pd.to_numeric(data['change'])
    except AttributeError:
        print("cannot conv data to astype float")
        time.sleep(1)
        pass
        
    data_vol = data['volume'].resample('1Min').mean()
    data_vol = pd.DataFrame([data_vol], columns=['volume'])
    data = data['price'].resample('1Min').ohlc()
    
    data['time'] = data.index
    data = data.reset_index(drop=True) 
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    """
    frames= [data, data_vol]
    data=pd.concat(frames)
    """
    data = data[['time', 'open', 'high', 'low', 'close']]

    print(data.head(1))
    
    index_with_nan = data.index[data.isnull().any(axis=1)]
    
    #data.drop(index_with_nan, 0, inplace=True)
    data.fillna(method="ffill" , inplace=True)
    data.reset_index(drop=True, inplace=True)
   
    return data
    
def get_stock_data():

    Running = True

    start_time = time.time()
    print("----------bef get_price_stock:(1)", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))   
    #time_stamp = datetime.datetime.now() - datetime.timedelta(hours=13)
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    price, change, volume = get_price_stock("NQ=F")
    
    if (float(price) == 0 ):
        print('price, == empyt')
        pass  
    else:
        print(" price:", price, " change", change, " volume", volume)
        info = []
        info.append(price)
        info.append(change)
        info.append(volume)
 
        col = []
        col = [time_stamp]
        col.extend(info)
        

    if (float(price) == 0):
            pass    
    else:
        df = pd.DataFrame(col)
        df = df.T
        print("-----col2", col)
        col = ""


    #*** read from df ***

    data = Process_Data(df)
    data = data.drop_duplicates(subset=['time'], keep='first')
    outputfile, time_stamp = Save_csv(data, '_out_stock_data.csv')
    print("----------EXCEL FILE", outputfile, time_stamp)
        
    #*** read from csv for reconcil ***    
    timefile, time_concile_stamp = Save_csv(df, '_stock_data.csv')
    #time.sleep(60)    
     
    finish_time = time.time()
    print("----------bef get_price_stock:(2)", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(finish_time)))  
    return data
           
