from dash import dcc, html, callback_context
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash
import datetime
import os
import plotly
import pandas as pd
import pandas_ta as ta
from dash.exceptions import PreventUpdate
import numpy as np
from dash_extensions import WebSocket
import statsmodels.api as sm
from scipy.stats import linregress
from decimal import Decimal
import datetime as dt

def macd_strategy(data):
    df = data.copy()
    

    
    ##### MA #####
    """
    import time
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    conv_date = df['Datetime'].apply(lambda x: time.mktime(x.timetuple()))
    #df['conv_date'] = df.Datetime.apply(lambda x: x.map(dt.datetime.toordinal))
    slopesxy, intercept, r_value, p_value, std_err = linregress(conv_date , df['Close'])
    
    
    slopesxy = df['Close'].apply(lambda s: linregress(s.reset_index())[0])
    df['slopes'] = (np.rad2deg(np.arctan(np.array(slopesxy))))
    """
    
    slopes = df['Close'].rolling(10).apply(lambda s: linregress(s.reset_index())[0])
    slopes20 = df['Close'].rolling(20).apply(lambda s: linregress(s.reset_index())[0])
    df['slopes'] = (np.rad2deg(np.arctan(np.array(slopes))))
    df['slopes20'] = (np.rad2deg(np.arctan(np.array(slopes20))))
    df['slopess1p'] = df['slopes'].shift(1)
    df['slopes20s1p'] = df['slopes20'].shift(1)
    """
    df['slope_chg_dn'] = np.where(((df.slopes < 0) & (df.slopess1p > 0)),"y", "NaN")
    df['slope_chg_up'] = np.where(((df.slopes > 0) & (df.slopess1p < 0)),"y", "NaN")
    
    df['slope_chg_up_s1'] = df['slope_chg_up'].shift(1)
    df['slope_chg_dn_s1'] = df['slope_chg_dn'].shift(1)
    
    df['slope_chg_dn20'] = np.where(((df['slopes20'] < 0) & (df['slopes20s1p'] > 0)),"y", "NaN")
    df['slope_chg_up20'] = np.where(((df['slopes20']  > 0) & (df['slopes20s1p'] < 0)),"y", "NaN")
    
    df['slope_chg_up20_s1'] = df['slope_chg_up20'].shift(1)
    df['slope_chg_dn20_s1'] = df['slope_chg_dn20'].shift(1)
    
    
    df['ma1s1'] = df['MA10'].shift(1)
    df['ma1s2'] = df['MA10'].shift(2)
    df['ma2s1'] = df['MA20'].shift(1)
    df['ma2s2'] = df['MA20'].shift(2)
    
    df['buy_pt'] = np.where((((df.MA10 <= df.MA20) & (df.ma1s1 > df.ma2s1)) | 
                ((df.MA10 >= df.MA20) & (df.ma1s1 < df.ma2s1))),
                "y", "NaN")
    df['sell_pt'] =  np.where((((df.MA10 <= df.MA20) & (df.ma1s1 > df.ma2s1)) | 
                ((df.MA10 >= df.MA20) & (df.ma1s1 < df.ma2s1))),       
                "y", "NaN")
    
    df['buy_pt_s1'] = df['buy_pt'].shift(1)
    df['sell_pt_s1'] = df['sell_pt'].shift(1)
    """
    ##### MACD #####
    
    df['macds1'] = df['MACD_12_26_9'].shift(-1)
    df['macdss1'] = df['MACDs_12_26_9'].shift(-1)
    
    df['buy_pt'] = np.where((((df.MACD_12_26_9 <= df.MACDs_12_26_9) & (df.macds1 > df.macdss1)) | 
                ((df.MACD_12_26_9 >= df.MACDs_12_26_9) & (df.macds1  < df.macdss1))), "y", "n")
    df['sell_pt'] = np.where((((df.MACD_12_26_9 <= df.MACDs_12_26_9) & (df.macds1 > df.macdss1)) | 
                ((df.MACD_12_26_9  >= df.MACDs_12_26_9) & (df.macds1 < df.macdss1))), "y", "n")
    
    df['buy_pt_s1'] = df['buy_pt'].shift(1)
    df['sell_pt_s1'] = df['sell_pt'].shift(1)
    
    df['macdps1'] = df['MACD_12_26_9'].shift(1)
    df['macdsps1'] = df['MACDs_12_26_9'].shift(1)
    
    df['macd_dn_to_up'] = np.where((df.macdps1 < df.macdsps1) & (df.MACD_12_26_9 > df.MACDs_12_26_9), "y", 'n')
    df['macd_up_to_dn'] = np.where((df.macdps1 > df.macdsps1) & (df.MACD_12_26_9 < df.MACDs_12_26_9), "y", 'n')
    
    df['vol_gt2'] = np.where(((df['MACD_12_26_9'] > 1.5) | (df['MACD_12_26_9'] < -1.5)), "y", "n") 
    
    
    ##### SUPPORT & AGAINST LINE #####
    
    max = df.Close.max()
    min = df.Close.min()
    
    df['max'] = np.where(df.Close == max, df.Close, "Nan" )
    df['min'] = np.where(df.Close == min, df.Close, "Nan" )
    df.max_20 = df.Close.tail(20).max()
    df.min_20 = df.Close.tail(20).min()
    
    df['RSI_14_B_30s1'] = df['RSI_14_B_30'].shift(1)
    df['RSI_14_A_70s1'] = df['RSI_14_A_70'].shift(1)
    
    ####### BUY #######
    
    """
    Buy = np.where(                   
                 (df['buy_pt_s1'] == 'y') &
                 #(old_slope20 > 0) & 
                 (df['macd_dn_to_up'] == 'y') & 
                 (df['vol_gt2'] == 'y') &
                 (df['macds1'] > df['macdss1'])
                 ,df['Close'], "NaN"),
    """
    
    Buy = np.where(((slopes > 0) &
                   #(df['slope_chg_up'] == 'y') & 
                   (df['macd_dn_to_up'] == 'y') & 
                   (df['vol_gt2'] == 'y') &
                   (df['macds1'] > df['macdss1'])) |
                   (df['Close'] == df['min']),
                    df['Close'], "NaN"),
    
    # sell compare with the buy price
    buy_price=np.where(Buy =='NaN', "NaN", Buy)
    buy_price = buy_price.T
    df.reset_index()
    df['buy_price'] = buy_price
    df['buy_price'].fillna(method='ffill', inplace=True)
    
    # total Buy/Buy count
    Buy_list = list(Buy)
    df_Buy = pd.DataFrame(Buy_list, index=['Buy'])
    df_Buy = df_Buy.T     
    df = pd.concat([df, df_Buy], axis=1)
    df['Buy_zero'] = np.where(df.Buy == "NaN", 0, df.Buy)
    df['tot_buy'] = np.add.accumulate(df['Buy_zero'].astype(float))
    
    df['Buy_count_tmp'] = np.where(df['Buy'] == 'NaN', 0, 1)
    df['Buy_count'] = np.add.accumulate(df['Buy_count_tmp'])
            
    
    ####### SELL #######
    
    Sell = np.where(((slopes < 0) &
                    #(df['slope_chg_dn'] == 'y') &
                     (df['macd_up_to_dn'] == 'y') &
                     (df['vol_gt2'] == "y") &
                     (df['macds1'] < df['macdss1'])) |
                     (df['Close'] == df['max']),
                      df['Close'], "NaN"),
    
    """
    Sell = np.where(
                    (df['sell_pt_s1'] == 'y') &
                     #(old_slope20 < 0) & 
                     (df['macd_up_to_dn'] == 'y') &
                     (df['vol_gt2'] == "y") &
                     (df['macds1'] < df['macdss1']) 
                      ,df['Close'], "NaN"),

     """
     
    # total SELL/count
    
    Sell_list = list(Sell)
    df_Sell = pd.DataFrame(Sell_list, index=['Sell'])
    df_Sell = df_Sell.T
    df = pd.concat([df, df_Sell], axis=1)
    df['Sell_zero'] = np.where(df.Sell == "NaN", 0, df.Sell)
    df['tot_sell'] = np.add.accumulate(df['Sell_zero'].astype(float))
    
    
    df['Sell_count_tmp'] = np.where(df['Sell'] == 'NaN', 0, 1)
    df['Sell_count'] = np.add.accumulate(df['Sell_count_tmp'])
    
      
    df.Sell = pd.to_numeric(df.Sell, errors='coerce') 
    
    # total SELL dep on BUY/count

    Sell_count = abs(df['Sell_count'] )
    Buy_count = abs(df['Buy_count'])
        


    
    df.to_csv("mamacd.csv")
    return Buy, Sell, df.tot_buy, df.tot_sell, Buy_count, Sell_count
         
               


def create_dash(flask_app):
        #font_awesome = "https://use.fontawesome.com/releases/v5.10.2/css/all.css" 
        app=dash.Dash(server=flask_app, name="stock_dash", url_base_pathname=("/macd/"), prevent_initial_callbacks=True)

        GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 2000)
        
        
        app.layout = html.Div([
            
            ##### Store data
                dcc.Store(id='df_value'),
             #### Header 
                html.Div([
                html.Div([
                    html.Img(src = app.get_asset_url('logo.jpeg'),
                             style = {'height': '30px'},
                             className = 'title_image'),
                    html.H6('Have Fun !!! with stock ...',
                            style = {'color': 'white'},
                            className = 'title'),
        
                ], className = 'logo_title'),
                
                html.Div([
                   html.Div([
                       html.Div([
                           html.P('Buy:',
                                   style = {'color': 'white', 'font-size':'12px'},
                                   className = 'stock_label'),
                           html.Div(id = 'tot_buy',
                                   style = {'color': 'white', },
                                   className = 'stock_score'),
                       ],className = 'stock_score_label'),
                       html.P(['  /  '], className = 'stock_score'),
                       html.Div(id = 'buy_count',
                               style = {'color': 'white', },
                               className = 'stock_score'),
                       html.P([' ('], className = 'stock_score'),
                       html.Div(id = 'tot_profit',
                               style = {'color': 'white', },
                               className = 'stock_ave'),
                       html.P([')'], className = 'stock_score'),
                        
                   ], className = 'buy_sell_b'),
                   
                   html.Div([
                   html.Div([
                       html.P('Sell:',
                               style = {'color': 'white', 'font-size':'12px'},
                               className = 'stock_label'),
                       html.Div(id = 'tot_sell',
                               style = {'color': 'white'},
                               className = 'stock_score'),
                       ],className = 'stock_score_label'),
                       html.P(['  /  '], className = 'stock_score'),
                       html.Div(id = 'sell_count',
                               style = {'color': 'white', },
                               className = 'stock_score'),
                       html.P([' ('], className = 'stock_score'),
                       html.Div(id = 'tot_long',
                               style = {'color': 'white', },
                               className = 'stock_ave'),
                       html.P([')'], className = 'stock_score'),
                   ], className = 'buy_sell_s'),
                 
                 html.Div([
                   html.Div([
                       html.P('Profit:',
                               style = {'color': 'white', 'font-size':'12px'},
                               className = 'stock_label'),
                       html.Div(id = 'profit'),
                       ],className = 'stock_score_label'),
                   html.Div([
                       html.Div(id = 'Img'),
                       ],className = 'profit_img'),
                   ], className = 'buy_sell_p'),
                ], className = 'stock_score_container'),
                                              
                
                html.H6(id = 'get_date_time',
                        style = {'color': 'white'},
                        className = 'adjust_date_time'),
                ], className = 'title_date_time_container'),
        
                html.Div([
                dcc.Interval(id = 'update_date_time',
                             interval = 1000,
                             n_intervals = 0)
                ]),
            ##### button   
        
            
        
            
            ##### Graph Display
                html.Div([
                    html.Div([
                        html.Div([
                            dcc.Graph(id = 'price_candlesticker', animate=False, 
                                    style={'width': 'auto', 
                                           'height': '60vh',
                                           'border': '1px #5c5c5c solid',
                                           'margin-top' : "40px"},
                                    config = {'displayModeBar': True, 'responsive': True},
                                    className = 'chart_width'),
                                
                            ]),
                        ]),
                    
                        html.Div([
                            html.Title('MACD',
                            style = {'color': 'black'},
                            ),
                            
                            dcc.Graph(id = 'price_macd', animate=False, 
                                    style={'width': 'auto', 
                                           'height': '20vh',
                                           'border': '1px #5c5c5c solid',},
                                    config = {'displayModeBar': False, 'responsive': True},
                                    className = 'chart_width'),
                            ]),
                        
                        html.Div([
                            html.Title('Volume',
                            style = {'color': 'black'},
                            ),
                            
                            dcc.Graph(id = 'price_vol', animate=False, 
                                    style={'width': 'auto', 
                                           'height': '10vh',
                                           'border': '1px #5c5c5c solid',},
                                    config = {'displayModeBar': False, 'responsive': True},
                                    className = 'chart_width'),
                            ]),
                        
                        
                        html.Div([
                            html.Title('RSI',
                            style = {'color': 'black'},
                            ),
                            
                            dcc.Graph(id = 'price_rsi', animate=False, 
                                    style={'width': 'auto', 
                                           'height': '20vh',
                                           'border': '1px #5c5c5c solid',},
                                    config = {'displayModeBar': False, 'responsive': True},
                                    className = 'chart_width'),
                            ]),


        
                        dcc.Interval(id = 'update_value',
                                         interval = int(GRAPH_INTERVAL) ,
                                         n_intervals = 0)
                    ]),
        
            
                
        ])
        
        @app.callback(Output('df_value', 'data'), Input('update_value', 'n_intervals'))
        
        def update_df(n_intervals): 
                if n_intervals == 0:
                    raise PreventUpdate
                else:
                    
                    p_filename = "_out_stock_data.csv"
                    
                    #time_stamp = datetime.datetime.now() - datetime.timedelta(hours=13)
                    time_stamp =  datetime.datetime.now().strftime('%Y-%m-%d')
                    #time_stamp = datetime.now().strftime('%Y-%m-%d')
                    filename = time_stamp+" NQ=F USTime"+p_filename
                    
                    #filename = "2022-08-26 NQ=F USTime_out_stock_data.csv"
                    cwd = os.getcwd()
                    path = os.path.dirname(cwd)
                
                    file_path = path + "\\stock\\data\\"
                    file = os.path.join(file_path, filename)
                    #df = pd.read_csv(os.path.basename(filename))
                    df = pd.read_csv(file)
                    df.columns =['Datetime','Open','High','Low','Close', 'volume']
                    df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
                    df.ta.rsi(close=df['Close'], length=14, append=True, signal_indicators=True, xa=70, xb=30)
                    
        
                return df.to_dict('records')
        

        @app.callback(Output('get_date_time', 'children'),
                      [Input('update_date_time', 'n_intervals')])
        def live_date_time(n_intervals):
            if n_intervals == 0:
                raise PreventUpdate
            else:
                now = datetime.datetime.now()
                dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        
            return [
                html.Div(dt_string)
            ]
        
        @app.callback([
                Output('price_candlesticker', 'figure'),
                Output('tot_buy', 'children'),
                Output('tot_sell', 'children'),
                Output('buy_count', 'children'),
                Output('sell_count', 'children'),
                Output('tot_profit', 'children'),
                Output('tot_long', 'children'),
                ],
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data')])
        
        def update_graph(n_intervals, data):
        
            if n_intervals == 0:
                raise PreventUpdate
            else:
                
                df = pd.DataFrame(data)
                
                df['MA10'] = df.Close.rolling(10).mean()
                df['MA20'] = df.Close.rolling(20).mean()
                df['MA50'] = df.Close.rolling(50).mean()
                
                max = df.Close.max()
                min = df.Close.min()
                max_20 = df.Close.tail(20).max()
                min_20 = df.Close.tail(20).min()

                buy, sell, tot_buy, tot_sell, Buy_count, Sell_count = macd_strategy(df)
                
                tt_buy = tot_buy.iloc[-1]
                tt_sell =  tot_sell.iloc[-1]                
                bbuy_count =  Buy_count.iloc[-1]
                ssell_count =  Sell_count.iloc[-1] 

                ave_b = tt_buy / bbuy_count
                ave_b = np.around(ave_b, 2)

                ave_s = tt_sell / ssell_count
                ave_s = np.around(ave_s, 2)

                buymacd = np.array(buy).tolist()
                buy_macd = []
                for xs in buymacd:
                    for x in xs:
                        buy_macd.append(x)
                        
                sellmacd = np.array(sell).tolist()
                #sell_macd = sum(sellmacd, [])
                
                sell_macd = []
                for xs in sellmacd:
                    for x in xs:
                        sell_macd.append(x)
                        

                
                ###### Calculate profit START
                
                buy_list = []
                buy_idx = []
                for idx, xs in enumerate(buy_macd):
                    if xs != 'NaN':
                        buy_list.append(xs)
                        buy_idx.append(idx)
                
                sell_list = []
                sell_idx = []
                for idx, xs in enumerate(sell_macd):
                    if xs != 'NaN':
                        sell_list.append(xs)
                        sell_idx.append(idx)
                
                buy_nplist = np.column_stack((buy_list, buy_idx))
                sell_nplist =  np.column_stack((sell_list, sell_idx))
                
                min_len = np.minimum(len(buy_nplist), len(sell_nplist))
                
                
                tot_profit = 0 
                buy_profit = 0
                tot_long = 0
                long_profit = 0
                
                for i in range (min_len):
                    #buy-> sell
                    if buy_nplist[i,1] < sell_nplist[i,1]:
                        buy_profit = float(buy_nplist[i,0]) - float(sell_nplist[i,0])
                        print("buy_profit",buy_nplist[i,0], float(sell_nplist[i,0]), profit)
                    else:
                        long_profit = float(sell_nplist[i,0]) - float(buy_nplist[i,0])
                        print("long_profit",sell_nplist[i,0], float(buy_nplist[i,0]), long_profit)
                    #sell -> buy
                    
                    tot_profit += buy_profit
                    tot_long +=  long_profit
                    print("total", tot_profit, tot_long)
                
                ###### Calculate profit FINISH
                
                figure = go.Figure(
                    data = [go.Scatter(x=df.index, y=df.Close, line=dict(color='#fc0080', width=2), 
                            name = 'Close',
                            hoverinfo = 'text',
                            hovertext =
                            '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                            '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.Close] + '<br>'),
                    
                            go.Candlestick(x=df.index.tolist(),
                                open=df['Open'].tolist(),
                                high=df['High'].tolist(),
                                low=df['Low'].tolist(),
                                close=df['Close'].tolist(),
                                #yaxis="y2",,
                                name = "Candlestick"),  
                            
                            go.Scatter(x=df.index, y=df.MA10, line=dict(color='#AA76DB', width=1), 
                                name = 'MA10',
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                                '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.MA10] + '<br>'),
                            go.Scatter(x=df.index, y=df.MA20, line=dict(color='#2ed9ff', width=1),
                                name = 'MA20',
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                                '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.MA20] + '<br>'),
                            go.Scatter(x=df.index, y=df.MA50, line=dict(color='#b6e880', width=1),
                                name = 'MA50',
                                hoverinfo = 'text',
                                hovertext =
                                '<b>Time</b>: ' + df.Datetime.astype(str) + '<br>' +
                                '<b>Price</b>: ' + [f'{x:,.2f}' for x in df.MA50] + '<br>'),
                        
            
                go.Scatter(x=[0, len(df)], 
                         y=[min,min], name='min',
                         line=dict(color='rgba(152,78,163,0.5)', width=1, dash='dash'),
                         ),
            
                go.Scatter(x=[0, len(df)], 
                         y=[max,max], name='max',
                         line=dict(color='rgba(152,78,163,0.5)', width=1, dash='dash'),
                         ),
            
                go.Scatter(x=[0, len(df)], 
                         y=[min_20,min_20], name='min20',
                         line=dict(color='rgba(124,124,124,0.5)', width=1, dash='dash'),
                         ),
            
                go.Scatter(x=[0, len(df)], 
                         y=[max_20,max_20], name='max20',
                         line=dict(color='rgba(124,124,124,0.5)', width=1, dash='dash'),
                         ),

                go.Scatter(x=df.index, y=buy_macd, name="macd_up", mode="markers",
                              marker=dict(
                              symbol="3" ,
                              color="#fad51e",
                              size=10)), 

                go.Scatter(x=df.index, y=sell_macd, name="macd_dn", mode="markers",
                              marker=dict(
                              symbol="4" ,
                              color="#827ABC",
                              size=10)),

                                            
                    ],
                    
                    
                    layout = go.Layout(
                                   xaxis_rangeslider_visible=False,
                                   hovermode = 'closest',
                                   uirevision = 'dataset',
                                   margin = dict(t = 35  , r = 0, l = 60, b=20),
                                   xaxis = dict(autorange= True,
                                                color = 'black',
                                                matches = 'x',
                                                showspikes = True,
                                                showline = True,
                                                showgrid = True,
                                                linecolor = 'black',
                                                linewidth = 1,
                                                ticks = 'outside',
                                                tickfont = dict(
                                                    family = 'Arial',
                                                    size = 10,
                                                    color = 'black'
                                                )),
                                   yaxis = dict(autorange= True,
                                                showspikes = True,
                                                showline = True,
                                                showgrid = False,
                                                linecolor = 'black',
                                                linewidth = 1,
                                                ticks = 'outside',
                                                tickfont = dict(
                                                    family = 'Arial',
                                                    size = 10,
                                                    color = 'black'
                                       )),
        
                                   font = dict(
                                       family = 'sans-serif',
                                       size = 10,
                                       color = 'black'
                                   )
                                   
                                   )
                 )
                               
                return [figure, tt_buy, tt_sell, bbuy_count, ssell_count, tot_profit, tot_long] 
            
        @app.callback([Output('profit', 'children'),
                       Output('Img', 'children'),
                       ],
                      [Input('update_value', 'n_intervals'),
                       Input('tot_profit', 'children'),
                       Input('tot_long', 'children'),

                       ])
        
        def profit(n_intervals, buy_gain, sell_gain):
            
            if n_intervals == 0:
                raise PreventUpdate
            else:
                sell_gain = Decimal(str(sell_gain)).quantize(Decimal('0.01'))
                buy_gain = Decimal(str(buy_gain)).quantize(Decimal('0.01'))
                profit = buy_gain + sell_gain

                
                if (profit > 0):
                        return [html.H6('${0:,.2f}'.format(profit), style = {'color': '#84de02', 'fontSize' : 17}),
                                html.Img(id = "Img",src = app.get_asset_url('money-bag.png'),
                                     style = {'height': '30px'},
                                     className = 'coin'),]
                    
                else:
                        return [
                            html.H6(profit, style = {'color': '#f20540', 'fontSize' : 17}),
                            html.Img(id = "Img",src = app.get_asset_url('cry.png'),
                                     style = {'height': '30px'},
                                     className = 'coin'),]
                

            
        @app.callback(
                    Output('price_macd', 'figure'),
                    [Input('update_value', 'n_intervals'),
                     Input('df_value', 'data')])
        
        
        def update_macd(n_intervals, data): 
                if n_intervals == 0:
                    raise PreventUpdate
                else:
                    df = pd.DataFrame(data)
                    
                    return{
                        "data" : [
                                    go.Scatter(
                                            x=df.index,
                                            y=df['MACD_12_26_9'],
                                            line=dict(color='#ff9900', width=1),
                                            name='macd',
                                            # showlegend=False,
                                            legendgroup='2',),
        
                                    go.Scatter(
                                            x=df.index,
                                            y=df['MACDs_12_26_9'],
                                            line=dict(color='#000000', width=1),
                                            # showlegend=False,
                                            legendgroup='2',
                                            name='signal'),
                                    go.Bar(
                                            x=df.index,
                                            y=df['MACDh_12_26_9'],
                                        marker_color=np.where(df['MACDh_12_26_9'] < 0, '#000', '#ff9900'),
                                        name='bar'),
                                  
                                    go.Scatter(x=[0, len(df)], 
                                         y=[-5,-5], showlegend=False,
                                         line=dict(color='#000000', width=1, dash='dash'),
                                 ),
        
        
                                 ],
        
                        "layout" : go.Layout(
                                   hovermode = 'x unified',
                                   uirevision = 'dataset',
                                   margin = dict(t = 0  , r = 10, l = 60, b=0),
                        )
                              
                    }
                    
                
        @app.callback(
                Output('price_rsi', 'figure'),
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data')])
        
        def update_rsi(n_intervals, data):
            if n_intervals == 0:
                raise PreventUpdate
            else:
                
                df = pd.DataFrame(data)

                return {'data': [go.Scatter(x=df.index, y=df.RSI_14, name='RSI',
                                 line=dict(color='#000000', width=1),
                                 # showlegend=False,
                                 legendgroup='3'),
                                 
                                 go.Scatter(x=[0, len(df)], 
                                 y=[30,30], name='OB(30)',
                                 line=dict(color='#f705c3', width=2, dash='dash'),
                                 ),
                        
                                 go.Scatter(x=[0, len(df)], 
                                 y=[70,70], name='OS(70)',
                                 line=dict(color='#f705c3', width=2, dash='dash'),
                                 ),
                                 
                                 go.Scatter(x=[0, len(df)], 
                                     y=[50,50], showlegend=False,
                                     line=dict(color='#000000', width=1, dash='dash')),
                                 ],
                                
                         'layout': go.Layout(
                                            hovermode = 'x unified',
                                            uirevision = 'dataset', 
                                            margin = dict(t = 0  , r = 0, l = 60, b=0),
        
                            )
                                
                       }
            
        @app.callback(
                Output('price_vol', 'figure'),
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data')])
        
        def update_rsi(n_intervals, data):
            if n_intervals == 0:
                raise PreventUpdate
            else:
                
                df = pd.DataFrame(data)
                delta_vol = df.volume - df.volume.shift(1)

                return {'data': [go.Bar(x=df.index, y=delta_vol, name='volume',
                                  showlegend=True,
                                 legendgroup='4'),
                            
                                 ],
                                
                         'layout': go.Layout(
                                            hovermode = 'x unified',
                                            uirevision = 'dataset', 
                                            margin = dict(t = 0  , r = 0, l = 60, b=0),
        
                            )
                                
                       }


 
        return(app)

"""
if __name__ == '__main__':
    app.run_server()
    
"""
