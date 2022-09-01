from dash import dcc, html, callback_context, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import dash
import datetime
import time
import os
import plotly
import pandas as pd
import pandas_ta as ta
from dash.exceptions import PreventUpdate
import numpy as np
from dash_extensions import WebSocket
import statsmodels.api as sm
import plotly.figure_factory as ff


#https://stackoverflow.com/questions/55443071/rolling-ols-using-time-as-the-independent-variable-with-pandas

# function to make a useful time structure as independent variable

def myTime(date_time_str):
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    return(time.mktime(date_time_obj.timetuple()))

def RegressionRoll(df, subset, dependent, independent, const, win):
    
    """
    Parameters:
    ===========
    df -- pandas dataframe
    subset -- integer - has to be smaller than the size of the df or 0 if no subset.
    dependent -- string that specifies name of denpendent variable
    independent -- LIST of strings that specifies name of indenpendent variables
    const -- boolean - whether or not to include a constant term
    win -- integer - window length of each model

    df_rolling = RegressionRoll(df=df, subset = 0, 
                                dependent = 'Price', independent = ['Time'],
                                const = False, win = 3)
    """

    # Data subset
    if subset != 0:
        df = df.tail(subset)
    else:
        df = df

    # Loopinfo
    end = df.shape[0]+1
    win = win
    rng = np.arange(start = win, stop = end, step = 1)

    # Subset and store dataframes
    frames = {}
    n = 1

    for i in rng:
        df_temp = df.iloc[:i].tail(win)
        newname = 'df' + str(n)
        frames.update({newname: df_temp})
        n += 1

    # Analysis on subsets
    df_results = pd.DataFrame()
    for frame in frames:

        #debug
        #print(frames[frame])

        # Rolling data frames
        dfr = frames[frame]
        y = dependent
        x = independent

        # Model with or without constant
        if const == True:
            x = sm.add_constant(dfr[x])
            model = sm.OLS(dfr[y], x).fit()
        else:
            model = sm.OLS(dfr[y], dfr[x]).fit()

    # Retrieve price and price prediction
    #Prediction = model.predict()[-1]
    # Retrieve price and price prediction
    #d = {'Price':dfr['Price'].iloc[-1], 'Predict':Prediction}
    #df_prediction = pd.DataFrame(d, index = dfr['Date'][-1:])
    
    Pre_list=[]
    for i in range(win):
        Prediction = model.predict()[i]
        Pre_list.append(Prediction)
        
    d = {'Price':dfr['Price'].iloc[:].values, 'Predict':Pre_list}
    df_prediction = pd.DataFrame(d, index = dfr['Date'][:])

    # Retrieve parameters (constant and slope, or slope only)
    theParams = model.params[0:]
    coefs = theParams.to_frame()
    df_temp = pd.DataFrame(coefs.T)
    df_temp.index = dfr['Date'][-1:]

    # Build dataframe with Price, Prediction and Slope (+constant if desired)
    df_temp2 = pd.concat([df_prediction, df_temp], axis = 1)
    df_temp2=df_temp2.rename(columns = {'Time':'Slope'})
    df_results = pd.concat([df_results, df_temp2], axis = 0)

    return(df_results)

def generate_table(dataframe, max_rows=10):
    return html.Table(
    # Header
    [html.Tr([html.Th(col) for col in dataframe.columns]) ] +
    # Body
    [html.Tr([
        html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
    ]) for i in range(min(len(dataframe), max_rows))]
)


def create_dash(flask_app):
    
        app=dash.Dash(server=flask_app, name="stock_dash", url_base_pathname=("/regression/"), prevent_initial_callbacks=True)

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
                        html.Div(id='table-container', className ='table_style'),

                        

                        dcc.Interval(id='update_value',interval=1000,n_intervals=0 ),
                        
                        #generate_table(df)

                    ]),
        
            
                
        ])
                
        @app.callback(Output('df_value', 'data'), Input('update_value', 'n_intervals'))
        
        def update_df(n_intervals): 
                if n_intervals == 0:
                    raise PreventUpdate
                else:
                    
                    p_filename = "_out_stock_data.csv"
                    
                    time_stamp = datetime.datetime.now() - datetime.timedelta(hours=13)
                    time_stamp =  time_stamp.strftime('%Y-%m-%d')
                    #time_stamp = datetime.now().strftime('%Y-%m-%d')
                    filename = time_stamp+" NQ=F USTime"+p_filename
                    
                    #filename = "2022-06-29 NQ=F_out_stock_data.csv"
                    cwd = os.getcwd()
                    path = os.path.dirname(cwd)
                
                    file_path = path + "\\stock\\data\\"
                    file = os.path.join(file_path, filename)
                    #df = pd.read_csv(os.path.basename(filename))
                    df = pd.read_csv(file, usecols=[0,4], names=['Date', 'Price'] )
                    #df.drop('Open', 'High', 'Low', inplace=True, axis=1)
                    #df.columns =['Date','Price']
                    #df.ta.macd(close=df['Close'], fast=12, slow=26, signal=9, append=True)
                    #df.ta.rsi(close=df['Close'], length=14, append=True, signal_indicators=True)
                    
        
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
        
        


        @app.callback(
                Output('table-container', 'children'),
                [Input('update_value', 'n_intervals'),
                 Input('df_value', 'data')])
        
        def update_table(n_intervals,data):
            
            if n_intervals == 0:
                raise PreventUpdate
            else:
            
                df = pd.DataFrame(data)
                                                
                df['Date'] = pd.to_datetime(df['Date'],format="%Y-%m-%d %H:%M:%S")  
                time_list=[]

                for obs in df['Date']:
                    date_time_obj = time.mktime(obs.timetuple())
                    time_list.append(date_time_obj)

                df['Time'] = time_list
 
                df_rolling = RegressionRoll(df=df, subset = 0, 
                            dependent = 'Price', independent = 'Time',
                            const = False, win = 10) 
                
                df_rolling['DateTime'] = df_rolling.index
                df_rolling.reset_index(drop=True)

                r_price = df_rolling.Price
                r_prices1  = df_rolling.Price.shift(1)
                df_rolling['Delta_Price'] = r_price - r_prices1 
                
                p_predict = df_rolling.Predict
                p_predicts1 = df_rolling.Predict.shift(1)
                df_rolling['Delta_Predict'] = df_rolling.Predict - df_rolling.Predict.shift(1)
                
                #df_rolling = df_rolling[['Date', 'Price', 'Predict', 'Delta_Price', 'Delta_Predict', 'Slope']]

                
                df = df.tail(10)
                return generate_table(df_rolling)
 
        return(app)

"""
if __name__ == '__main__':
    app.run_server()
    
"""
