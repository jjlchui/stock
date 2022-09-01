import plotly.graph_objects as go
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import statsmodels.api as sm
import numpy as np

app = Dash(__name__)

def slope20(ser, n=20):
    #"function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    for i in range(n, len(ser)+1):
        y=ser[i-n:i]
        x=np.array(range(n))
        y_scales = (y-y.min())/(y.max()-y.min())
        x_scales = (x-x.min())/(x.max()-x.min())
        x_scales = sm.add_constant(x_scales)
        model = sm.OLS(y_scales, x_scales)
        results = model.fit()
        
        slopes.append(results.params[-1])
        #results.summary()
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

df = pd.read_csv('D:\\Development\\flask dev\\stock\\data\\2022-07-22 NQ=F USTime_out_stock_data.csv')
df.drop(df.columns[[1, 2, 3]], axis=1, inplace=True)
df.columns =['Datetime','Close']

df['MA10'] = df.Close.rolling(10).mean()
df['MA20'] = df.Close.rolling(20).mean()
df['MA50'] = df.Close.rolling(50).mean()
df['slope'] = slope20(df.Close, 20)


fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df.Datetime, df.Close, df.MA10, df.MA20, df.MA50, df.slope],
               fill_color='lavender',
               align='left'))
    

])

fig.show()


