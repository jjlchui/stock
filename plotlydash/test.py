import dash
from dash.dependencies import Input, Output
from dash import dcc, html, callback_context
import dash_core_components as dcc
import datetime

def getLayout(flask_app):
    app=dash.Dash(server=flask_app, name="stock_dash", url_base_pathname=("/test/"))

    app.layout = html.Div([
        html.Div(id='my-output-interval'),
        dcc.Interval(id='my-interval', interval=5),
    ])


    @app.callback(
        Output('my-output-interval', 'children'),
        [Input('my-interval', 'n_intervals')])
    def display_output(n):
        now = datetime.datetime.now()
        return '{} intervals have passed. It is {}:{}:{}'.format(
            n,
            now.hour,
            now.minute,
            now.second
        )
    return(app)


