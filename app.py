import base64
import io
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from analysis import get_outliers, load_data

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
raw_data = None
embedded_data = pd.DataFrame()


def parse_content(content, filename, is_vec):
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), header=None)
        elif 'tsv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter='\t', header=None)
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded), header=None)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    if is_vec:
        df = df.drop(df.columns[0], axis=1)
    else:
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
    return df


def parse_contents(list_of_contents, list_of_names):
    for contents, filename in zip(list_of_contents, list_of_names):
        if 'vec' in filename:
            df_vec = parse_content(contents, filename, True)
            print(df_vec.head)
        if 'meta' in filename:
            df_meta = parse_content(contents, filename, False)
            print(df_meta.head)

    try:
        global raw_data, embedded_data
        raw_data, embedded_data = load_data(df_vec, df_meta)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])


def display_outliers(data):
    fig_outlier = go.Figure(data=go.Scatter(
        x=data['x'],
        y=data['y'],
        mode='markers',
        text=data['question'],
        marker=dict(symbol=data['outlier_score'], color=data['FAQ_id'])
    ))
    table_data = data.drop(['outlier_score'], axis=1)[data['outlier_score'] == 4]
    return html.Div([
        dcc.Graph(
            id='scatterplot-outliers',
            figure=fig_outlier
        ),
        dash_table.DataTable(
            id='outlier_table',
            columns=[{"name": i, "id": i} for i in table_data.columns],
            data=table_data.to_dict('records'),
            filter_action="native",
            sort_action="native"
        )
    ])


def display_scatter():
    print("Plotting scatterplot")
    fig = px.scatter(embedded_data, x='x', y='y', color='FAQ_id', hover_name='question')
    # fig = px.scatter_3d(dataframe, x='x', y='y', z='z', color='FAQ_id', hover_name='question')
    return html.Div([
        dcc.Graph(
            id='scatterplot',
            figure=fig
        ),
    ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(list_of_contents, list_of_names)
        ]
        return children


@app.callback(
    dash.dependencies.Output('outlier-list', 'children'),
    [dash.dependencies.Input('outliers-btn', 'n_clicks')])
def update_output(n_clicks):
    print(n_clicks)
    if n_clicks > 0:
        try:
            outliers = get_outliers(raw_data, embedded_data)
            return display_outliers(outliers)
        except Exception as e:
            print(e)
            return html.Div([
                'Please load data'
            ])


app.layout = html.Div(children=[
    html.H1(children='Multilingual NLP Explorer'),

    html.Div(children='''
        Exploratory data analysis for multilingual NLP.
    '''),
    html.Hr(),
    dcc.Upload(id='upload-data', children=html.Button('Upload Files'), multiple=True),
    html.Hr(),

    html.Div(id='output-data-upload'),

    html.Button('Get Local Outliers', id='outliers-btn', n_clicks=0),
    html.Div(id='outlier-list'),
])

if __name__ == '__main__':
    app.run_server(debug=True)
