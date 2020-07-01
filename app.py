import base64
import io
import plotly.express as px
import pandas as pd
import umap
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from sklearn.preprocessing import StandardScaler


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'tsv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter='\t')
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    df = df.drop(df.columns[0], axis=1)
    return load_data(df)


def load_data(df):
    reducer = umap.UMAP(n_components=3)
    data = df[df.columns].values
    scaled_data = StandardScaler().fit_transform(data)
    embedding = reducer.fit_transform(scaled_data)
    print(embedding)
    print(embedding.shape)
    return make_figure(pd.DataFrame(embedding, columns=['x', 'y', 'z']))


def make_figure(dataframe):
    fig = px.scatter_3d(dataframe, x='x', y='y', z='z')
    return html.Div([
        dcc.Graph(
            id='scatter',
            figure=fig
        )
    ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output(content, name):
    if content is not None:
        children = [
            parse_contents(content, name)]
        return children


app.layout = html.Div(children=[
    html.H1(children='Multilingual NLP Explorer'),

    html.Div(children='''
        Exploratory data analysis for multilingual NLP.
    '''),
    html.Hr(),
    dcc.Upload(id='upload-data', children=html.Button('Upload File'), multiple=False),
    html.Hr(),

    html.Div(id='output-data-upload'),

])

if __name__ == '__main__':
    app.run_server(debug=True)