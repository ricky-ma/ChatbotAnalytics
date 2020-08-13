import base64
import io
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table

from analysis import get_outliers, load_data, novel_df, analyze_mkts, confidence_over_time

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.scripts.config.serve_locally = True
application = app.server
raw_data = None
embedded_data = pd.DataFrame()
novel = novel_df()
mkt_analysis = analyze_mkts(novel)
time_series = confidence_over_time()


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
        if is_vec:
            df = df.drop(df.columns[0], axis=1)
        else:
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
        return df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])


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


def display_scatter():
    print("Plotting scatterplot")
    fig = px.scatter(embedded_data, x='x', y='y', color='FAQ_id', hover_name='question', title='UMAP Visualization')
    # fig = px.scatter_3d(dataframe, x='x', y='y', z='z', color='FAQ_id', hover_name='question')
    return html.Div(
        [
            dcc.Graph(
                id='scatterplot',
                figure=fig
            ),
        ],
        style={"margin-left": "5%", "margin-right": "5%"}
    )


def display_outliers():
    outliers = get_outliers(raw_data, embedded_data)
    fig_outlier = go.Figure(
        data=go.Scatter(
            x=outliers['x'],
            y=outliers['y'],
            mode='markers',
            text=outliers['question'],
            marker=dict(symbol=outliers['outlier_score'], color=outliers['FAQ_id'])
        ),
    )
    fig_outlier.update_layout(
        title='Outliers'
    )
    table_data = outliers.drop(['outlier_score'], axis=1)[outliers['outlier_score'] == 4]
    return html.Div(
        [
            dcc.Graph(
                id='scatterplot-outliers',
                figure=fig_outlier,
            ),
            dash_table.DataTable(
                id='outlier_table',
                columns=[{"name": i, "id": i} for i in table_data.columns],
                data=table_data.to_dict('records'),
                filter_action="native",
                sort_action="native",
                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                export_format="csv",
                export_headers="display",
            ),
        ],
        style={"margin-left": "5%", "margin-right": "5%"}
    )


def display_novelty():
    histogram = px.histogram(novel, x='score', color='dataset', title='Histogram of Novelty Scores')
    mkt_feedback_bar = display_market_feedback()
    mkt_novel_bar = display_market_novelty()
    time_series_line = display_time_series()
    return html.Div(
        children=[
            html.Div(dcc.Graph(id='novelty_hist', figure=histogram)),
            dbc.Row([
                dbc.Col(dcc.Graph(id='mkt_feedback_bar', figure=mkt_feedback_bar), width=6),
                dbc.Col(dcc.Graph(id='mkt_novel_bar', figure=mkt_novel_bar), width=6),
                dbc.Col(dcc.Graph(id='conf_line', figure=time_series_line), width=12),
            ]),
            html.H3('Novelty Scores'),
            dash_table.DataTable(
                id='novelty_table',
                columns=[{"name": i, "id": i} for i in novel.columns],
                data=novel.to_dict('records'),
                fixed_rows={'headers': True},
                style_table={'height': 500},
                filter_action="native",
                sort_action="native",
                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                export_format="csv",
                export_headers="display",
            )
        ],
        style={"margin-left": "5%", "margin-right": "5%", "margin-top": "5%", "margin-bottom": "5%"}
    )


def display_market_feedback():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mkt_analysis['market'],
                         y=mkt_analysis['positive feedbacks'],
                         name='positive feedbacks',
                         marker_color='rgb(26, 118, 255)'
                         ))
    fig.add_trace(go.Bar(x=mkt_analysis['market'],
                         y=mkt_analysis['negative feedbacks'],
                         name='negative feedbacks',
                         marker_color='rgb(55, 83, 109)'
                         ))
    fig.add_trace(go.Bar(x=mkt_analysis['market'],
                         y=mkt_analysis['something else'],
                         name='something else',
                         marker_color='rgb(20, 40, 150)'
                         ))
    fig.update_layout(
        title='Chatbot Feedback by Market',
        xaxis_tickfont_size=14,
        yaxis=dict(title='count', titlefont_size=16, tickfont_size=14),
        legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
        barmode='group',
        bargap=0.15,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.1  # gap between bars of the same location coordinate.
    )
    return fig


def display_market_novelty():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mkt_analysis['market'],
                         y=mkt_analysis['avg top intent confidence'],
                         name='avg top intent confidence',
                         marker_color='rgb(26, 118, 255)'
                         ))
    fig.add_trace(go.Bar(x=mkt_analysis['market'],
                         y=mkt_analysis['avg novelty score'],
                         name='avg novelty score',
                         marker_color='rgb(55, 83, 109)'
                         ))
    fig.update_layout(
        title='Chatbot Confidence and Novelty by Market',
        xaxis_tickfont_size=14,
        yaxis=dict(title='count', titlefont_size=16, tickfont_size=14),
        legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
        barmode='group',
        bargap=0.15,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.1  # gap between bars of the same location coordinate.
    )
    return fig


def display_time_series():
    fig = px.line(time_series, x="timestamp", y="confidence", title="Weekly Avg Chatbot Confidence"
                  # color="market", line_group="market", hover_name="market"
                  )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


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
            return display_outliers()
        except Exception as e:
            print(e)
            return html.Div(
                    children="Please load data.",
                    style={"margin-left": "25px", "margin-top": "25px"},
                )


@app.callback(Output("tabs-figures", "children"), [Input("tabs", "value")])
def render_tab(tab):
    try:
        if tab == "tab-0":
            return html.Div([
                display_scatter(),
                display_outliers(),
            ])

        elif tab == "tab-1":
            return display_novelty()

    except Exception as e:
        print(e)
        return html.Div(
            children="Please load data.",
            style={"margin-left": "25px", "margin-top": "25px"},
        )


app.config["suppress_callback_exceptions"] = True
app.layout = html.Div(children=[
    html.H1(
        children='Multilingual Chatbot Analytics',
        style={"margin-top": "5%", "margin-left": "5%"}
    ),

    html.H3(
        children='Visualization, anomaly detection, and novelty detection for multilingual text.',
        style={"margin-left": "5%"},
        className="text-muted"
    ),
    html.Hr(),
    dcc.Upload(
        id='upload-data',
        children=dbc.Button('Upload Files', style={"margin-left": "5%"}, className="btn btn-secondary btn-lg"),
        multiple=True),
    html.Div(
        'Select one metadata file with text data and one file with the corresponding vector embeddings.',
        style={"margin-left": "5%"}
    ),
    html.Hr(),
    html.Div(id='output-data-upload'),

    dcc.Tabs(
        id="tabs",
        value="tab-0",
        children=[
            dcc.Tab(label="Model Analysis", value="tab-0"),
            dcc.Tab(label="Feedback Analysis", value="tab-1"),
        ],
    ),
    html.Div(id="tabs-figures"),
])

if __name__ == '__main__':
    application.run(host="0.0.0.0")
