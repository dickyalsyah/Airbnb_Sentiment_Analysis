import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import operator
from sklearn.feature_extraction.text import CountVectorizer
# from gensim.models.word2vec import Word2Vec
# from sklearn.manifold import TSNE

def create_dashboard():
    app_name = 'dashboard'
    print('app_name = {}'.format(app_name))

    BS = "https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css"

    app = dash.Dash(name = 'mydash', requests_pathname_prefix='/dashboard/', 
                    external_stylesheets=[BS, '/static/css/style.css',
                    'https://fonts.googleapis.com/css2?family=Damion&display=swap'])

    navbar = dbc.NavbarSimple(
        children=[
            dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Sentiment Analysis", href="/", external_link=True)),
                dbc.NavItem(dbc.NavLink("Dataset", href="/dataset", external_link=True)),
                dbc.NavItem(dbc.NavLink("Dashboard", active=True, href="#")),
            ]
            )
        ],
        brand="dickyalsyah",
        brand_href="/",
        brand_style={
            "font-family" : "Damion",
            "font-weight" : "600",
            "font-size": "1.5em",
            # "letter-spacing" : "2px"
        },
        dark=True,
        style={
            "background-image": "linear-gradient(15deg, #13547a 0%, #80d0c7 100%)",
        },
    )

    intro = dbc.Container(
    [   
        dbc.Row(
            dbc.Col([
                html.Img(src = app.get_asset_url('logo.png'), style={'width':'170px'}),
            ])
        ),
        html.Hr(),
        html.P(
            "Dashboard for Sentiment Analysis Reviews in New York City"
        )
    ], style= {
        "padding-top": "2rem",
    },
    )

    df = pd.read_csv('http://kinilabs.com/data_airbnb/nyc_dashboard.csv.gz')
    df['date'] = pd.to_datetime(df['date'])
    time_series = df.copy()
    time_series.index = time_series['date']

    grouper = time_series.groupby(['sentiment_meaningful', pd.Grouper(key='date', freq='M')]).id.count()
    result = grouper.unstack('sentiment_meaningful').fillna(0)

    pos_num = 1000314
    neg_num = 25578
    neu_num = 198206

    row1 = dbc.Container(
        dbc.Row(
            [
            dbc.Col(
                [
                    dcc.Graph(
                        id = "stream",
                        figure = {
                            "data" : [  
                                go.Scatter(
                                    x = result.index,
                                    y = result['Positive'],
                                    name = 'Positive',
                                    opacity=0.8,
                                    mode='lines',
                                    line=dict(width=0.5, color='rgb(131, 90, 241)'),
                                    stackgroup='one'
                                ),
                                go.Scatter(
                                    x = result.index,
                                    y = result['Negative'],
                                    name = 'Negative',
                                    opacity=0.8,
                                    mode='lines',
                                    line=dict(width=0.5, color='rgb(255, 50, 50)'),
                                    stackgroup='two'
                                ),
                                go.Scatter(
                                    x = result.index,
                                    y = result['Neutral'],
                                    name = 'Neutral',
                                    opacity=0.8,
                                    mode='lines',
                                    line=dict(width=0.5, color='rgb(184, 247, 212)'),
                                    stackgroup='three'
                                )
                            ]
                        }
                    )
                ], width = 9, style = {'display': 'inline-block', 'padding': '0 0 0 20'},
            ),
            dbc.Col(
                [
                    dcc.Graph(
                        id = "pie",
                        figure = {
                            'data': [
                                go.Pie(
                                    labels = ['Positives', 'Negatives', 'Neutrals'], 
                                    values = [pos_num, neg_num, neu_num],
                                    name = "View Metrics",
                                    marker_colors = ['rgba(131, 90, 241, 0.6)','rgba(255, 50, 50, 0.6)','rgba(184, 247, 212, 0.6)'],
                                    textinfo = 'value',
                                    hole = .65)
                            ],
                            'layout':{
                                'showlegend':False,
                                'title':'Sentiment Count',
                                'annotations':[
                                    dict(
                                        text = '{0:.1f}K'.format((pos_num+neg_num+neu_num)/1000),
                                        font = dict(
                                            size = 20
                                        ),
                                        showarrow = False
                                    )
                                ]
                            }
                        }
                    )
                ], width = 3, style = {'display': 'inline-block'}
            )
            ], no_gutters = True,
        ), fluid = True
    )

    nyc_listing = pd.read_csv('http://data.insideairbnb.com/united-states/ny/new-york-city/2020-06-08/visualisations/listings.csv')    
    
    row2 = dbc.Container(
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Form(
                            [
                                dbc.FormGroup(
                                    [
                                        dbc.Label("Mood", className="mr-2"),
                                        dbc.Select(
                                            id = "mood",
                                            options = [i for i in [
                                                            { 'label': 'Positive', 'value': 'Positive' },
                                                            { 'label': 'Neutral', 'value': 'Neutral' },
                                                            { 'label': 'Negative', 'value': 'Negative' }
                                                            ]],
                                            value = "Positive",
                                            className = "col-xs-3"
                                        ),
                                    ],
                                    className = "mr-3", style = {"display": "flex", "flexWrap": "wrap"}
                                ),
                                dbc.FormGroup(
                                    [
                                        dbc.Label("Count Words", className="mr-2"),
                                        dbc.Input(id = "n_words", type = "number", value = 10),
                                    ],
                                    className="mr-3",
                                ),
                                dbc.Button("Search", id = "buttonword", color="info"),
                            ],
                            inline=True,
                        ),
                        dcc.Graph(id="Word-Freq")
                    ], width = 7, style={'display': 'inline-block', 'padding': '0 0 0 20'}
                ),
                dbc.Col([
                    dcc.Graph(
                        figure = {
                            'data' : [
                                go.Scattergeo(
                                    lon = nyc_listing['longitude'],
                                    lat = nyc_listing['latitude'],
                                    text = nyc_listing['id'],
                                    mode = 'markers',
                                    marker_color = nyc_listing['number_of_reviews'],
                                    )
                            ],
                            'layout' : go.Layout(
                                title = {'text' : 'Mapping number of reviews', 'x':0.5},
                                geo_scope = 'usa',
                            )
                        }
                    )
                ], width=4)
            ], no_gutters=True,
        )
    )
    # Initialize word2vec and fit to tsne
    # model_w2v = Word2Vec.load('data/w2v_train_model_balance.w2v')
    # word_vectors = [model_w2v[w] for w in model_w2v.wv.vocab.keys()]
    # tsne_model = TSNE(n_components=2)
    # tsne_results = tsne_model.fit_transform(word_vectors)
    # tsne_df = pd.DataFrame(tsne_results, columns=['x', 'y'])
    # tsne_df['words'] = model_w2v.wv.vocab.keys()
    tsne_df = pd.read_csv('data/tsne_w2v.csv')

    row3 = dbc.Container(
        dbc.Row(
        [
            dbc.Col(
                dcc.Graph(
                    figure = {
                        'data' : [
                            go.Scatter(
                                x = tsne_df['x'],
                                y = tsne_df['y'],
                                name = 'TSNE',
                            #     hoveron = Target,
                                mode = 'markers',
                                text = tsne_df['words'],
                                hovertemplate = 'Word: %{text}',
                                showlegend = True,
                                marker = dict(
                                    size = 8,
                                    # color = Target.unique(),
                                    colorscale ='Jet',
                                    showscale = False,
                                    opacity = 0.8,
                                    line = dict(
                                        width = 1,
                                        color = 'rgb(255, 255, 255)'
                                    )
                                )
                            )
                        ],
                        'layout' : go.Layout(
                            title = {'text' : 'W2V TSNE (T-Distributed Stochastic Neighbour Embedding)', 'x':0.5},
                            hovermode= 'closest',
                            yaxis = dict(zeroline = False),
                            xaxis = dict(zeroline = False),
                            showlegend= False,    
                        )
                    }
                ), style={'display': 'inline-block', 'padding': '0 0 0 20'}   
            )
        ]
        )
    )

    tab1_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("Wordcloud from Sentiment", style={'text-align': 'center', 'font-size':'20px'}),
            html.Img(src=app.get_asset_url('Positive.png'), style = {'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
        ]
    ),
    className="mt-3",
    )

    tab2_content = dbc.Card(
        dbc.CardBody(
            [
                html.P("Wordcloud from Sentiment", style={'text-align': 'center', 'font-size':'20px'}),
                html.Img(src=app.get_asset_url('Neutral.png'), style = {'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
            ]
        ),
        className="mt-3",
    )

    tab3_content = dbc.Card(
        dbc.CardBody(
            [
                html.P("Wordcloud from Sentiment", style={'text-align': 'center', 'font-size':'20px'}),
                html.Img(src=app.get_asset_url('Negative.png'), style = {'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}),
            ]
        ),
        className="mt-3",
    )

    row4 = dbc.Container(
        dbc.Row([
            dbc.Col(
                dbc.Tabs(
                    [   
                        dbc.Tab(tab1_content, label="Positive"),
                        dbc.Tab(tab2_content, label="Neutral"),
                        dbc.Tab(tab3_content, label="Negative"),
                    ]
                )
            )
        ]), style={'padding-bottom': '2rem'}
    )

    footer = dbc.Container(
        dbc.Row([
                dbc.Col(html.Div("2020 Dicky Alamsyah"), width=4),
                dbc.Col(html.Div(dcc.Link('Back to Top', href='#'), style={'text-align': 'right'}), width=4)
            ], justify= 'between'
        ), style={'padding-bottom': '2rem'}
    )

    app.title = 'Sentiment Analysis by Dicky Alamsyah'

    app.layout = html.Div([navbar, intro, row1, row2, row3, row4, footer])

    @app.callback(
        Output("Word-Freq", component_property='figure'),
        [Input("buttonword", "n_clicks")],
        [
            State("mood", 'value'),
            State("n_words",'value')
        ]
    )

    def word_freq(n_clicks, sentiment, n):
        df = pd.read_csv('http://kinilabs.com/data_airbnb/nyc_dashboard.csv.gz')
        df_en = df[(df['language']=='en')]
        reviews = (df_en[df_en['sentiment_meaningful']==sentiment]['comments_meaningful']).tolist()
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(reviews)
        freq = np.ravel(X.sum(axis=0))

        vocab = [v[0] for v in sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))]
        fdist = dict(zip(vocab, freq))
        df_word = pd.DataFrame(fdist.items(), columns=['word','freq'])
        show_word = df_word.sort_values(by='freq', ascending=False)[:n]

        figure_word = {
            'data' : [
                go.Bar(
                    x = show_word['freq'].iloc[::-1],
                    y = show_word['word'].iloc[::-1],
                    name = sentiment,
                    orientation = 'h',
                    marker_color= 'rgba(131, 90, 241, 0.6)'
                )
            ],
            'layout' : go.Layout(
                paper_bgcolor = 'rgba(0, 0, 0, 0)',
                hovermode = 'closest',
                autosize=False,
                title= {'text': 'Word Frequency in Sentiment Analysis', 'x': 0.5},
            )
        }
        
        return figure_word

    return app