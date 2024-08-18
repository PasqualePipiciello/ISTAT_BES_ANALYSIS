from dash import Dash, dcc, html, Input, Output, callback, dash_table, State
from sklearn.preprocessing import StandardScaler
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import utils
import os
import json
from cluster_wise import *

from regression_section import create_regression_section
from utils import create_ag_grid

data = pd.read_excel("raw_data.xlsx")
years = list(np.linspace(2004, 2023, 20).astype(int))
files = os.listdir("BES_DATA")
dict_preprocessed_data = {str(year): pd.read_csv("BES_DATA/" + file, index_col=0).drop(
    ["North", "North-west", "North-east", "Centre", "South and islands", "South", "Islands", "Italy"], axis=0) for
    year, file in zip(years, files)}

domains = list(np.unique(data["DOMAIN"]))
all_options = utils.create_domain_indicator_mapping(data)

with open('italian_provinces.geojson') as response:
    geojson_data = json.load(response)
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']

style = {'backgroundColor': 'blue'}

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H1(children='BES Dashboard', style={'textAlign': 'center', 'margin-bottom': '20px'}),

    html.Div(children=[
        html.Img(src="/assets/istat.png", style={'width': '30%', 'height': 'auto', 'margin-bottom': '20px'}),
        html.Figcaption("ISTAT Logo")
    ], style={'textAlign': 'center'}),

    dcc.Markdown('''
    **BES at Local Level** is a system of equitable and sustainable well-being indicators at small-regions level 
    that are consistent with the national BES measures. Istat issues BES measures at local level to deepen 
    the knowledge on the well-being distribution across Italy to assess territorial inequalities in more detail.

    To meet the statistical information needs of local communities, Istat designed BES at local level in 
    cooperation with local authorities, investigating the specific information needs of Italian Municipalities, 
    Provinces, and Metropolitan Cities and tuning a shared theoretical framework.

    BES measures at local level maintain a high level of quality and consistency with the BES indicators system 
    and constantly follow the evolution of the BES framework.

    The two frameworks share a core of common and harmonized indicators. In addition, BES at local level includes 
    specific well-being indicators, concerning some issues that are related to responsibilities and functions 
    of local authorities.
    ''', style={'margin-bottom': '20px', 'textAlign': 'justify'}),

    html.H5("Select a year"),

    dcc.Slider(
        min=min(years),
        max=max(years),
        step=1,
        marks={str(year): str(year) for year in years},
        value=max(years),
        id='Year',
    ),
    html.Br(),
    html.Div(children=[
        html.Div(children=[
            html.H5("Select a domain"),
            dcc.RadioItems(
                options=[{'label': category, 'value': category} for category in all_options.keys()],
                value='Health',
                id='categories-radio',
                style={'display': 'block'}
            )
        ], style={'flex': 1, 'padding': '10px'}),

        html.Div(children=[
            html.H5("Select an indicator"),
            dcc.RadioItems(
                id='indicators-radio',
                style={'display': 'block'}
            )
        ], style={'flex': 1, 'padding': '10px'})
    ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'}),

    dbc.Container([
        dbc.Row([
            dbc.Col(create_ag_grid(id="Table"), width=5
                    ),
            dbc.Col(
                html.Div(children=[
                    dcc.Graph(id='Italy map', style={'height': '50vh'}),
                ], style={'margin-bottom': '20px'}), width=7)

        ])
    ]),
    create_regression_section(domains, years)
])


@app.callback(
    Output('indicators-radio', 'options'),
    Input('categories-radio', 'value'),
    Input('Year', 'value')
)
def set_indicators_options(selected_category, selected_year):
    preprocessed_data = dict_preprocessed_data.get(str(selected_year))
    if preprocessed_data is not None:
        available_indicators = [indicator for indicator in all_options.get(selected_category, []) if
                                indicator in preprocessed_data.columns]
        return [{'label': indicator, 'value': indicator} for indicator in available_indicators]
    return []


@app.callback(
    Output('indicators-radio', 'value'),
    Input('indicators-radio', 'options')
)
def set_indicators_value(available_options):
    if available_options:
        return available_options[0]['value']
    return None


@app.callback(
    [Output('Italy map', 'figure'),
     Output('Table', "rowData"),
     Output('Table', 'columnDefs')],
    [Input('indicators-radio', 'value'),
     Input('Year', 'value')]
)
def update_map(indicator, year):
    year_str = str(year)  # Ensure year is a string for data lookup
    preprocessed_data = dict_preprocessed_data.get(year_str)

    if preprocessed_data is None or indicator not in preprocessed_data.columns:
        return px.choropleth(title="No Indicator Selected"), [], []

    data = preprocessed_data[[indicator]].reset_index()
    data.columns = ['Province', indicator]

    fig = px.choropleth(
        data_frame=data,
        geojson=geojson_data,
        locations='Province',
        featureidkey="properties.prov_name",  # Adjust this to match the GeoJSON file's property key
        color=indicator,
        color_continuous_scale="Magenta",
        labels={'Value': indicator + " in " + str(year)}
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update(layout_coloraxis_showscale=False)

    return fig, data.to_dict("records"), [{"field": i} for i in data.columns]


@app.callback(
    Output("x-variable-indicator", "options"),
    Input("x-variable_domain", "value"),
    Input('select-year', "value")
)
def updateX_domain(selected_domains, year):
    if selected_domains is None or len(selected_domains) == 0:
        return []

    preprocessed_data = dict_preprocessed_data.get(str(year))
    available_indicators = []

    # Concatenate indicators for all selected domains
    for domain in selected_domains:
        indicators = all_options.get(domain, [])
        available_indicators.extend([indicator for indicator in indicators if indicator in preprocessed_data.columns])

    # Remove duplicates by converting to a set and back to a list
    available_indicators = list(set(available_indicators))

    return [{"label": indicator, "value": indicator} for indicator in available_indicators]


@app.callback(
    Output("y-indicator", "options"),
    Input("y-variable_domain", "value"),
    Input('select-year', "value")
)
def updateY_domain(selected_domain, year):
    if selected_domain is None:
        return []

    preprocessed_data = dict_preprocessed_data.get(str(year))
    available_indicators = [indicator for indicator in all_options.get(selected_domain, []) if
                            indicator in preprocessed_data.columns]

    return [{"label": indicator, "value": indicator} for indicator in available_indicators]


@app.callback(
    Output("df_res", "rowData"),
    Output("df_res", "columnDefs"),
    Output("regression-graph", "figure"),
    Input("confirm-button", "n_clicks"),
    State('select-year', "value"),
    State('y-indicator', 'value'),
    State('x-variable-indicator', "value"),
    State("cluster-count", "value"),
    prevent_initial_call=True
)
def update_res_dataframe(_, year, y_indicator, X_names, k):
    # Perform the regression analysis
    data_to_analyze = dict_preprocessed_data[str(year)]
    X = data_to_analyze[X_names].to_numpy()
    y = np.array(data_to_analyze[y_indicator].values)
    X = StandardScaler().fit_transform(X)
    y = np.squeeze(StandardScaler().fit_transform(y.reshape(-1, 1)))
    model = ClusterwiseLR(int(k), gamma=0, max_iter=100)
    model.fit(X, y)
    X_names = ["Intercept"] + list(X_names) + ["R2"]
    # Get the performance dataframe
    df_res = model.performance_dataframe(X_names)
    df_res["Cluster"] = [f"Cluster{i}" for i in range(int(k))]
    df_res["Cluster_Count"] = [np.sum(model.labels == cluster) for cluster in range(int(k))]
    # Prepare data for AgGrid
    df_res = df_res.round(2)
    row_data = df_res.to_dict('records')
    X_names = X_names + ["Cluster", "Cluster_Count"]  # Ensure this matches the DataFrame columns
    column_defs = [{"headerName": i, "field": i} for i in X_names]

    data_to_analyze["Province"] = list(data_to_analyze.index)
    data_to_analyze["Cluster"] = model.labels
    fig = px.choropleth(
        data_frame=data_to_analyze,
        geojson=geojson_data,
        locations='Province',
        featureidkey="properties.prov_name",  # Adjust this to match the GeoJSON file's property key
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Safe,  # Using a discrete colormap

    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update(layout_coloraxis_showscale=True)
    return row_data, column_defs, fig

if __name__ == '__main__':
    app.run_server(debug=True)
