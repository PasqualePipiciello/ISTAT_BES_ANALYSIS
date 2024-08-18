import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, callback, dash_table
import pandas as pd
import dash_ag_grid as dag



def create_regression_section(domains, years):
    regression_section = dbc.Card(
        dbc.Container(
            [
                # Row for selecting the year
                dbc.Row(
                    dbc.Col(
                        [
                            dbc.Label("Select Year"),
                            dcc.Dropdown(
                                id="select-year",
                                options=[{"label": year, "value": year} for year in years],
                                value=years[0]
                            )
                        ]
                    )
                ),
                # Row for selecting X domain and X indicator variables
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Select X domain variables"),
                                dcc.Dropdown(
                                    id="x-variable_domain",
                                    options=[{"label": col, "value": col} for col in domains],
                                    multi=True  # Allow multiple selections
                                )
                            ]
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Select X indicator variables"),
                                dcc.Dropdown(
                                    id="x-variable-indicator",
                                    multi=True  # Keep multi-selection for indicators
                                )
                            ]
                        )
                    ]
                ),
                # Row for selecting Y domain and Y indicator variables
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Choose Y variable domain"),
                                dcc.Dropdown(
                                    id="y-variable_domain",
                                    options=[{"label": col, "value": col} for col in domains],
                                )
                            ]
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Choose Y indicator"),
                                dcc.Dropdown(
                                    id="y-indicator"
                                )
                            ]
                        )
                    ]
                ),
                # Row for inputting the cluster count
                dbc.Row(
                    dbc.Col(
                        [
                            dbc.Label("Cluster count"),
                            dbc.Input(id="cluster-count", type="number", value=3)
                        ]
                    )
                ),
                # Row for the Confirm button
                dbc.Row(
                    dbc.Col(
                        dbc.Button("Confirm", id="confirm-button", color="primary"),
                        className="d-grid gap-2"  # This ensures the button takes the full width of the column
                    )
                ),
                # Row for the AgGrid and Graph
                dbc.Row(
                    [
                        dbc.Col(
                            dag.AgGrid(
                                id='df_res',
                                className='ag-theme-alpine',
                                style={'height': '400px', 'width': '100%'}
                            ),
                            width=6
                        ),
                        dbc.Col(
                            dcc.Graph(id='regression-graph'),
                            width=6
                        )
                    ]
                )
            ],
            fluid=True
        ),
        body=True,
    )
    return regression_section