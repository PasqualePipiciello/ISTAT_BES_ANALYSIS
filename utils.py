import pandas as pd
import numpy as np
import os
import dash_ag_grid as dag



def columns_without_nan(data):
    """
    Function to get the names of the columns with NaN values.

    Input:
    - data: DataFrame with rows = observations and columns = variables.

    Returns:
    - res: A Series with the count of NaN values for each column that contains NaNs.
    """
    # Identify columns with NaN values
    without_nan = data.isna().any()

    # Get the count of NaN values for each column that contains NaNs
    res = data.loc[:, without_nan].isna().sum()

    return res

def replace_nan(preprocessed_data):
    nan_check = columns_without_nan(preprocessed_data)
    to_remove = []
    for i in nan_check.index:
        if nan_check[i] > 10:
            to_remove.append(i)

    preprocessed_data = preprocessed_data.drop(to_remove, axis=1)
    preprocessed_data_imputed = preprocessed_data.fillna(preprocessed_data.median())
    return preprocessed_data_imputed


def inertia_vs_k(dati, k=np.arange(2, 9)):
    """
    Function to compute and plot within-cluster sum of squares (WSS) for different values of k.

    Parameters:
    - dati : DataFrame of BES preprocessed data.
    - k : array-like, vector of k values to compute corresponding inertia (default = range(2, 9)).

    Returns:
    - WSSS : list of within-cluster sum of squares for each k.
    """
    WSSS_s = []
    WSSS = []

    # Standardize the data
    scaler = StandardScaler()
    dati_scaled = scaler.fit_transform(dati)

    for k_ in k:
        kmeans = KMeans(n_clusters=k_, random_state=0)
        kmeans.fit(dati_scaled)

        # Inertia: Sum of squared distances of samples to their closest cluster center
        wcss = kmeans.inertia_

        WSSS.append(wcss)

    return WSSS


def create_domain_indicator_mapping(dati):
    """
    Function to create a mapping of domains to their respective indicators.

    Parameters:
    - dati : DataFrame containing BES data with at least the columns "DOMINIO" and "INDICATORE".

    Returns:
    - dom_ind : Dictionary where keys are domains and values are lists of unique indicators for each domain.
    """
    dom_ind = {}

    # Get the unique domains
    domini = dati['DOMAIN'].unique()

    # Iterate over each domain to find the corresponding indicators
    for dominio in domini:
        indicators = dati.loc[dati['DOMAIN'] == dominio, 'INDICATOR'].unique().tolist()
        dom_ind[dominio] = indicators

    return dom_ind

def create_ag_grid(id):
    grid = dag.AgGrid(
        id=id,
        className = "ag-theme-alpine-auto-dark"
    )
    return grid