# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 18:13:29 2024

@author: sindh
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Function to extract a DataFrame slice for a specific year and indicator
def slice(df, year):
    """
    Extracts a DataFrame slice containing 'Country Name' and the specified
    'year' column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - year (str): The target year.

    Returns:
    - pd.DataFrame: DataFrame slice with 'Country Name' and the specified
    'year' column.
    """
    df_slice = df[['Country Name', year]]
    return df_slice


# Function to merge two DataFrames on 'Country Name' using an outer join
def merge_df(x1, x2):
    """
    Merges two DataFrames on 'Country Name' using an outer join.

    Parameters:
    - x1 (pd.DataFrame): First DataFrame.
    - x2 (pd.DataFrame): Second DataFrame.

    Returns:
    - pd.DataFrame: Merged DataFrame.
    """
    merge = pd.merge(x1, x2, on='Country Name', how='outer')
    mer = merge.reset_index(drop=True)
    trans_data = mer.transpose()
    trans_data.columns = trans_data.iloc[0]
    trans_data = trans_data.iloc[1:]

    return mer, trans_data


# Function to generate elbow plot data for KMeans clustering
def elbow_plot(data1, data2, max_k=10):
    """
    Generates elbow plot data for KMeans clustering.

    Parameters:
    - data1 (pd.DataFrame): First dataset for clustering.
    - data2 (pd.DataFrame): Second dataset for clustering.
    - max_k (int): Maximum number of clusters to evaluate.

    Returns:
    - tuple: Distortion values for data1 and data2.
    """
    distortions1 = []
    distortions2 = []

    for k in range(1, max_k + 1):
        kmeans1 = KMeans(n_clusters=k, random_state=42)
        kmeans1.fit(data1)
        distortions1.append(kmeans1.inertia_)

        kmeans2 = KMeans(n_clusters=k, random_state=42)
        kmeans2.fit(data2)
        distortions2.append(kmeans2.inertia_)

    return distortions1, distortions2


# Function to perform clustering and generate elbow plots
def perform_clustering_elbow(df, indicator1, indicator2, year1, year2,
                             max_k=10):
    """
    Performs clustering and generates elbow plots for two indicators and two
    years.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - indicator1 (str): First indicator for clustering.
    - indicator2 (str): Second indicator for clustering.
    - year1 (str): First year for clustering.
    - year2 (str): Second year for clustering.
    - max_k (int): Maximum number of clusters to evaluate.
    """
    data1 = df[df['Indicator Name'] == indicator1].reset_index(drop=True)
    data2 = df[df['Indicator Name'] == indicator2].reset_index(drop=True)

    # Slicing and merging for the first year
    slice1 = slice(data1, year1).rename(
        columns={year1: f'{indicator1} {year1}'})
    slice2 = slice(data2, year1).rename(
        columns={year1: f'{indicator2} {year1}'})
    mer1, trans1 = merge_df(slice1, slice2)
    f1 = mer1.dropna(how='any').reset_index(drop=True)
    X = f1[[f'{indicator1} {year1}', f'{indicator2} {year1}']]

    # Slicing and merging for the second year
    slice3 = slice(data1, year2).rename(
        columns={year2: f'{indicator1} {year2}'})
    slice4 = slice(data2, year2).rename(
        columns={year2: f'{indicator2} {year2}'})
    mer2, trans2 = merge_df(slice3, slice4)
    f2 = mer2.dropna(how='any').reset_index(drop=True)
    Y = f2[[f'{indicator1} {year2}', f'{indicator2} {year2}']]

    # Generate elbow plots
    distortions_X, distortions_Y = elbow_plot(X, Y)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), distortions_X, marker='o', label=f'{year1}')
    plt.plot(range(1, max_k + 1), distortions_Y, marker='o', label=f'{year2}')
    plt.title('Elbow Plot')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.legend()
    plt.savefig('Elbow_Plot.png')

    # Cluster Plot for 2000
    create_cluster_plot(X, x_col=f'{indicator1} {year1}',
                        y_col=f'{indicator2} {year1}', n_clusters=5,
                        label=f'{year1}', savefig=True)

    # Cluster Plot for 2022
    create_cluster_plot(Y, x_col=f'{indicator1} {year2}',
                        y_col=f'{indicator2} {year2}', n_clusters=5,
                        label=f'{year2}', savefig=True)


# Function to create a cluster plot with KMeans clustering results
def create_cluster_plot(data, x_col, y_col, n_clusters=5, label='', ax=None, 
                        savefig=False):
    """
    Creates a cluster plot with KMeans clustering results.

    Parameters:
    - data (pd.DataFrame): Input DataFrame.
    - x_col (str): Name of the x-axis column.
    - y_col (str): Name of the y-axis column.
    - n_clusters (int): Number of clusters for KMeans.
    - label (str): Label for the plot.
    - ax (matplotlib.axes._axes.Axes): Axes object for plotting (optional).
    - savefig (bool): Whether to save the plot as an image.

    Returns:
    - str: Path to the saved image file (if savefig is True).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data['Cluster'] = kmeans.fit_predict(data[[x_col, y_col]])

    # If ax is not provided, create subplots
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plotting the cluster plot
    for cluster in range(n_clusters):
        cluster_data = data[data['Cluster'] == cluster]
        ax.scatter(cluster_data[x_col], cluster_data[y_col],
                   label=f'Cluster {cluster}')

    ax.set_title(f'Cluster Plot ({label})')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend()

    # Plotting the cluster centers
    centers = kmeans.cluster_centers_
    for i, center in enumerate(centers):
        ax.scatter(center[0], center[1], marker='+', s=200, c='black',
                   label=f'Cluster {i} Center')

    if savefig:
        save_path = f'Cluster_Plot_{label.replace(" ", "_")}.png'
        plt.savefig(save_path)
        plt.show()
        return save_path
    else:
        return None


# Read the data
df = pd.read_csv('API_19_DS2_en_csv_v2_5998250.csv', skiprows=3)

# Perform clustering and plot elbow plot
perform_clustering_elbow(df, 'CO2 emissions (kt)',
                         'Cereal yield (kg per hectare)', '2000', '2020')

# Read the data
df = pd.read_csv('API_19_DS2_en_csv_v2_5998250.csv', skiprows=3)

# Select three countries
selected_countries = ['India', 'Japan', 'United States']
indicator_name = 'CO2 emissions (kt)'

# Filter the data
data_selected = df[(df['Country Name'].isin(selected_countries)) & (
    df['Indicator Name'] == indicator_name)].reset_index(drop=True)

# Melt the DataFrame
data_forecast = data_selected.melt(id_vars=['Country Name', 'Indicator Name'],
                                   var_name='Year', value_name='Value')

# Filter out non-numeric values in the 'Year' column
data_forecast = data_forecast[data_forecast['Year'].str.isnumeric()]

# Convert 'Year' to integers
data_forecast['Year'] = data_forecast['Year'].astype(int)

# Handle NaN values by filling with the mean value
data_forecast['Value'].fillna(data_forecast['Value'].mean(), inplace=True)

# Prepare X and y for polynomial regression using data from 1990 to 2020
X = data_forecast[(data_forecast['Year'] >= 1990) & (
    data_forecast['Year'] <= 2020)][['Year']]
y = data_forecast[(data_forecast['Year'] >= 1990) & (
    data_forecast['Year'] <= 2020)]['Value']

# Create a dictionary to store predictions for each country
predictions = {}

# Fit polynomial regression model for each country
for country in selected_countries:
    country_data = data_forecast[data_forecast['Country Name'] == country]

    # Prepare data for the current country
    X_country = country_data[(country_data['Year'] >= 1990) & (
        country_data['Year'] <= 2020)][['Year']]
    y_country = country_data[(country_data['Year'] >= 1990) & (
        country_data['Year'] <= 2020)]['Value']

    # Fit polynomial regression model with degree 3
    degree = 3
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X_country)

    model = LinearRegression()
    model.fit(X_poly, y_country)

    # Predict values for the years between 1990 and 2025
    all_years = list(range(1990, 2026))
    X_pred = poly_features.transform(pd.DataFrame(all_years, columns=['Year']))
    forecast_values = model.predict(X_pred)

    # Store the predictions for the current country
    predictions[country] = {'values': forecast_values}

save_plots = True

# Plotting the results with error range in separate plots for each country
for i, country in enumerate(selected_countries):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot actual data
    country_data_actual = data_forecast[(
        data_forecast['Country Name'] == country) & (
            data_forecast['Year'] >= 1990) & (data_forecast['Year'] <= 2020)]
    ax.plot(country_data_actual['Year'], country_data_actual['Value'],
            label=f'{country} (Actual)', linestyle='solid', marker='o',
            color='Purple')

    # Plot fitting curve up to the predicted value (year 2025)
    ax.plot(all_years, predictions[country]['values'],
            label=f'{country} (Fitting Curve)', linestyle='solid',color='red')

    # Highlight the predicted value for 2025
    ax.scatter(2025, predictions[country]['values'][-1], marker='o',
               color='red')
    ax.text(2025, predictions[country]['values'][-1],
            f'{country}: {predictions[country]["values"][-1]:.2f}',
            fontsize=8, ha='right')

    # Calculate and plot the error range (within 1 standard deviation)
    error = country_data_actual['Value'] - predictions[
        country]['values'][:len(country_data_actual)]
    std_dev = np.std(error)
    upper_bound = predictions[country]['values'] + std_dev
    lower_bound = predictions[country]['values'] - std_dev

    ax.fill_between(all_years, lower_bound, upper_bound, color='purple',
                    alpha=0.2, label=f'{country} (Error Range)')

    ax.set_title(
        f'CO2 Emissions of {country} Actual and Fitting Curve(1990-2020)')
    ax.set_xlabel('Year')
    ax.set_ylabel('CO2 Emissions (kt)')
    ax.legend()
    ax.grid(True)

    if save_plots:
        save_path = f'Forecast_Plot_{country.replace(" ", "_")}.png'
        plt.savefig(save_path)
        print(f"Forecast plot saved as {save_path}")
        plt.tight_layout()
        plt.show()
    else:
        plt.show()
