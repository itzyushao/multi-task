import pandas as pd
import numpy as np 
from sklearn.neighbors import DistanceMetric
import networkx as nx
import torch 
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data

def get_geometric_data():
    cities_df = load_raw_cities_df()
    cities_df = convert_lat_lon_to_radian(cities_df)
    dist_matrix_df = get_city_dist_matrix(cities_df)
    G = convert_to_networkx_graph(dist_matrix_df, threshold = 1200)
    geometric_data = convert_to_geometric_data(cities_df[['x1','x2']], cities_df[['y']], G)
    return geometric_data


def load_raw_cities_df():
    cities_df = pd.DataFrame({
        'city':['bangalore','Mumbai','Delhi','kolkatta','chennai','bhopal'],
        'lat':[12.9716,19.076,28.7041,22.5726,13.0827,23.2599],
        'lon':[77.5946,72.877,77.1025,88.639,80.2707,77.4126],
        'x1':[20,35,24,33,35,18],
        'x2':[5,5,7,13,16,21],
        'y':[1200,1500,2000,1780,1450,3000]})
    return cities_df 

def convert_lat_lon_to_radian(cities_df):
    cities_df['lat'] = np.radians(cities_df['lat'])
    cities_df['lon'] = np.radians(cities_df['lon'])
    return cities_df
def get_city_dist_matrix(cities_df):
    dist = DistanceMetric.get_metric('haversine')
    cities_df[['lat','lon']].to_numpy()
    dist_matrix_df = pd.DataFrame(dist.pairwise(cities_df[['lat','lon']].to_numpy())*6373,  columns=cities_df.city.unique(), index=cities_df.city.unique())
    return dist_matrix_df

def convert_to_networkx_graph(dist_matrix_df, threshold = 1200):
    D = dist_matrix_df.values
    D[D < threshold] = 1
    D[D >= threshold] = 0
    G = nx.from_numpy_matrix(D)
    return G

def convert_to_geometric_data(x_df, y_df, G):
    x = torch.tensor(x_df.values, dtype=torch.float)
    y = torch.tensor(y_df.values, dtype=torch.float)
    edge_index = from_networkx(G).edge_index
    data = Data(x=x, y=y, edge_index=edge_index)
    return data