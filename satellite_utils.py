import numpy as np
import pandas as pd
import networkx as nx
from math import nan


def import_simple_problem(rate_core_satellite_file_path, key_dict_file_path):
    key_dict_df = pd.read_csv(key_dict_file_path)
    rate_satellite_graph_structure_df = pd.read_csv(rate_core_satellite_file_path)
    rate_satellite_graph_structure_df["capacity"] = rate_satellite_graph_structure_df.apply(lambda x:  x["capacity"] / (24 * 60 * 60), axis=1)
    possible_ids = key_dict_df["ID"].unique()
    graphs = {}
    key_dict = {}
    sat_keys = {}
    for id in possible_ids:
        rate_satellite_graph_structure_current = rate_satellite_graph_structure_df[rate_satellite_graph_structure_df["ID"] == id].drop(["ID"], axis=1)
        graph = nx.from_pandas_edgelist(rate_satellite_graph_structure_current, "ground_core_station", "satellite_path", ["capacity"])
        graph = graph.to_undirected()
        graph = graph.to_directed()
        graphs[id] = graph
        key_dict_current = key_dict_df[key_dict_df["ID"] == id].drop(["ID"], axis =1)
        key_dict_current_dict = {}
        for index, row in key_dict_current.iterrows():
            key_dict_current_dict[(row["ground_core_station_1"],row["ground_core_station_2"])] = row["Tij"]
        key_dict[id] = key_dict_current_dict
        possible_sats = rate_satellite_graph_structure_current["satellite_path"].unique()
        sat_keys[id] = possible_sats
    return graphs, key_dict, sat_keys



def import_full_problem(core_sites_file_path, key_dict_file_path, satellite_rates_file_path, Tij_file_path):
    core_sites_df = pd.read_csv(core_sites_file_path)
    key_dict_df = pd.read_csv(key_dict_file_path)
    satellite_rates_df =  pd.read_csv(satellite_rates_file_path)
    Tij_df = pd.read_csv(Tij_file_path)
    satellite_rates_df["capacity"] = satellite_rates_df.apply( lambda x: x["capacity"] / (24 * 60 * 60), axis=1)
    possible_ids = key_dict_df["ID"].unique()
    graphs = {}
    key_dict = {}
    sat_keys = {}
    core_sites = {}
    ground_stations = {}
    Tij = {}
    for id in possible_ids:
        rate_satellite_graph_structure_current = satellite_rates_df[ satellite_rates_df["ID"] == id].drop(["ID"], axis=1)
        graph = nx.from_pandas_edgelist(rate_satellite_graph_structure_current, "ground_core_station", "satellite_path",
                                        ["capacity"])
        graph = graph.to_undirected()
        graph = graph.to_directed()
        graphs[id] = graph
        key_dict_current = key_dict_df[key_dict_df["ID"] == id].drop(["ID"], axis=1)
        key_dict_current_dict = {}
        for index, row in key_dict_current.iterrows():
            key_dict_current_dict[(row["ground_core_station_1"], row["ground_core_station_2"])] = 1
        key_dict[id] = key_dict_current_dict
        possible_sats = rate_satellite_graph_structure_current["satellite_path"].unique()
        sat_keys[id] = possible_sats
        possible_gs = rate_satellite_graph_structure_current["ground_core_station"].unique()
        ground_stations[id] = possible_gs
        Tij_current_dataframe = Tij_df[Tij_df["ID"] == id].drop(["ID"], axis =1)
        Tij_current_dict = {}
        for index, row in Tij_current_dataframe.iterrows():
            Tij_current_dict[(row["core_site_1"], row["core_site_2"])] = row["Tij"]
        Tij[id] = Tij_current_dict
        core_sites_current_dataframe = core_sites_df[core_sites_df["ID"] == id].drop(["ID"], axis = 1)
        core_sites_dict = {}
        column_names = list(core_sites_current_dataframe.columns.values)
        column_names.remove("core_site")
        for index, row in core_sites_current_dataframe.iterrows():
            available_ground_stations = []
            for name in column_names:
                null_series = core_sites_current_dataframe[name].isnull()
                if not null_series[index]:
                    available_ground_stations.append(row[name])
            core_sites_dict[row["core_site"]] = available_ground_stations
        core_sites[id] = core_sites_dict
    return graphs, key_dict, sat_keys, core_sites, ground_stations, Tij


