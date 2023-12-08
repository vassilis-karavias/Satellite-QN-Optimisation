# Satellite-QN-Optimisation

# Requirements
Numpy 1.20.1+  
pandas 1.5.2+  
matplotlib 3.6.2+  
cplex V20.1  
networkx 2.8.8+  


# How to use

## Simple Model

To use the simple model, you first need to import the graph problem from the required .csv files using   
*graphs, key_dict, sat_keys = satellite_utils.import_simple_problem(rate_core_satellite_file_path, key_dict_file_path)*  
*rate_core_satellite_file_path* is a .csv file with fields [ID, ground_station, satellite_path, capacity]. ID is the current graph ID for the row. Each row represents a (ground_station, satellite_path) link with the number of keys per day generated given by capacity. *key_dict_file_path* is a .csv file with fields [ID, ground_core_station_1, ground_core_station_2, Tij]. ID is the current graph ID for the row. Each row represents the transmission requirement Tij between two ground stations (ground_core_station_1, ground_core_station_2) in bits/s.  
The import will give a dictionary of *graphs* of the form *graphs = {ID: networkx.Graph()}*, a dictionary of dictionary of key requirements between ground stations in the graph: *key_dict = {ID: {(ground_core_station_1,ground_core_station_2): Tij}}*, and a dictionary with an array of satellite keys of the form: *sat_keys = {ID: [possible_sat_keys]}*. The required property for a given graph with ID=i can be extracted as *property[i]*.  

To run the simple metric model, first create a class:  
*sat_m_opt = satellite_simple_metric.Satellite_Metric_Optimisation(prob, sat_keys, g, key_dict)*  
*prob = cplex.Cplex()* is the cplex class, while *sat_keys*, *g* and *key_dict* are the parameters for a given graph id: *sat_keys = sat_keys[ID]*, *g = graphs[ID]*, *key_dict = key_dict[ID]*. To run the optimisation use the method:  
*sol_dict, prob = sat_m_opt.optimise_satellite_run(time_limit)*  
Which return the solution dictionary of *{variable_name: value}* and the cplex problem class. The objective value can be obtained by  
*prob.solution.get_objective_value()*

## Full Optimisation

To use the full optimisation model, you first need to import the graph problem from the required .csv files using  
*graphs, key_dict, sat_keys, core_sites, ground_stations, Tij = satellite_utils.import_full_problem(core_sites_file_path, key_dict_file_path, satellite_rates_file_path, Tij_file_path)*  
*core_sites_file_path* is a .csv file with fields [ID, core_site, ground_station_allowed_0, ground_station_allowed_1,..., ground_station_allowed_n]. ID is the current graph ID for the row. Each row represents the allowed ground stations (ground_station_allowed_i) for the core_site. *key_dict_file_path* is a .csv file with fields [ID, ground_core_station_1, ground_core_station_2]. ID is the current graph ID for the row. Each row contains one possible pair of ground stations in the graph, this will be the possible commodities in the problem. *satellite_rates_file_path* is a .csv file with fields [ID, ground_core_station, satellite_path, capacity]. ID is the current graph ID for the row. Each row represents the capacity between a ground station (ground_core_station) and a satellite path. capacity is the number of key bits generated per day. *Tij_file_path* is a .csv file with fields [ID, core_site_1, core_site_2, Tij]. ID is the current graph ID for the row. Each row represents the transmission requirements, Tij in bits/s, for a pair of core metropolitan sites (core_site_1, core_site_2).  
The import will give a dictionary of graphs, key_dict of the form *key_dict = {ID: {(ground_core_station_1,ground_core_station_2): 1}}*, sat_keys. Further, *core_sites* which is a dictionary that contains information of which ground stations are allowed to connect to the core site *core_sites = {ID: {core_site = [available_ground_stations]}}*. *ground_stations* is a dict containing a list of ground stations for each graph ID: *ground_stations = {ID: [ground_stations]}*. Finally, *Tij* is a dict containing information on the required transmission requirement rate between core sites *Tij =  {ID: {(core_site_1, core_site_2) : Tij}}*.  


To run the model, first create a class:  
*sat_opt = satellite_full_optimisation_problem.Satellite_Full_Network_Problems(prob = cplex.Cplex(), sat_keys = sat_keys[ID], g = graphs[ID], key_dict = key_dict[ID], Tij = Tij[ID], ground_station_keys = ground_stations[ID], core_site_keys = core_sites[ID])*  
To run the optimisation without core network cost use the method:  
*sol_dict, prob = sat_opt.optimisation_satellite_run(M, time_limit)*  
*M* is a large number. This should be selected appropriately for the given problem. To select the best *M* for the problem use:  
*max_sum_t, fractional_error = satellite_full_optimisation_problem.select_appropriate_M(Tij_list, core_sites, epsilon_tol = 1e-10)*  
*max_sum_t* is the best value to use for *M* and *fractional_error* is a bound on up to how much the model will underestimate the required transmission between any two core sites with the given *M*. *Tij_list = Tij[ID]*, *core_sites = core_sites[ID]* and *epsilon_tol* is the tolerance set on the optimisation. If this is not changed, it is set to 1e-10 in the current implementation.   

To run the optimisation including core network cost use the method:  
*sol_dict, prob = sat_opt.optimisation_satellite_and_core_network_run(C_ij_core, C_is_core, M, time_limit)*  
*C_ij_core* is a dict of cost of connecting two core sites *C_ij_core = {(core_site_1, core_site_2): cost}* while *C_is_core* is a dict of cost of connecting one core site to a ground station *C_is_core = {(core_site, ground_station): cost}*. These can be defined by the user. The value can also be generated using a uniform cost per km using:  
*C_ij_core = satellite_full_optimisation_problem.calculate_C_ij_core(core_sites = core_sites[ID], cost_factor_per_km)*  
*C_is_core = satellite_full_optimisation_problem.calculate_C_is_core(core_sites = core_sites[ID], ground_station_list = ground_stations[ID], cost_factor_per_km)*  
*cost_factor_per_km* is the uniform cost per km cost of the core network. If these methods are used, *satellite_full_optimisation_problem.ground_station_potential_sites* and *satellite_full_optimisation_problem.sites_lat_lon* need to be changed to have the correct sites and latitude,longitude pairs for the given problem. Two examples are provided in the python file corresponding to the graphs in the paper.







