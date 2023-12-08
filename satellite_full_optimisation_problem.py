import cplex
import time
from satellite_utils import import_full_problem
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import copy


# sites_lat_lon = { "Athens": [37.9838, 23.7275], "Thessaloniki":[40.6401, 22.9444],"London": [51.5072, -0.1276],
#                   "Birmingham": [52.48652, -1.8904], "Manchester": [53.4808,-2.2426], "Leeds": [53.8008, -1.5491],
#                   "Newcastle": [54.9783, -1.6178], "Portsmouth": [50.8198, -1.0880], "Paris": [48.8566,2.3522],
#                   "Lyon": [45.7640,4.8357], "Lille": [50.6292, 3.0573], "Marseille": [43.2965,5.3698],
#                   "Nantes": [47.2184, -1.5538], "Frankfurt": [50.1109,8.6921], "Berlin": [52.5200, 13.4050],
#                   "Munich": [48.1351, 11.5820], "Dusseldorf": [51.2277, 6.7735], "Nurenburg": [49.4521,11.0767],
#                   "Leipzig": [51.3397,12.3731], "Stuttgart": [48.7758,9.1829], "Amsterdam": [52.3676,4.9041],
#                   "Madrid": [40.4168, -3.7038], "Barcelona": [41.3874,2.1686],"Malaga": [36.7213, -4.4213],
#                   "Valencia": [39.4699, -0.3763], "Porto": [41.1579,-8.6291],"Lisbon": [38.7223, -9.1393],
#                   "Zurich": [47.3769,8.5417], "Geneva": [46.2044, 6.1432], "Bern": [46.9480, 7.4474],
#                   "Lucerne": [47.0502, 8.3093], "Milan": [45.4642,9.19000], "Florence": [43.7696, 11.2558],
#                   "Venice": [45.4408, 12.3155], "Rome": [41.9028, 12.4964], "Brussels": [50.8476,4.3572],
#                   "Luxembourg": [49.8153, 6.1296], "Stockholm": [59.3293, 18.0686], "Copenhagen": [55.6761, 12.5683],
#                   "Oslo": [59.9139, 10.7522], "Sofia": [42.6977, 23.3219], "Bucharest": [44.4268, 26.1025]}


# ground_station_potential_sites = {"London": [51.5072, -0.1276], "Madrid": [40.4168, -3.7038], "Edinburgh": [55.9533, -3.188267],
#                  "Barcelona": [41.3874,2.1686], "Paris": [48.8566,2.3522], "Nantes": [47.2184, -1.5538],
#                  "Lyon": [45.7640,4.8357], "Lisbon": [38.7223, -9.1393], "Bern": [46.9480, 7.4474],
#                  "Freiburg": [47.9990,7.8421], "Milan": [45.4642,9.19000], "Florence": [43.7696, 11.2558],
#                  "Naples": [40.8518, 14.2681], "Frankfurt": [50.1109,8.6921], "Dortmund": [51.5136,7.4653],
#                  "Berlin": [52.5200, 13.4050], "Athens": [37.9838, 23.7275], "Ioannina": [39.6650,20.8537]}


ground_station_potential_sites = {"London": [51.5072, -0.1276], "Frankfurt": [50.1109, 8.6821], "Graz": [47.0707, 15.4395], "Johanessburg": [-26.2041, 28.0473],
                       "Sao Paulo": [-23.5558, -46.6496], "Tokyo": [35.653, 139.839], "Auckland": [-36.8509, 174.7645],
                       "New Delhi": [28.6139, 77.2090], "Mumbai": [19.0760, 72.8777], "Bangalore": [12.9716, 77.5946],
                       "Xinglong": [40.395833, 117.5775], "Nanshan": [43.475278, 87.176667], "Perth": [-31.9523, 115.8613],
                       "Brisbane": [-27.4705, 153.0260], "Melborne": [-37.8136, 144.9631], "Albany": [42.6526, -73.7562],
                       "Denver": [39.7392, -104.9903], "Nashville": [36.1627, -86.7816]}


sites_lat_lon = {"London": [51.5072, -0.1276], "Manchester": [53.4808,-2.2426], "Amsterdam": [52.3676,4.9041],
                 "Stockholm": [59.3293, 18.0686], "Zurich": [47.3769,8.5417], "Frankfurt": [50.1109,8.6921],
                 "Berlin": [52.5200, 13.4050], "Paris": [48.8566,2.3522], "Madrid": [40.4168, -3.7038],
                  "Milan": [45.4642,9.19000], "Johanessburg": [-26.2041, 28.0473], "Sao Paulo": [-23.5558, -46.6496],
                    "Buenos Aires": [-34.6037, -58.3816], "Tokyo": [35.653, 139.839], "Auckland": [-36.8509, 174.7645],
                  "New Delhi": [28.6139, 77.2090], "Mumbai": [19.0760, 72.8777], "Bangalore": [12.9716, 77.5946],
                   "Beijing":  [39.9042, 116.4074], "Shanghai": [31.2304, 121.4737], "Hong Kong": [22.3193, 114.1694], "Perth": [-31.9523, 115.8613],
                 "Adelade": [-34.9285, 138.6007], "Melborne": [-37.8136, 144.9631], "Sydney": [-33.8688, 151.2093],
                    "Brisbane": [-27.4705, 153.0260], "LA": [34.055, -118.24], "Seattle": [47.6061, -122.3328], "Dallas": [32.7767, -96.7970],
                 "Chicago": [41.8781, 87.6298], "Washington": [38.9072, -77.0369], "Jacksonville": [30.3322, -81.6557]}

def great_circle_distance_calculation(lat_1, lat_2, lon_1, lon_2):
    # Use Haversine formula to calculate the distance between the sites
    hav_lat = np.power(np.sin((lat_2 - lat_1) / 2), 2)
    hav_lon = np.power(np.sin((lon_2 - lon_1) / 2), 2)
    hav_dist = hav_lat + np.cos(lat_1) * np.cos(lat_2) * hav_lon
    if abs(hav_dist) <= 1:
        return 2 * 6371000 * np.arcsin(np.sqrt(hav_dist))
    else:
        return 2 * 6371000 * np.arcsin(1)

def split_sol_to_flow(sol_dict):
    """
    Split the solution dictionary into 2 dictionaries containing the flow variables only and the binary variables only
    Parameters
    ----------
    sol_dict : The solution dictionary containing solutions to the primary flow problem

    Returns
    -------

    """
    binary_on_off_dict = {}
    alpha_dict = {}
    satellite_dict = {}
    Tij_key = {}
    Tinj_key = {}
    M_key = {}
    for key in sol_dict:
        # get all keys that are flow and add to dictionary
        if key[:2] == "x_":
            binary_on_off_dict[key] = sol_dict[key]
        elif key[0] == "N":
            # get the keys that are binary 'on' 'off' and add to dictionary
            satellite_dict[key] = sol_dict[key]
        elif key[0] == "a":
            # get the keys that represent the lambda_{i,j} - representing number of detectors and add to dictionary
            alpha_dict[key] = sol_dict[key]
        elif key[0] == "T":
            Tij_key[key] = sol_dict[key]
        elif key[0] == "t":
            Tinj_key[key] = sol_dict[key]
        elif key[0] == "M":
            M_key[key] = sol_dict[key]
    return binary_on_off_dict, satellite_dict, alpha_dict, Tij_key, Tinj_key, M_key


def create_sol_dict(prob):
    """
    Create a dictionary with the solution of the parameters
    """
    names=prob.variables.get_names()
    values=prob.solution.get_values()
    sol_dict={names[idx]: (values[idx]) for idx in range(prob.variables.get_num())}
    return sol_dict

class Satellite_Full_Network_Problems:

    def __init__(self, prob, sat_keys, g, key_dict, Tij, ground_station_keys, core_site_keys):
        self.prob = prob
        # [satellite_RAAN]
        self.sat_keys = sat_keys
        # {core_site: [allowed ground stations]}
        self.core_site_keys = core_site_keys
        # [ground_stations]
        self.ground_station_keys = ground_station_keys
        # graph of the underlaying effective network
        self.g = g
        # {(ground_station_1, ground_station_2): 1}
        self.key_dict = key_dict
        # {(core_site_1, core_stite_2): Tij}
        self.Tij  = Tij

    def add_max_capacity_contraint(self):
        variable_names = [f'x{i}_{j}_k{k[0]}_{k[1]}' for k in self.key_dict for i, j in list(self.g.edges)]
        self.prob.variables.add(names=variable_names, types=[self.prob.variables.type.continuous] * len(variable_names))
        N_sat_var = []
        for sat in self.sat_keys:
            N_sat_var.append(f"N_sat_{sat}")
        self.prob.variables.add(names=N_sat_var, types=[self.prob.variables.type.integer] * len(N_sat_var))
        for i,j in list(self.g.edges):
            if j in self.sat_keys:
                R_ij = int(self.g.edges[[i, j]]["capacity"])
                if R_ij > 0:
                    ind = []
                    val = []
                    for k in self.key_dict:
                        ind.extend([f"x{i}_{j}_k{k[0]}_{k[1]}", f"x{j}_{i}_k{k[0]}_{k[1]}"])
                        val.extend([1/R_ij, 1/R_ij])
                    ind.extend([f"N_sat_{j}"])
                    val.extend([-1])
                    contraint = [cplex.SparsePair(ind=ind, val=val)]
                    self.prob.linear_constraints.add(lin_expr=contraint, senses=["L"], rhs=[0.0])

                else:
                    ind = []
                    val = []
                    for k in self.key_dict:
                        ind.extend([f"x{i}_{j}_k{k[0]}_{k[1]}", f"x{j}_{i}_k{k[0]}_{k[1]}"])
                        val.extend([1, 1])
                    contraint = [cplex.SparsePair(ind=ind, val=val)]
                    self.prob.linear_constraints.add(lin_expr=contraint, senses=["L"], rhs=[0.0])

    def add_required_flow_rate_constraint(self):
        Tnm_var = []
        for k in self.key_dict:
            Tnm_var.append(f"T_{k[0]}_{k[1]}")
        self.prob.variables.add(names=Tnm_var, types=[self.prob.variables.type.continuous] * len(Tnm_var))
        for k in self.key_dict:
            ind = []
            val = []
            for l in self.g.adj[k[1]]:
                ind.extend([f"x{l}_{k[1]}_k{k[0]}_{k[1]}"])
                val.extend([1])
            ind.extend([f"T_{k[0]}_{k[1]}"])
            val.extend([-1])
            contraint = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=contraint, senses=["G"], rhs=[0.0])

    def add_conservation_of_flow_constraint(self):
        for i in self.g.nodes:
            for k in self.key_dict:
                if i != k[0] and i != k[1]:
                    ind = []
                    val = []
                    for j in self.g.adj[i]:
                        ind.extend([f"x{i}_{j}_k{k[0]}_{k[1]}", f"x{j}_{i}_k{k[0]}_{k[1]}"])
                        val.extend([1,-1])
                    contraint = [cplex.SparsePair(ind=ind, val=val)]
                    self.prob.linear_constraints.add(lin_expr=contraint, senses=["E"], rhs=[0.0])

    def ensure_only_one_ground_station_per_core_site_constraint(self):
        binary_core_site_var = []
        for l in self.core_site_keys:
            for a in self.ground_station_keys:
                binary_core_site_var.append(f"x_{l}_{a}")
        self.prob.variables.add(names=binary_core_site_var, types=[self.prob.variables.type.binary] * len(binary_core_site_var))
        for l in self.core_site_keys:
            ind = []
            val = []
            for a in self.ground_station_keys:
                ind.extend([f"x_{l}_{a}"])
                val.extend([1])
            contraint = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=contraint, senses=["E"], rhs=[1])

    def ensure_only_allowed_ground_stations_can_be_on_constraint(self):
        for l in self.core_site_keys:
            for a in self.ground_station_keys:
                if a in self.core_site_keys[l]:
                    ind = [f"x_{l}_{a}"]
                    val = [1]
                    contraint = [cplex.SparsePair(ind=ind, val=val)]
                    self.prob.linear_constraints.add(lin_expr=contraint, senses=["L"], rhs=[1])
                else:
                    ind = [f"x_{l}_{a}"]
                    val = [1]
                    contraint = [cplex.SparsePair(ind=ind, val=val)]
                    self.prob.linear_constraints.add(lin_expr=contraint, senses=["L"], rhs=[0])


    def calculate_total_transmission_requirement_constraint(self):
        Tnim_var = []
        for k in self.key_dict:
            for i in self.core_site_keys:
                Tnim_var.append(f"t_{k[0]}_{i}_{k[1]}")
        self.prob.variables.add(names=Tnim_var, types=[self.prob.variables.type.continuous] * len(Tnim_var))
        for k in self.key_dict:
            ind = []
            val = []
            for i in self.core_site_keys:
                ind.extend([f"t_{k[0]}_{i}_{k[1]}"])
                val.extend([1])
            ind.extend([f"T_{k[0]}_{k[1]}"])
            val.extend([-1])
            contraint = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=contraint, senses=["E"], rhs=[0])

    def get_Tnim_constraint(self, M):
        Min_var = []
        for i in self.core_site_keys:
            for n in self.ground_station_keys:
                Min_var.append(f"M_{i}_{n}")
        self.prob.variables.add(names=Min_var, types=[self.prob.variables.type.continuous] * len(Min_var))
        for k in self.key_dict:
            for l in self.core_site_keys:
                ind = [f"t_{k[0]}_{l}_{k[1]}", f"M_{l}_{k[0]}"]
                val = [1,1]
                for j in self.core_site_keys:
                    if l != j:
                        ind.extend([f"x_{j}_{k[1]}"])
                        val.extend([-self.Tij[l,j] - self.Tij[j,l]])
                contraint = [cplex.SparsePair(ind=ind, val=val)]
                self.prob.linear_constraints.add(lin_expr=contraint, senses=["G"], rhs=[0])
        for i in self.core_site_keys:
            for n in self.ground_station_keys:
                ind = [f"M_{i}_{n}", f"x_{i}_{n}"]
                val = [1, +M]
                contraint = [cplex.SparsePair(ind=ind, val=val)]
                self.prob.linear_constraints.add(lin_expr=contraint, senses=["L"], rhs=[M])


    def flow_out_sink_in_source_zero(self):
        for k in self.key_dict:
            for j in self.g.adj[k[1]]:
                ind = [f"x{k[1]}_{j}_k{k[0]}_{k[1]}"]
                val = [1]
                contraint = [cplex.SparsePair(ind=ind, val=val)]
                self.prob.linear_constraints.add(lin_expr=contraint, senses=["E"], rhs=[0.0])
        for k in self.key_dict:
            for i in self.g.adj[k[0]]:
                ind = [f"x{i}_{k[0]}_k{k[0]}_{k[1]}"]
                val = [1]
                contraint = [cplex.SparsePair(ind=ind, val=val)]
                self.prob.linear_constraints.add(lin_expr=contraint, senses=["E"], rhs=[0.0])

    def objective_minimise_number_satellites(self):
        obj_vals = []
        for sat in self.sat_keys:
            obj_vals.append((f"N_sat_{sat}", 1))
        self.prob.objective.set_linear(obj_vals)
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)



    def objective_minimise_cost(self, C_ij_core, C_is_core):
        alphaijs_var = []
        for i in self.core_site_keys:
            for j in self.core_site_keys:
                if i < j:
                    for n in self.ground_station_keys:
                        alphaijs_var.append(f"alpha_{i}_{j}_{n}")
        self.prob.variables.add(names=alphaijs_var, types=[self.prob.variables.type.binary] * len(alphaijs_var))
        for i in self.core_site_keys:
            for j in self.core_site_keys:
                if i < j:
                    for n in self.ground_station_keys:
                        ind = [f"alpha_{i}_{j}_{n}", f"x_{i}_{n}", f"x_{j}_{n}"]
                        val = [1, -1, -1]
                        contraint = [cplex.SparsePair(ind=ind, val=val)]
                        self.prob.linear_constraints.add(lin_expr=contraint, senses=["G"], rhs=[-1])

        obj_vals = []
        for sat in self.sat_keys:
            obj_vals.append((f"N_sat_{sat}", 1))
        for i in self.core_site_keys:
            for j in self.core_site_keys:
                if i < j:
                    for n in self.ground_station_keys:
                        obj_vals.append((f"alpha_{i}_{j}_{n}",C_ij_core[i,j]))
        for i in self.core_site_keys:
            for s in self.ground_station_keys:
                obj_vals.append((f"x_{i}_{s}", C_is_core[i,s]))
        self.prob.objective.set_linear(obj_vals)
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

    def optimisation_satellite_run(self, M = 5e6,  time_limit=1e5):
        """
               set up and solve the problem for minimising the overall cost of the network
               """
        t_0 = time.time()
        print("Start Optimisation")
        self.add_max_capacity_contraint()
        self.add_required_flow_rate_constraint()
        self.add_conservation_of_flow_constraint()
        self.ensure_only_one_ground_station_per_core_site_constraint()
        self.ensure_only_allowed_ground_stations_can_be_on_constraint()
        self.calculate_total_transmission_requirement_constraint()
        self.flow_out_sink_in_source_zero()
        self.get_Tnim_constraint(M)
        self.objective_minimise_number_satellites()
        self.prob.write("test_1.lp")
        self.prob.parameters.lpmethod.set(4)
        self.prob.parameters.mip.limits.cutpasses.set(1)
        self.prob.parameters.mip.strategy.probe.set(-1)
        self.prob.parameters.mip.strategy.variableselect.set(4)
        self.prob.parameters.mip.strategy.kappastats.set(1)
        self.prob.parameters.mip.tolerances.integrality.set(float(1e-10))
        self.prob.parameters.mip.tolerances.mipgap.set(float(0.01))
        # prob.parameters.simplex.limits.iterations = 50
        print(self.prob.parameters.get_changed())
        self.prob.parameters.timelimit.set(time_limit)
        t_1 = time.time()
        print("Time to set up problem: " + str(t_1 - t_0))
        self.prob.solve()
        t_2 = time.time()
        print("Time to solve problem: " + str(t_2 - t_1))
        print(f"The minimum Cost of Network: {self.prob.solution.get_objective_value()}")
        print(f"Number of Variables = {self.prob.variables.get_num()}")
        print(f"Number of Conditions = {self.prob.linear_constraints.get_num()}")
        sol_dict = create_sol_dict(self.prob)
        return sol_dict, self.prob

    def optimisation_satellite_and_core_network_run(self,C_ij_core, C_is_core, M = 5e9,  time_limit=1e5):
        """
               set up and solve the problem for minimising the overall cost of the network
               """
        t_0 = time.time()
        print("Start Optimisation")
        self.add_max_capacity_contraint()
        self.add_required_flow_rate_constraint()
        self.add_conservation_of_flow_constraint()
        self.ensure_only_one_ground_station_per_core_site_constraint()
        self.ensure_only_allowed_ground_stations_can_be_on_constraint()
        self.calculate_total_transmission_requirement_constraint()
        self.get_Tnim_constraint(M)
        self.objective_minimise_cost(C_ij_core, C_is_core)
        self.prob.write("test_1.lp")
        self.prob.parameters.lpmethod.set(3)
        self.prob.parameters.mip.limits.cutpasses.set(1)
        self.prob.parameters.mip.strategy.probe.set(-1)
        self.prob.parameters.mip.strategy.variableselect.set(4)
        self.prob.parameters.mip.strategy.kappastats.set(1)
        self.prob.parameters.mip.tolerances.integrality.set(float(1e-10))
        self.prob.parameters.mip.tolerances.mipgap.set(float(0.01))
        # prob.parameters.simplex.limits.iterations = 50
        print(self.prob.parameters.get_changed())
        self.prob.parameters.timelimit.set(time_limit)
        t_1 = time.time()
        print("Time to set up problem: " + str(t_1 - t_0))
        self.prob.solve()
        t_2 = time.time()
        print("Time to solve problem: " + str(t_2 - t_1))
        print(f"The minimum Cost of Network: {self.prob.solution.get_objective_value()}")
        print(f"Number of Variables = {self.prob.variables.get_num()}")
        print(f"Number of Conditions = {self.prob.linear_constraints.get_num()}")
        sol_dict = create_sol_dict(self.prob)
        return sol_dict, self.prob


class Satellite_Full_Network_Problems_Variable_Parameters(Satellite_Full_Network_Problems):

    def __init__(self, prob, sat_keys, g, key_dict, Tij, ground_station_keys, core_site_keys):
        self.prob = prob
        # [satellite_RAAN]
        self.sat_keys = sat_keys
        # {core_site: [allowed ground stations]}
        self.core_site_keys = core_site_keys
        # [ground_stations]
        self.ground_station_keys = ground_station_keys
        # graph of the underlaying effective network
        self.g = g
        # {(ground_station_1, ground_station_2): 1}
        self.key_dict = key_dict
        # {(core_site_1, core_stite_2): Tij}
        self.Tij  = Tij



    def get_Tnim_constraint(self, gammaij, M):
        Min_var = []
        for i in self.core_site_keys:
            for n in self.ground_station_keys:
                Min_var.append(f"M_{i}_{n}")
        self.prob.variables.add(names=Min_var, types=[self.prob.variables.type.continuous] * len(Min_var))
        for k in self.key_dict:
            for l in self.core_site_keys:
                ind = [f"t_{k[0]}_{l}_{k[1]}", f"M_{l}_{k[0]}"]
                val = [1,1]
                for j in self.core_site_keys:
                    if l != j:
                        ind.extend([f"x_{j}_{k[1]}"])
                        if gammaij != None:
                            val.extend([-(self.Tij[l,j] + self.Tij[j,l]) * gammaij[l,j]])
                        else:
                            val.extend([-(self.Tij[l, j] + self.Tij[j, l])])
                contraint = [cplex.SparsePair(ind=ind, val=val)]
                self.prob.linear_constraints.add(lin_expr=contraint, senses=["G"], rhs=[0])
        for i in self.core_site_keys:
            for n in self.ground_station_keys:
                ind = [f"M_{i}_{n}", f"x_{i}_{n}"]
                val = [1, +M]
                contraint = [cplex.SparsePair(ind=ind, val=val)]
                self.prob.linear_constraints.add(lin_expr=contraint, senses=["L"], rhs=[M])




    def objective_minimise_cost(self, C_ij_core, C_is_core):
        alphaijs_var = []
        for i in self.core_site_keys:
            for j in self.core_site_keys:
                if i < j:
                    for n in self.ground_station_keys:
                        alphaijs_var.append(f"alpha_{i}_{j}_{n}")
        self.prob.variables.add(names=alphaijs_var, types=[self.prob.variables.type.binary] * len(alphaijs_var))
        for i in self.core_site_keys:
            for j in self.core_site_keys:
                if i < j:
                    for n in self.ground_station_keys:
                        ind = [f"alpha_{i}_{j}_{n}", f"x_{i}_{n}", f"x_{j}_{n}"]
                        val = [1, -1, -1]
                        contraint = [cplex.SparsePair(ind=ind, val=val)]
                        self.prob.linear_constraints.add(lin_expr=contraint, senses=["G"], rhs=[-1])

        obj_vals = []
        for sat in self.sat_keys:
            obj_vals.append((f"N_sat_{sat}", 1))
        for i in self.core_site_keys:
            for j in self.core_site_keys:
                if i < j:
                    for n in self.ground_station_keys:
                        obj_vals.append((f"alpha_{i}_{j}_{n}",C_ij_core[i,j]))
        for i in self.core_site_keys:
            for s in self.ground_station_keys:
                obj_vals.append((f"x_{i}_{s}", C_is_core[i,s]))
        self.prob.objective.set_linear(obj_vals)
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)


    def optimisation_satellite_and_core_network_run(self,gammaij,C_ij_core, C_is_core, M = 5e9,  time_limit=1e5):
        """
               set up and solve the problem for minimising the overall cost of the network
               """
        t_0 = time.time()
        print("Start Optimisation")
        self.add_max_capacity_contraint()
        self.add_required_flow_rate_constraint()
        self.add_conservation_of_flow_constraint()
        self.ensure_only_one_ground_station_per_core_site_constraint()
        self.ensure_only_allowed_ground_stations_can_be_on_constraint()
        self.calculate_total_transmission_requirement_constraint()
        self.get_Tnim_constraint(gammaij,M)
        self.objective_minimise_cost(C_ij_core, C_is_core)
        self.prob.write("test_1.lp")
        self.prob.parameters.lpmethod.set(3)
        self.prob.parameters.mip.limits.cutpasses.set(1)
        self.prob.parameters.mip.strategy.probe.set(-1)
        self.prob.parameters.mip.strategy.variableselect.set(4)
        self.prob.parameters.mip.strategy.kappastats.set(1)
        self.prob.parameters.mip.tolerances.integrality.set(float(1e-10))
        self.prob.parameters.mip.tolerances.mipgap.set(float(0.01))
        # prob.parameters.simplex.limits.iterations = 50
        print(self.prob.parameters.get_changed())
        self.prob.parameters.timelimit.set(time_limit)
        t_1 = time.time()
        print("Time to set up problem: " + str(t_1 - t_0))
        self.prob.solve()
        t_2 = time.time()
        print("Time to solve problem: " + str(t_2 - t_1))
        print(f"The minimum Cost of Network: {self.prob.solution.get_objective_value()}")
        print(f"Number of Variables = {self.prob.variables.get_num()}")
        print(f"Number of Conditions = {self.prob.linear_constraints.get_num()}")
        sol_dict = create_sol_dict(self.prob)
        return sol_dict, self.prob


def calculate_cost_local_network(binary_on_off_dict, alpha_dict, C_ij_core, C_is_core):
    cost = 0.0
    for a in alpha_dict:
        a_split = a.split("_")
        i = a_split[1]
        j = a_split[2]
        cost += alpha_dict[a] * C_ij_core[i,j]
    for x in binary_on_off_dict:
        x_split = x.split("_")
        i = x_split[1]
        s = x_split[2]
        cost += binary_on_off_dict[x] * C_is_core[i,s]
    return cost


def calculate_C_ij_core(core_sites, cost_factor_per_km):
    C_ij_core = {}
    for core_site in core_sites.keys():
        for core_site_2 in core_sites.keys():
            if core_site != core_site_2:
                lat_lon_1 = sites_lat_lon[core_site]
                lat_lon_2 = sites_lat_lon[core_site_2]
                dist = great_circle_distance_calculation(np.radians(lat_lon_1[0]), np.radians(lat_lon_2[0]), np.radians(lat_lon_1[1]), np.radians(lat_lon_2[1]))
                C_ij_core[(core_site,core_site_2)] = dist * cost_factor_per_km/ 1000
    return C_ij_core

def calculate_C_is_core(core_sites, ground_station_list, cost_factor_per_km):
    C_is_core = {}
    for core_site in core_sites.keys():
        for ground_station in ground_station_list:
            lat_lon_1 = sites_lat_lon[core_site]
            lat_lon_2 = ground_station_potential_sites[ground_station]
            dist = great_circle_distance_calculation(np.radians(lat_lon_1[0]), np.radians(lat_lon_2[0]), np.radians(lat_lon_1[1]), np.radians(lat_lon_2[1]))
            C_is_core[(core_site,ground_station)] = dist * cost_factor_per_km/ 1000
    return C_is_core


def calculate_true_Tij(ground_stations, x_dict, Tij):
    tij_req_array = {}
    for i in ground_stations:
        for j in ground_stations:
            tij_req = 0.0
            if i != j:
                for key in x_dict:
                    k = key.split("_")
                    core_site = k[1]
                    area = k[2]
                    if area == i:
                        for key_2 in x_dict:
                            k_2 = key_2.split("_")
                            core_site_2 = k_2[1]
                            area_2 = k_2[2]
                            if area_2 == j and core_site != core_site_2:
                                tij_req += Tij[(core_site, core_site_2)] * x_dict[key] * x_dict[key_2]
                tij_req_array[i,j] = tij_req
    return tij_req_array

def check_chi_array_alpha_array(core_sites, ground_stations, x_dict, alpha_dict):
    chi_total ={}
    for key in x_dict.keys():
        k = key.split("_")
        core_site = k[1]
        area = k[2]
        if core_site in chi_total.keys():
            chi_total[core_site] += x_dict[key]
        else:
            chi_total[core_site] = x_dict[key]
    all_chi_total_1 = all(abs(value-1)< 0.001 for value in chi_total.values())
    alphas_correct = True
    violation_list = []
    for i in core_sites.keys():
        for j in core_sites.keys():
            if i < j:
                for s in ground_stations:
                    alpha_true = x_dict[f"x_{i}_{s}"] * x_dict[f"x_{j}_{s}"]
                    if abs(alpha_true - alpha_dict[f"alpha_{i}_{j}_{s}"]) > 0.01:
                        alphas_correct = False
                        violation_list.append(f"alpha_{i}_{j}_{s}")
    return all_chi_total_1, chi_total, alphas_correct, violation_list


def check_tij_correct(tij_req_array, tij_true_array):
    tij_check_array = {}
    for key in tij_true_array.keys():
        tij = key.split("_")
        i = tij[1]
        j = tij[2]
        tij_check_array[key] = bool(abs((tij_true_array[key] - tij_req_array[i,j] - tij_req_array[j,i])/tij_true_array[key]) <=  1e-4)
    return tij_check_array



def select_appropriate_M(Tij_list, core_sites, epsilon_tol = 1e-10):
    tij_min = min(list(Tij_list.values()))
    max_sum_t = 0.0
    for i in core_sites.keys():
        current_sum_t =0.0
        for j in core_sites.keys():
            if i != j:
                current_sum_t += Tij_list[i,j] + Tij_list[j,i]
        if current_sum_t > max_sum_t:
            max_sum_t = current_sum_t
    fractional_error = max_sum_t * epsilon_tol / tij_min
    return max_sum_t , fractional_error


def get_alpha_ijs_values(core_site_keys, ground_station_keys, chi_dict):
    alpha_ijs_dict = {}
    for i in core_site_keys:
        for j in core_site_keys:
            if i < j:
                for s in ground_station_keys:
                    alpha_ijs_dict[f"alpha_{i}_{j}_{s}"] = chi_dict[f"x_{i}_{s}"] * chi_dict[f"x_{j}_{s}"]
    return alpha_ijs_dict



def get_gamma_ij(alpha_dict, core_site_keys, ground_station_keys, gamma_dict= None):
    gamma_ij = {}
    for i in core_site_keys:
        for j in core_site_keys:
            if i < j:
                total = 0
                for s in ground_station_keys:
                    total += alpha_dict[f"alpha_{i}_{j}_{s}"]
                if total < 0.001:
                    if gamma_dict == None:
                        gamma_ij[i,j] =  1
                        gamma_ij[j,i] = 1
                    else:
                        if gamma_dict[i,j] == 0:
                            gamma_ij[i, j] = 0
                            gamma_ij[j, i] = 0
                        else:
                            gamma_ij[i, j] = 1
                            gamma_ij[j, i] = 1
                else:
                    gamma_ij[i, j] = 0
                    gamma_ij[j, i] = 0
    return gamma_ij

def get_Cij_core_Cis_core(alpha_dict, chi_dict, core_site_keys, ground_station_keys, cost_factor_per_km, gamma_dict= None, Cis_core_prev = None):
    C_ij_core = {}
    for core_site in core_site_keys:
        for core_site_2 in core_site_keys:
            if core_site != core_site_2:
                lat_lon_1 = sites_lat_lon[core_site]
                lat_lon_2 = sites_lat_lon[core_site_2]
                dist = great_circle_distance_calculation(np.radians(lat_lon_1[0]), np.radians(lat_lon_2[0]),
                                                         np.radians(lat_lon_1[1]), np.radians(lat_lon_2[1]))
                if core_site < core_site_2:
                    total = 0
                    for s in ground_station_keys:
                        total += alpha_dict[f"alpha_{core_site}_{core_site_2}_{s}"]
                else:
                    total = 0
                    for s in ground_station_keys:
                        total += alpha_dict[f"alpha_{core_site_2}_{core_site}_{s}"]
                if total < 0.001:
                    if gamma_dict == None:
                        C_ij_core[(core_site, core_site_2)] = dist * cost_factor_per_km / 1000
                    else:
                        if gamma_dict[core_site,core_site_2] == 0:
                            C_ij_core[(core_site, core_site_2)] = 0.0
                        else:
                            C_ij_core[(core_site, core_site_2)] = dist * cost_factor_per_km / 1000
                else:
                    C_ij_core[(core_site, core_site_2)] = 0.0
    C_is_core = {}
    for core_site in core_site_keys:
        for ground_station in ground_station_keys:
            lat_lon_1 = sites_lat_lon[core_site]
            lat_lon_2 = ground_station_potential_sites[ground_station]
            dist = great_circle_distance_calculation(np.radians(lat_lon_1[0]), np.radians(lat_lon_2[0]),
                                                     np.radians(lat_lon_1[1]), np.radians(lat_lon_2[1]))
            if chi_dict[f"x_{core_site}_{ground_station}"] < 0.001:
                if Cis_core_prev == None:
                    C_is_core[(core_site, ground_station)] = dist * cost_factor_per_km / 1000
                else:
                    if Cis_core_prev[(core_site, ground_station)] < 0.001:
                        C_is_core[(core_site, ground_station)] = 0.0
                    else:
                        C_is_core[(core_site, ground_station)] = dist * cost_factor_per_km / 1000
            else:
                C_is_core[(core_site, ground_station)] = 0.0
    return C_ij_core, C_is_core




def get_cost_terms(core_sites_file_paths, key_dict_file_paths, satellite_rates_file_paths, Tij_file_paths, ground_station_list,cost_factor_per_km, data_storage_location_keep_each_loop = None):
    graphs = {}
    key_dicts = {}
    sat_keys = {}
    core_sites = {}
    ground_stations = {}
    Tij_s =  {}
    for i in range(len(ground_station_list)):
        graph, key_dict, sat_key, core_site, ground_station, Tij = import_full_problem(
            core_sites_file_path= core_sites_file_paths + str(ground_station_list[i]) + ".csv",
            key_dict_file_path= key_dict_file_paths + str(ground_station_list[i]) + ".csv",
            satellite_rates_file_path=satellite_rates_file_paths + str(ground_station_list[i]) + ".csv",
            Tij_file_path= Tij_file_paths + str(ground_station_list[i]) + ".csv")
        graphs[ground_station_list[i]] = graph[0]
        key_dicts[ground_station_list[i]] = key_dict[0]
        sat_keys[ground_station_list[i]] = sat_key[0]
        core_sites[ground_station_list[i]] = core_site[0]
        ground_stations[ground_station_list[i]] = ground_station[0]
        Tij_s[ground_station_list[i]] = Tij[0]
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_ground_station_number = last_row_explored["Number_of_Ground_Stations"].iloc[0]
        else:
            current_ground_station_number = None
            dictionary_fieldnames = ["Number_of_Ground_Stations", "objective_value", "satellite_cost", "core_cost"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_ground_station_number = None

    no_solution_list = []
    objective_list = {}
    for gs_number in graphs.keys():
        if current_ground_station_number != None and current_ground_station_number != int(gs_number):
            continue
        elif current_ground_station_number != None and current_ground_station_number == int(gs_number):
            current_ground_station_number = None
            continue
        try:
            C_ij_core = calculate_C_ij_core(core_sites[gs_number], cost_factor_per_km)
            C_is_core = calculate_C_is_core(core_sites[gs_number], ground_stations[gs_number], cost_factor_per_km)
            M, fractional_error  = select_appropriate_M(Tij_s[gs_number], core_sites[gs_number], epsilon_tol=1e-10)
            prob = cplex.Cplex()
            sat_prob = Satellite_Full_Network_Problems(prob, sat_keys[gs_number], graphs[gs_number], key_dicts[gs_number], Tij_s[gs_number],
                                                       ground_stations[gs_number], core_sites[gs_number])
            sol_dict, prob = sat_prob.optimisation_satellite_and_core_network_run(C_ij_core, C_is_core, M = M, time_limit=1e3)
            binary_on_off_dict, satellite_dict, alpha_dict, Tij_key, Tinj_key, M_key = split_sol_to_flow(sol_dict)
            satellite_cost = sum(list(satellite_dict.values()))
            tij_req_array = calculate_true_Tij(ground_stations[gs_number], binary_on_off_dict, Tij_s[gs_number])
            all_chi_total_1, chi_total, alphas_correct, violation_list = check_chi_array_alpha_array(core_sites[gs_number], ground_stations[gs_number], binary_on_off_dict, alpha_dict)
            tij_check_array =  check_tij_correct(tij_req_array, Tij_key)

            core_network_cost = calculate_cost_local_network(binary_on_off_dict, alpha_dict, C_ij_core, C_is_core)
            objective_value = prob.solution.get_objective_value()
            objective_list[gs_number] =  (objective_value, satellite_cost, core_network_cost)
            if data_storage_location_keep_each_loop != None:
                dictionary = [
                    {"Number_of_Ground_Stations": gs_number,
                     "objective_value": objective_value, "satellite_cost": satellite_cost, "core_cost": core_network_cost}]
                dictionary_fieldnames = ["Number_of_Ground_Stations", "objective_value", "satellite_cost", "core_cost"]
                if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)

        except:
            no_solution_list.append(gs_number)
            continue
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["Number_of_Ground_Stations"] not in objective_list.keys():
                objective_list[row["Number_of_Ground_Stations"]] = (row["objective_value"], row["satellite_cost"], row["core_cost"])
    #

    # sat_number = ("3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18")
    #
    #
    # cost_total = {"Satellite Cost": np.array([748, 943, 1004, 1118, 1124, 1152, 1177, 1187,1258, 1267, 1284, 1328, 1331, 1326, 1326, 1336]),
    #               "Core Network Cost": np.array([16703.9, 7381.2, 4271.2, 2680.6, 2570.7, 1915.6, 1580.7, 1295.4, 1155.9, 1084.5, 951.9, 842.3, 646.9, 587.7, 547.7, 501.1])}
    # cost_total_sat = {"Satellite Cost": np.array(
    #     [748, 943, 1004, 1118, 1124, 1152, 1177, 1187, 1258, 1267, 1284, 1328, 1331, 1326, 1326, 1336]),
    #               "Core Network Cost": np.array(
    #                   [83.5, 36.9, 21.4, 13.4, 12.9, 9.6, 7.9, 6.5, 5.8, 5.4, 4.8, 4.2, 3.2, 2.9, 2.7, 2.5])}
    # cost_total_cheap_sat = {"Satellite Cost": np.array(
    #     [748, 943, 1004, 1118, 1124, 1152, 1177, 1187, 1258, 1267, 1284, 1328, 1331, 1326, 1326, 1336]),
    #     "Core Network Cost": np.array(
    #         [835, 369, 214, 134, 129, 96, 79, 65, 58, 54, 48, 42, 32, 29, 27, 25])}
    sat_number = (3, 4, 5, 6,7,8,9,10,11,12)

    cost_total = {"Satellite Cost Shortest Paths": np.array(
        [748, 943, 1004, 1118, 1124, 1152, 1177, 1187,1258, 1267]),
                  "Core Network Cost Shortest Paths": np.array(
                      [16703.9, 7381.2, 4271.2, 2680.6, 2570.7, 1915.6, 1580.7, 1295.4,  1155.9, 1084.5])}
    cost_total_sat = {"Satellite Cost": np.array(
        [748, 943, 1004, 1118, 1124, 1152, 1177, 1187, 1258, 1267, 1284, 1328, 1331, 1326, 1326]),
        "Core Network Cost": np.array(
            [83.5, 36.9, 21.4, 13.4, 12.9, 9.6, 7.9, 6.5, 5.8, 5.4, 4.8, 4.2, 3.2, 2.9, 2.7])}
    cost_total_cheap_sat = {"Satellite Cost": np.array(
        [748, 943, 1004, 1118, 1124, 1152, 1177, 1187, 1258, 1267, 1284, 1328, 1331, 1326, 1326]),
        "Core Network Cost": np.array(
            [835, 369, 214, 134, 129, 96, 79, 65, 58, 54, 48, 42, 32, 29, 27])}
    x = np.arange(len(sat_number))
    width = 0.3
    multiplier = -1
    cost_current_values = {"Satellite Cost Full Optimisation": [], "Core Network Cost Full Optimisation" : []}
    for key in objective_list.keys():
        cost_current_values["Satellite Cost Full Optimisation"].append(objective_list[key][1])
        cost_current_values["Core Network Cost Full Optimisation"].append(objective_list[key][2])
    cost_array = {"Shortest Path": cost_total, "Choice of Ground Stations": cost_current_values}


    fig, ax = plt.subplots()
    for attribute, measurement in cost_array.items():
        bottom = np.zeros(len(sat_number))
        offset = width * multiplier
        for boolean, cost in cost_array[attribute].items():
            sat_number_temp  = [sat_i + width * multiplier / 2 for sat_i in sat_number]
            p = ax.bar(sat_number_temp, cost, width * multiplier, label = boolean, bottom = bottom)
            bottom += cost
        if multiplier == -1:
            multiplier = 1
        elif multiplier== 1:
            multiplier = -1

    labels = []
    for sat_no in sat_number:
        labels.extend([str(sat_no) + ", Shortest Path", str(sat_no) + ", Choice of Ground Stations"])


    ax.set_xlabel("Number of Ground Stations")
    ax.set_ylabel("Overall Cost of Network Normalised by Satellite Cost")
    # ax.x_label(labels)
    # ax.set_xticks(x + width, labels)
    ax.legend(loc = "best")
    plt.savefig("satellite_optimisation/bar_plot_cost_CubeSatWithFullData.png")
    plt.show()


def get_cost_terms_optimising_sat_number_only(core_sites_file_paths, key_dict_file_paths, satellite_rates_file_paths, Tij_file_paths, ground_station_list,cost_factor_per_km, data_storage_location_keep_each_loop = None):
    graphs = {}
    key_dicts = {}
    sat_keys = {}
    core_sites = {}
    ground_stations = {}
    Tij_s =  {}
    for i in range(len(ground_station_list)):
        graph, key_dict, sat_key, core_site, ground_station, Tij = import_full_problem(
            core_sites_file_path= core_sites_file_paths + str(ground_station_list[i]) + ".csv",
            key_dict_file_path= key_dict_file_paths + str(ground_station_list[i]) + ".csv",
            satellite_rates_file_path=satellite_rates_file_paths + str(ground_station_list[i]) + ".csv",
            Tij_file_path= Tij_file_paths + str(ground_station_list[i]) + ".csv")
        graphs[ground_station_list[i]] = graph[0]
        key_dicts[ground_station_list[i]] = key_dict[0]
        sat_keys[ground_station_list[i]] = sat_key[0]
        core_sites[ground_station_list[i]] = core_site[0]
        ground_stations[ground_station_list[i]] = ground_station[0]
        Tij_s[ground_station_list[i]] = Tij[0]
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_ground_station_number = last_row_explored["Number_of_Ground_Stations"].iloc[0]
        else:
            current_ground_station_number = None
            dictionary_fieldnames = ["Number_of_Ground_Stations", "objective_value", "satellite_cost", "core_cost"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_ground_station_number = None

    no_solution_list = []
    objective_list = {}
    for gs_number in graphs.keys():
        if current_ground_station_number != None and current_ground_station_number != int(gs_number):
            continue
        elif current_ground_station_number != None and current_ground_station_number == int(gs_number):
            current_ground_station_number = None
            continue
        try:
            C_ij_core = calculate_C_ij_core(core_sites[gs_number], cost_factor_per_km)
            C_is_core = calculate_C_is_core(core_sites[gs_number], ground_stations[gs_number], cost_factor_per_km)
            C_ij_core_zero = calculate_C_ij_core(core_sites[gs_number], 0.0)
            C_is_core_zero = calculate_C_is_core(core_sites[gs_number], ground_stations[gs_number], 00.0)
            M, fractional_error  = select_appropriate_M(Tij_s[gs_number], core_sites[gs_number], epsilon_tol=1e-10)
            prob = cplex.Cplex()
            sat_prob = Satellite_Full_Network_Problems(prob, sat_keys[gs_number], graphs[gs_number], key_dicts[gs_number], Tij_s[gs_number],
                                                       ground_stations[gs_number], core_sites[gs_number])
            sol_dict, prob = sat_prob.optimisation_satellite_and_core_network_run(C_ij_core, C_is_core, M = M, time_limit=1000)
            binary_on_off_dict, satellite_dict, alpha_dict, Tij_key, Tinj_key, M_key = split_sol_to_flow(sol_dict)
            # alpha_dict =get_alpha_ijs_values(core_sites[gs_number], ground_stations[gs_number], binary_on_off_dict)
            satellite_cost = sum(list(satellite_dict.values()))
            # tij_req_array = calculate_true_Tij(ground_stations[gs_number], binary_on_off_dict, Tij_s[gs_number])
            # all_chi_total_1, chi_total, alphas_correct, violation_list = check_chi_array_alpha_array(core_sites[gs_number], ground_stations[gs_number], binary_on_off_dict, alpha_dict)
            # tij_check_array =  check_tij_correct(tij_req_array, Tij_key)
            dictionary = []
            for key in satellite_dict:
                dictionary = [{"Name": key, "Value": satellite_dict[key]}]
                dictionary_fieldnames = ["Name", "Value"]
                if os.path.isfile('SolutionFileSat.csv'):
                    with open('SolutionFileSat.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open('SolutionFileSat.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)
            core_network_cost = calculate_cost_local_network(binary_on_off_dict, alpha_dict, C_ij_core, C_is_core)
            objective_value = prob.solution.get_objective_value()
            objective_list[gs_number] =  (objective_value, satellite_cost, core_network_cost)
            if data_storage_location_keep_each_loop != None:
                dictionary = [
                    {"Number_of_Ground_Stations": gs_number,
                     "objective_value": objective_value, "satellite_cost": satellite_cost, "core_cost": core_network_cost}]
                dictionary_fieldnames = ["Number_of_Ground_Stations", "objective_value", "satellite_cost", "core_cost"]
                if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)

        except:
            no_solution_list.append(gs_number)
            continue
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["Number_of_Ground_Stations"] not in objective_list.keys():
                objective_list[row["Number_of_Ground_Stations"]] = (row["objective_value"], row["satellite_cost"], row["core_cost"])
    #


    # cost_total = {"Satellite Cost": np.array([15,18,21,28,28,31,34,34,35,35,36,37,37,37,37,38]),
    #               "Core Network Cost": np.array(
    #                   [10375.6, 5243.6, 4022.7, 1929.4, 1865.4, 1185.1, 1084.9, 917.4, 869.7, 832.5, 655.7, 573.3, 490.4, 460.37, 448.2, 410.9])}
    # cost_total_sat = {"Satellite Cost": np.array(
    #     [15,18,21,28,28,31,34,34,35,35,36,37,37,37,37,38]),
    #     "Core Network Cost": np.array(
    #         [51.9, 26.2, 20.1, 9.65, 9.33, 5.93, 5.43, 4.59, 4.35, 4.16, 3.28, 2.87, 2.45, 2.3, 2.24, 2.05])}
    #
    # cost_total = {"Satellite Cost Shortest Path": np.array([15, 18, 21, 28, 28, 31, 34, 34]),
    #               "Core Network Cost Shortest Path": np.array(
    #                   [10375.6, 5243.6, 4022.7, 1929.4, 1865.4, 1185.1, 1084.9, 917.4])}
    # cost_total_sat = {"Satellite Cost Shortest Path": np.array(
    #     [15, 18, 21, 28, 28, 31, 34, 34]),
    #     "Core Network Cost Shortest Path": np.array(
    #         [51.9, 26.2, 20.1, 9.65, 9.33, 5.93, 5.43, 4.59])}

    cost_total = {"Satellite Cost Shortest Path": np.array([105,102,110,110,111,112,112,110,159,159]),
                  "Core Network Cost Shortest Path": np.array(
                      [3847, 3611, 3154, 3073, 3040, 2750, 2649, 1572, 1232, 1232])}
    cost_total_sat = {"Satellite Cost Shortest Path": np.array(
        [105,102,110,110,111,112,112,110,159,159]),
        "Core Network Cost Shortest Path": np.array(
            [19.3, 18.055, 15.77, 15.37, 15.20, 13.75, 13.24, 7.86, 6.16, 6.16])}

    sat_number = (9,10,11,12,13,14,15,16,17,18)

    # FOR 20 CORE SITES

    x = np.arange(len(sat_number))
    width = 0.3
    multiplier = -1
    cost_current_values = {"Satellite Cost Full Optimisation": [], "Core Network Cost Full Optimisation" : []}
    for key in objective_list.keys():
        cost_current_values["Satellite Cost Full Optimisation"].append(objective_list[key][1])
        cost_current_values["Core Network Cost Full Optimisation"].append(objective_list[key][2])
    cost_array = {"Shortest Path": cost_total, "Choice of Ground Stations": cost_current_values}


    fig, ax = plt.subplots()
    for attribute, measurement in cost_array.items():
        bottom = np.zeros(len(sat_number))
        offset = width * multiplier
        for boolean, cost in cost_array[attribute].items():
            sat_number_temp  = [sat_i + width * multiplier / 2 for sat_i in sat_number]
            p = ax.bar(sat_number_temp, cost, width * multiplier, label = boolean, bottom = bottom)
            bottom += cost
        if multiplier == -1:
            multiplier = 1
        elif multiplier== 1:
            multiplier = -1

    labels = []
    for sat_no in sat_number:
        labels.extend([str(sat_no) + ", Shortest Path", str(sat_no) + ", Choice of Ground Stations"])


    ax.set_xlabel("Number of Ground Stations")
    ax.set_ylabel("Overall Cost of Network Normalised by Satellite Cost")
    # ax.x_label(labels)
    # ax.set_xticks(x + width, labels)
    ax.legend(loc = "best")
    plt.savefig("satellite_optimisation/bar_plot_cost_CubeSatWithFullData_DataCentresGlobal.png")
    plt.show()

def optimisation_chaining(core_sites_file_paths, key_dict_file_paths, satellite_rates_file_paths, Tij_file_paths,years, cost_factor_per_km, data_storage_location_keep_each_loop = None, data_storage_location_keep_each_loop_keep_last_soln = None):
    graphs = {}
    key_dicts = {}
    sat_keys = {}
    core_sites = {}
    ground_stations = {}
    Tij_s =  {}
    for i in range(len(years)):
        graph, key_dict, sat_key, core_site, ground_station, Tij = import_full_problem(
            core_sites_file_path= "satellite_optimisation/Data_Centres_Global_" + str(years[i]) + core_sites_file_paths + ".csv",
            key_dict_file_path="satellite_optimisation/Data_Centres_Global_" + str(years[i]) + key_dict_file_paths + ".csv",
            satellite_rates_file_path="satellite_optimisation/Data_Centres_Global_" + str(years[i]) +satellite_rates_file_paths + ".csv",
            Tij_file_path= "satellite_optimisation/Data_Centres_Global_" + str(years[i]) +Tij_file_paths + ".csv")
        graphs[years[i]] = graph[0]
        key_dicts[years[i]] = key_dict[0]
        sat_keys[years[i]] = sat_key[0]
        core_sites[years[i]] = core_site[0]
        ground_stations[years[i]] = ground_station[0]
        Tij_s[years[i]] = Tij[0]
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_ground_station_number = last_row_explored["Years"].iloc[0]
        else:
            current_ground_station_number = None
            dictionary_fieldnames = ["Years", "objective_value", "satellite_cost", "core_cost"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_ground_station_number = None

    no_solution_list = []
    objective_list = {}
    for year in graphs.keys():
        if current_ground_station_number != None and current_ground_station_number != int(year):
            continue
        elif current_ground_station_number != None and current_ground_station_number == int(year):
            current_ground_station_number = None
            continue
        try:
            C_ij_core = calculate_C_ij_core(core_sites[year], cost_factor_per_km)
            C_is_core = calculate_C_is_core(core_sites[year], ground_stations[year], cost_factor_per_km)
            C_ij_core_zero = calculate_C_ij_core(core_sites[year], 0.0)
            C_is_core_zero = calculate_C_is_core(core_sites[year], ground_stations[year], 00.0)
            M, fractional_error  = select_appropriate_M(Tij_s[year], core_sites[year], epsilon_tol=1e-10)
            prob = cplex.Cplex()
            sat_prob = Satellite_Full_Network_Problems(prob, sat_keys[year], graphs[year], key_dicts[year], Tij_s[year],
                                                       ground_stations[year], core_sites[year])
            sol_dict, prob = sat_prob.optimisation_satellite_and_core_network_run(C_ij_core, C_is_core, M = M, time_limit=3000)
            binary_on_off_dict, satellite_dict, alpha_dict, Tij_key, Tinj_key, M_key = split_sol_to_flow(sol_dict)
            # alpha_dict =get_alpha_ijs_values(core_sites[gs_number], ground_stations[gs_number], binary_on_off_dict)
            satellite_cost = sum(list(satellite_dict.values()))
            # tij_req_array = calculate_true_Tij(ground_stations[gs_number], binary_on_off_dict, Tij_s[gs_number])
            # all_chi_total_1, chi_total, alphas_correct, violation_list = check_chi_array_alpha_array(core_sites[gs_number], ground_stations[gs_number], binary_on_off_dict, alpha_dict)
            # tij_check_array =  check_tij_correct(tij_req_array, Tij_key)
            dictionary = []
            for key in satellite_dict:
                dictionary = [{"Name": key, "Value": satellite_dict[key]}]
                dictionary_fieldnames = ["Name", "Value"]
                if os.path.isfile('SolutionFileSat.csv'):
                    with open('SolutionFileSat.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open('SolutionFileSat.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)
            core_network_cost = calculate_cost_local_network(binary_on_off_dict, alpha_dict, C_ij_core, C_is_core)
            objective_value = prob.solution.get_objective_value()
            objective_list[year] =  (objective_value, satellite_cost, core_network_cost)
            if data_storage_location_keep_each_loop != None:
                dictionary = [
                    {"Years": year,
                     "objective_value": objective_value, "satellite_cost": satellite_cost, "core_cost": core_network_cost}]
                dictionary_fieldnames = ["Years", "objective_value", "satellite_cost", "core_cost"]
                if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)

        except:
            no_solution_list.append(year)
            continue
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["Years"] not in objective_list.keys():
                objective_list[row["Years"]] = (row["objective_value"], row["satellite_cost"], row["core_cost"])
    #


    if data_storage_location_keep_each_loop_keep_last_soln != None:
        if os.path.isfile(data_storage_location_keep_each_loop_keep_last_soln + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop_keep_last_soln + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_ground_station_number = last_row_explored["Years"].iloc[0]
        else:
            current_ground_station_number = None
            dictionary_fieldnames = ["Years", "objective_value", "satellite_cost", "core_cost"]
            with open(data_storage_location_keep_each_loop_keep_last_soln + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_ground_station_number = None
    objective_list_build_up = {}
    no_solution_list = []
    year_0 = list(graphs.keys())[0]
    C_ij_core = calculate_C_ij_core(core_sites[year_0], cost_factor_per_km)
    C_is_core = calculate_C_is_core(core_sites[year_0], ground_stations[year_0], cost_factor_per_km)
    C_ij_core_tot = copy.deepcopy(C_ij_core)
    C_is_core_tot = copy.deepcopy(C_is_core)
    gamma_dict = None
    for year in graphs.keys():
        if current_ground_station_number != None and current_ground_station_number != int(year):
            continue
        elif current_ground_station_number != None and current_ground_station_number == int(year):
            current_ground_station_number = None
            continue
        try:
            M, fractional_error  = select_appropriate_M(Tij_s[year], core_sites[year], epsilon_tol=1e-10)
            prob = cplex.Cplex()
            sat_prob = Satellite_Full_Network_Problems_Variable_Parameters(prob, sat_keys[year], graphs[year], key_dicts[year], Tij_s[year],
                                                       ground_stations[year], core_sites[year])
            sol_dict, prob = sat_prob.optimisation_satellite_and_core_network_run(gamma_dict,C_ij_core, C_is_core, M = M, time_limit=3000)
            binary_on_off_dict, satellite_dict, alpha_dict, Tij_key, Tinj_key, M_key = split_sol_to_flow(sol_dict)
            # alpha_dict =get_alpha_ijs_values(core_sites[gs_number], ground_stations[gs_number], binary_on_off_dict)
            satellite_cost = sum(list(satellite_dict.values()))
            # tij_req_array = calculate_true_Tij(ground_stations[gs_number], binary_on_off_dict, Tij_s[gs_number])
            # all_chi_total_1, chi_total, alphas_correct, violation_list = check_chi_array_alpha_array(core_sites[gs_number], ground_stations[gs_number], binary_on_off_dict, alpha_dict)
            # tij_check_array =  check_tij_correct(tij_req_array, Tij_key)
            dictionary = []

            core_network_cost = calculate_cost_local_network(binary_on_off_dict, alpha_dict, C_ij_core, C_is_core)
            objective_value = prob.solution.get_objective_value()
            objective_list_build_up[year] = (objective_value, satellite_cost, core_network_cost)
            extra_costs= 0
            if gamma_dict != None:
                for core_site in core_sites[year]:
                    for core_site_2 in core_sites[year]:
                        if core_site < core_site_2:
                            extra_costs += (1-gamma_dict[core_site, core_site_2]) * C_ij_core_tot[core_site, core_site_2]
                for core_site in core_sites[year]:
                    for ground_stat in ground_stations[year]:
                        if C_is_core[core_site, ground_stat] < 0.001:
                            extra_costs  += (C_is_core_tot[core_site,ground_stat])
            gamma_dict = get_gamma_ij(alpha_dict, core_site_keys= core_sites[year], ground_station_keys= ground_stations[year], gamma_dict=gamma_dict)
            C_ij_core, C_is_core = get_Cij_core_Cis_core(alpha_dict, chi_dict = binary_on_off_dict, core_site_keys = core_sites[year], ground_station_keys= ground_stations[year], cost_factor_per_km = cost_factor_per_km, gamma_dict= gamma_dict, Cis_core_prev = C_is_core)
            for key in satellite_dict:
                dictionary = [{"Name": key, "Value": satellite_dict[key]}]
                dictionary_fieldnames = ["Name", "Value"]
                if os.path.isfile('SolutionFileSat.csv'):
                    with open('SolutionFileSat.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open('SolutionFileSat.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)

            if data_storage_location_keep_each_loop_keep_last_soln != None:
                dictionary = [
                    {"Years": year,
                     "objective_value": objective_value+ extra_costs, "satellite_cost": satellite_cost, "core_cost": core_network_cost + extra_costs}]
                dictionary_fieldnames = ["Years", "objective_value", "satellite_cost", "core_cost"]
                if os.path.isfile(data_storage_location_keep_each_loop_keep_last_soln + '.csv'):
                    with open(data_storage_location_keep_each_loop_keep_last_soln + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open(data_storage_location_keep_each_loop_keep_last_soln + '.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)

        except:
            no_solution_list.append(year)
            continue

    # cost_total = {"Satellite Cost": np.array([15,18,21,28,28,31,34,34,35,35,36,37,37,37,37,38]),
    #               "Core Network Cost": np.array(
    #                   [10375.6, 5243.6, 4022.7, 1929.4, 1865.4, 1185.1, 1084.9, 917.4, 869.7, 832.5, 655.7, 573.3, 490.4, 460.37, 448.2, 410.9])}
    # cost_total_sat = {"Satellite Cost": np.array(
    #     [15,18,21,28,28,31,34,34,35,35,36,37,37,37,37,38]),
    #     "Core Network Cost": np.array(
    #         [51.9, 26.2, 20.1, 9.65, 9.33, 5.93, 5.43, 4.59, 4.35, 4.16, 3.28, 2.87, 2.45, 2.3, 2.24, 2.05])}
    #
    # cost_total = {"Satellite Cost Shortest Path": np.array([15, 18, 21, 28, 28, 31, 34, 34]),
    #               "Core Network Cost Shortest Path": np.array(
    #                   [10375.6, 5243.6, 4022.7, 1929.4, 1865.4, 1185.1, 1084.9, 917.4])}
    # cost_total_sat = {"Satellite Cost Shortest Path": np.array(
    #     [15, 18, 21, 28, 28, 31, 34, 34]),
    #     "Core Network Cost Shortest Path": np.array(
    #         [51.9, 26.2, 20.1, 9.65, 9.33, 5.93, 5.43, 4.59])}

    if data_storage_location_keep_each_loop_keep_last_soln != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop_keep_last_soln + ".csv")
        for index, row in plot_information.iterrows():
            if row["Years"] not in objective_list_build_up.keys():
                objective_list_build_up[row["Years"]] = (row["objective_value"], row["satellite_cost"], row["core_cost"])

    # FOR 20 CORE SITES

    x = np.arange(len(years))
    width = 2
    multiplier = -1
    cost_current_values = {"Satellite Cost": [], "Core Network Cost" : []}
    for key in objective_list.keys():
        cost_current_values["Satellite Cost"].append(objective_list[key][1])
        cost_current_values["Core Network Cost"].append(objective_list[key][2])
    cost_build_up_values = {"Satellite Cost": [], "Core Network Cost" : []}
    for key in objective_list_build_up.keys():
        cost_build_up_values["Satellite Cost"].append(objective_list_build_up[key][1])
        cost_build_up_values["Core Network Cost"].append(objective_list_build_up[key][2])
    cost_array = {"Network Built From Scratch": cost_current_values, "Network Built Up": cost_build_up_values}


    fig, ax = plt.subplots()
    for attribute, measurement in cost_array.items():
        bottom = np.zeros(len(years))
        offset = width * multiplier
        for boolean, cost in cost_array[attribute].items():
            year_number_temp  = [year + width * multiplier / 2 for year in years]
            p = ax.bar(year_number_temp, cost, width * multiplier, label = boolean + " for " + str(attribute), bottom = bottom)
            bottom += cost
        if multiplier == -1:
            multiplier = 1
        elif multiplier== 1:
            multiplier = -1

    labels = []
    for year in years:
        labels.extend([str(2023 + year) + ", Shortest Path", str(2023 + year) + ", Choice of Ground Stations"])


    ax.set_xlabel("Year")
    ax.set_ylabel("Overall Cost of Network Normalised by Satellite Cost")
    # ax.x_label(labels)
    # ax.set_xticks(x + width, labels)
    ax.legend(loc = "best")
    plt.savefig("satellite_optimisation/bar_plot_cost_SatelliteWithFullData_DataCentreYear.png")
    plt.show()



if __name__ == "__main__":

    # get_cost_terms(core_sites_file_paths = "satellite_optimisation/corrected_core_sites_transmission_modulated_full_range_", key_dict_file_paths = "satellite_optimisation/corrected_key_dict_transmission_modulated_full_range_",
    #                satellite_rates_file_paths = "satellite_optimisation/corrected_satellites_rates_transmission_modulated_full_range_", Tij_file_paths = "satellite_optimisation/corrected_Tij_satellites_path_transmission_modulated_full_range_",
    #                ground_station_list = ["3", "4", "5", "6", "7", "8", "9", "10"], cost_factor_per_km = 0.02, data_storage_location_keep_each_loop="satellite_EU_transmission_requirement_cost_several_02")

    # get_cost_terms_optimising_sat_number_only(
    #     core_sites_file_paths="satellite_optimisation/corrected_core_sites_transmission_modulated_full_range_",
    #     key_dict_file_paths="satellite_optimisation/corrected_key_dict_transmission_modulated_full_range_",
    #     satellite_rates_file_paths="satellite_optimisation/corrected_satellites_rates_transmission_modulated_",
    #     Tij_file_paths="satellite_optimisation/corrected_Tij_satellites_path_transmission_modulated_full_range_",
    #     ground_station_list=["3", "4", "5", "6", "7", "8", "9", "10"], cost_factor_per_km=0.001,
    #     data_storage_location_keep_each_loop="satellite_EU_transmission_requirement_cost_several_0001")
    #  "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"
    # get_cost_terms_optimising_sat_number_only(
        # core_sites_file_paths="satellite_optimisation/Data_Centres_Global_core_sites_transmission_modulated_full_range_",
        # key_dict_file_paths="satellite_optimisation/Data_Centres_Global_key_dict_transmission_modulated_full_range_",
        # satellite_rates_file_paths="satellite_optimisation/Data_Centres_Global_Satellite_Rate_Transmission_Modulated_",
        # Tij_file_paths="satellite_optimisation/Data_Centres_Global_Full_Problem_Tij_satellites_path_transmission_modulated_full_range_",
        # ground_station_list=["9", "10", "11", "12", "13", "14", "15", "16", "17", "18"], cost_factor_per_km=0.02,
        # data_storage_location_keep_each_loop="satellite_optimisation/satellite_data_centres_global_0.02_full_optimisation_results")

    optimisation_chaining(core_sites_file_paths = "_Years_core_sites_transmission_modulated_full_range",
                          key_dict_file_paths = "_Years_key_dict_transmission_modulated_full_range",
                          satellite_rates_file_paths  = "_Years_Future_Satellite_Rate_Transmission_Modulated",
                          Tij_file_paths = "_Years_Full_Problem_Tij_satellites_path_transmission_modulated_full_range",
                          years = [0,10,20,30],
                          cost_factor_per_km = 0.0001, data_storage_location_keep_each_loop="satellite_optimisation/satellite_data_centres_global_00001_years_full_optimisation",
                          data_storage_location_keep_each_loop_keep_last_soln="satellite_optimisation/satellite_data_centres_global_00001_years_optimisation_build_up_from_smaller")

    # graphs, key_dict, sat_keys, core_sites, ground_stations, Tij = import_full_problem(core_sites_file_path = "satellite_optimisation/corrected_core_sites_transmission_modulated_420_km.csv"
    #                                                                                    , key_dict_file_path = "satellite_optimisation/corrected_key_dict_transmission_modulated_420_km.csv",
    #                                                                                    satellite_rates_file_path = "satellite_optimisation/corrected_satellites_rates_transmission_modulated_420_km.csv",
    #                                                                                    Tij_file_path = "satellite_optimisation/corrected_Tij_satellites_path_transmission_modulated_420_km.csv")
    # for key in graphs.keys():
    #     prob = cplex.Cplex()
    #     sat_prob = Satellite_Full_Network_Problems(prob, sat_keys[key], graphs[key], key_dict[key], Tij[key], ground_stations[key], core_sites[key])
    #     sol_dict, prob = sat_prob.optimisation_satellite_run()