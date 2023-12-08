import cplex
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
import os
import pandas as pd
import csv
import copy
from satellite_utils import import_simple_problem
import networkx as nx



def create_sol_dict(prob):
    """
    Create a dictionary with the solution of the parameters
    """
    names=prob.variables.get_names()
    values=prob.solution.get_values()
    sol_dict={names[idx]: (values[idx]) for idx in range(prob.variables.get_num())}
    return sol_dict

def split_sol_to_flow(sol_dict):
    """
    Split the solution dictionary into 2 dictionaries containing the flow variables only and the binary variables only
    Parameters
    ----------
    sol_dict : The solution dictionary containing solutions to the primary flow problem

    Returns
    -------

    """
    flow_dict = {}
    alpha_dict = {}
    satellite_dict = {}
    Tij_key = {}
    Tinj_key = {}
    M_key = {}
    for key in sol_dict:
        # get all keys that are flow and add to dictionary
        if key[0] == "N":
            # get the keys that are binary 'on' 'off' and add to dictionary
            satellite_dict[key] = sol_dict[key]
        elif key[0] == "x":
            flow_dict[key] = sol_dict[key]
    return satellite_dict, flow_dict
class Satellite_Metric_Optimisation:


    def __init__(self, prob, sat_keys, g, key_dict):
        self.prob = prob
        self.sat_keys = sat_keys
        self.g = g
        self.key_dict = key_dict

    def add_max_capacity_contraint(self):
        variable_names = [f'x{i}_{j}_k{k[0]}_{k[1]}' for k in self.key_dict for i, j in list(self.g.edges)]
        self.prob.variables.add(names=variable_names, types=[self.prob.variables.type.continuous] * len(variable_names))
        N_sat_var = []
        for sat in self.sat_keys:
            N_sat_var.append(f"N_sat_{sat}")
        self.prob.variables.add(names=N_sat_var, types=[self.prob.variables.type.integer] * len(N_sat_var))
        for i,j in list(self.g.edges):
            if isinstance(j, (int, float)):
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
        for k in self.key_dict:
            ind = []
            val = []
            for l in self.g.adj[k[1]]:
                ind.extend([f"x{l}_{k[1]}_k{k[0]}_{k[1]}"])
                val.extend([1])
            contraint = [cplex.SparsePair(ind=ind, val=val)]
            self.prob.linear_constraints.add(lin_expr=contraint, senses=["G"], rhs=[self.key_dict[k]])

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

    def objective_minimise_number_satellites(self):
        obj_vals = []
        for sat in self.sat_keys:
            obj_vals.append((f"N_sat_{sat}", 1))
        self.prob.objective.set_linear(obj_vals)
        self.prob.objective.set_sense(self.prob.objective.sense.minimize)

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

    def optimisation_satellite_run(self,  time_limit=1e5):
        """
               set up and solve the problem for minimising the overall cost of the network
               """
        t_0 = time.time()
        print("Start Optimisation")
        self.add_max_capacity_contraint()
        self.add_required_flow_rate_constraint()
        self.add_conservation_of_flow_constraint()
        self.flow_out_sink_in_source_zero()
        self.objective_minimise_number_satellites()
        self.prob.write("test_1.lp")
        self.prob.parameters.lpmethod.set(3)
        self.prob.parameters.mip.limits.cutpasses.set(1)
        self.prob.parameters.mip.strategy.probe.set(-1)
        self.prob.parameters.mip.strategy.variableselect.set(4)
        self.prob.parameters.mip.strategy.kappastats.set(1)
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



def compare_segragation_methods(initial_modulation__factor, final_modulation_factor, rate_core_satellite_file_path, key_dict_file_path, segragation_methods, segragation_methods_to_label, data_storage_location_keep_each_loop = None):
    graphs = {}
    key_dicts = {}
    sat_keys = {}
    for i in range(len(segragation_methods)):
        graph, key_dict, sat_key = import_simple_problem(rate_core_satellite_file_path = rate_core_satellite_file_path + segragation_methods[i] + "_18.csv"
                                                       , key_dict_file_path = key_dict_file_path + segragation_methods[i] + "_18.csv")
        graphs[segragation_methods[i]] = graph[0]
        key_dicts[segragation_methods[i]] = key_dict[0]
        sat_keys[segragation_methods[i]] = sat_key[0]
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_modulation_factor = last_row_explored["Modulation_Factor"].iloc[0]
            current_segragation_method = last_row_explored["Segragation_Method"].iloc[0]
        else:
            current_segragation_method = None
            current_modulation_factor = initial_modulation__factor
            dictionary_fieldnames = ["Modulation_Factor", "Segragation_Method", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_segragation_method = None
        current_modulation_factor = initial_modulation__factor

    no_solution_list = []
    objective_list = {}
    for seg_method in graphs.keys():
        if current_segragation_method != None and current_segragation_method != seg_method:
            continue
        elif current_segragation_method != None and current_segragation_method == seg_method:
            current_segragation_method = None
        for modulation_factor in np.arange(current_modulation_factor, final_modulation_factor, 0.1):
            try:
                key_dict_current = copy.deepcopy(key_dicts[seg_method])
                for key in key_dict_current.keys():
                    key_dict_current[key] = key_dict_current[key] * modulation_factor
                prob = cplex.Cplex()
                optimisation = Satellite_Metric_Optimisation(prob, sat_keys[seg_method], graphs[seg_method], key_dict_current)
                sol_dict, prob = optimisation.optimisation_satellite_run(time_limit=2e2)
                objective_value = prob.solution.get_objective_value()
                satellite_dict, flow_dict =  split_sol_to_flow(sol_dict)
                dictionary = []
                for key in satellite_dict:
                    dictionary = [{"Name": key, "Value": satellite_dict[key]}]
                    dictionary_fieldnames = ["Name", "Value"]
                    if os.path.isfile('SolutionFileSat2.csv'):
                        with open('SolutionFileSat2.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writerows(dictionary)
                    else:
                        with open('SolutionFileSat2.csv', mode='a') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                            writer.writeheader()
                            writer.writerows(dictionary)

                if seg_method not in objective_list.keys():
                    objective_list[seg_method] = {modulation_factor: objective_value}
                else:
                    objective_list[seg_method][modulation_factor] = objective_value
                if data_storage_location_keep_each_loop != None:
                    dictionary = [
                        {"Modulation_Factor": modulation_factor, "Segragation_Method": seg_method,
                         "objective_value": objective_value}]
                    dictionary_fieldnames = ["Modulation_Factor", "Segragation_Method", "objective_value"]
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
                no_solution_list.append(seg_method)
                continue
        current_modulation_factor = initial_modulation__factor
    if data_storage_location_keep_each_loop != None:
        plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
        for index, row in plot_information.iterrows():
            if row["Segragation_Method"] not in objective_list.keys():
                objective_list[row["Segragation_Method"]] = {row["Modulation_Factor"]: row["objective_value"]}
            else:
                objective_list[row["Segragation_Method"]][row["Modulation_Factor"]]= row["objective_value"]
  #
    for seg_method in objective_list.keys():
        mean_values = []

        x = []
        for mod_f in objective_list[seg_method].keys():
            mean_values.append(objective_list[seg_method][mod_f])
            x.append(mod_f)

        plt.plot(x, mean_values, label = segragation_methods_to_label[seg_method][0], color = segragation_methods_to_label[seg_method][1])
    plt.xlabel("Bits per Second Scale Factor", fontsize=12)
    plt.ylabel("Number of Satellites Needed", fontsize=12)
    plt.legend(loc = "best", fontsize = 12)
    plt.savefig("satellite_allocation_variation_with_transmission_requirements_data_centres_global.png")
    plt.show()


def get_results_for_varying_satellite_number(modulation_factor, rate_core_satellite_file_path, key_dict_file_path,ground_station_list, data_storage_location_keep_each_loop = None):
    graphs = {}
    key_dicts = {}
    sat_keys = {}
    for i in range(len(ground_station_list)):
        graph, key_dict, sat_key = import_simple_problem(
            rate_core_satellite_file_path=rate_core_satellite_file_path + str(ground_station_list[i])+ ".csv"
            , key_dict_file_path=key_dict_file_path + str(ground_station_list[i]) + ".csv")
        graphs[ground_station_list[i]] = graph[0]
        key_dicts[ground_station_list[i]] = key_dict[0]
        sat_keys[ground_station_list[i]] = sat_key[0]
    if data_storage_location_keep_each_loop != None:
        if os.path.isfile(data_storage_location_keep_each_loop + '.csv'):
            plot_information = pd.read_csv(data_storage_location_keep_each_loop + ".csv")
            last_row_explored = plot_information.iloc[[-1]]
            current_ground_station_number = last_row_explored["Number_of_Ground_Stations"].iloc[0]
        else:
            current_ground_station_number = None
            dictionary_fieldnames = ["Number_of_Ground_Stations", "objective_value"]
            with open(data_storage_location_keep_each_loop + '.csv', mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                writer.writeheader()
    else:
        current_ground_station_number = None

    no_solution_list = []
    objective_list = {}
    for gs_number in graphs.keys():
        if current_ground_station_number != None and current_ground_station_number != gs_number:
            continue
        elif current_ground_station_number != None and current_ground_station_number == gs_number:
            current_ground_station_number = None
            continue
        try:
            key_dict_current = copy.deepcopy(key_dicts[gs_number])
            for key in key_dict_current.keys():
                key_dict_current[key] = key_dict_current[key] * modulation_factor
            prob = cplex.Cplex()
            optimisation = Satellite_Metric_Optimisation(prob, sat_keys[gs_number], graphs[gs_number],
                                                         key_dict_current)
            sol_dict, prob = optimisation.optimisation_satellite_run(time_limit=2e2)
            objective_value = prob.solution.get_objective_value()
            objective_list[gs_number] =  objective_value
            satellite_dict, flow_dict = split_sol_to_flow(sol_dict)
            dictionary = []
            for key in flow_dict:
                dictionary = [{"Name": key, "Value": flow_dict[key]}]
                dictionary_fieldnames = ["Name", "Value"]
                if os.path.isfile('SolutionFileFlow2.csv'):
                    with open('SolutionFileFlow2.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writerows(dictionary)
                else:
                    with open('SolutionFileFlow2.csv', mode='a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=dictionary_fieldnames)
                        writer.writeheader()
                        writer.writerows(dictionary)

            if data_storage_location_keep_each_loop != None:
                dictionary = [
                    { "Number_of_Ground_Stations": gs_number,
                     "objective_value": objective_value}]
                dictionary_fieldnames = ["Number_of_Ground_Stations", "objective_value"]
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
                objective_list[row["Number_of_Ground_Stations"]] = row["objective_value"]
    # core_cost_CubeSat = [10375.6, 5243.6, 4022.7, 1929.4, 1865.4, 1185.1, 1084.9, 917.4, 869.7, 832.5, 655.7, 573.3, 490.4, 460.37, 448.2, 410.9]
    # core_cost_CheapSat = [835, 369, 214, 134, 129, 96, 79, 65, 58, 54, 48, 42, 32, 29, 27, 25]
    # core_cost_Sat = [51.9, 26.2, 20.1, 9.65, 9.33, 5.93, 5.43, 4.59, 4.35, 4.16, 3.28, 2.87, 2.45, 2.3, 2.24, 2.05]

    core_cost_CubeSat = [3847, 3611, 3154, 3073, 3040, 2750, 2649, 1572, 1232, 1232]
    core_cost_Sat = [19.3, 18.055, 15.77, 15.37, 15.20, 13.75, 13.24, 7.86, 6.16, 6.16]

    width = 0.7
    multiplier = -1
    cost_current_values = {"Satellite Cost": [], "Core Network Cost": []}
    for key in objective_list.keys():
        cost_current_values["Satellite Cost"].append(objective_list[key])
        cost_current_values["Core Network Cost"].append(core_cost_Sat[int(key)-9])
    cost_array = {"Shortest Path": cost_current_values}


    sat_number = ("9", "10", "11", "12", "13", "14", "15", "16", "17", "18")
    fig, ax = plt.subplots()
    bottom = np.zeros(len(sat_number))
    for boolean, cost in cost_array["Shortest Path"].items():
        p = ax.bar(sat_number, cost, width, label = boolean, bottom = bottom)
        bottom += cost

    ax.set_xlabel("Number of Ground Stations")
    ax.set_ylabel("Overall Cost of Network Normalised by Satellite Cost")
    ax.legend(loc = "best")
    plt.savefig("satellite_optimisation/bar_plot_data_centre_global_cost_Satellite.png")
    plt.show()


if __name__ == "__main__":
    compare_segragation_methods(initial_modulation__factor = 0.5, final_modulation_factor = 5, rate_core_satellite_file_path = "~/PycharmProjects/GraphTF/satellite_optimisation/Data_Centres_Global_Satellite_Rate_",
                                key_dict_file_path = "~/PycharmProjects/GraphTF/satellite_optimisation/Data_Centres_Global_Tij_", segragation_methods = ["Key_Rate_Modulated", "No_Allocation", "Random_Allocation", "Transmission_Modulated", "Random_Allocation_N_2"],
                                segragation_methods_to_label={"Key_Rate_Modulated": ["Based on Keys Generated", "tab:orange"], "No_Allocation": ["No Allocation Needed", "tab:green"], "Random_Allocation": ["Random Allocation with M=1","tab:red"],"Random_Allocation_N_2": ["Random Allocation with M=2", "tab:blue"], "Transmission_Modulated": ["Based on Transmission Requirement","tab:purple"]}, data_storage_location_keep_each_loop = "satellite_optimisation/satellite_data_centres_global_full_network")

    # get_results_for_varying_satellite_number(modulation_factor = 1, rate_core_satellite_file_path = "satellite_optimisation/Data_Centres_Global_Satellite_Rate_Transmission_Modulated_", key_dict_file_path = "satellite_optimisation/Data_Centres_Global_Tij_Transmission_Modulated_",
    #                                          ground_station_list = [9,10,11,12,13,14,15,16,17,18], data_storage_location_keep_each_loop="satellite_optimisation/data_centres_global_satellite_shortest_distance_satellite_number_results")
    #
    # ,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
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
    #
    # width = 0.7
    #
    # fig, ax = plt.subplots()
    # bottom = np.zeros(16)
    # for boolean, cost in cost_total_cheap_sat.items():
    #     p = ax.bar(sat_number, cost, width, label = boolean, bottom = bottom)
    #     bottom += cost
    #
    # ax.set_xlabel("Number of Ground Stations")
    # ax.set_ylabel("Overall Cost of Network Normalised by Satellite Cost")
    # ax.legend(loc = "best")
    # plt.savefig("satellite_optimisation/bar_plot_cost_Cheap_Satellite.png")
    # plt.show()



