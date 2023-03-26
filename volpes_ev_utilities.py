import pandas as pd
import pyomo.environ as pyo
import numpy as np

# TODO: Please delete this for the final demo
PATH_TO_GLPK = 'C:\\Users\\tbors\\anaconda3\\Library\\bin\\glpsol.exe'
LOCAL = False


""" OPTIMAL EV DISPATCH """


def ev_optimal_dispatch(price, k_max, t_0,
                        arrival_time, departure_time, arrival_soc, departure_soc,
                        number_of_trucks, p_max, soc_max,
                        p_total_max, residual_load,
                        eta_in=np.arange(0.9, 0.88, -(0.9-0.88)/3),
                        eta_out=np.arange(0.9, 0.88, -(0.9-0.88)/3),
                        V2G=False,
                        time_step_s=3600):
    """
    :param residual_load: inflexible demand at location
    :param price: list of power prices
    :param k_max: number of time steps considered
    :param t_0: time stamp of first time step (e.g., UTC, CET)
    :param arrival_time: list of EV arrival times
    :param departure_time: list of EV expected departure times
    :param arrival_soc: list of EV arrival State of Charge
    :param departure_soc: list of EV required State of Charge at departure
    :param number_of_trucks: number of EVs
    :param p_max: list of maximum charging/ discharging power of each EV
    :param soc_max: list of battery capacity of each EV
    :param p_total_max: total maximum available power at this location
    :param eta_in: efficiency when charging, default 90%. Lower efficiency at higher charging speeds, model as
        piecewise linear
    :param eta_out: efficiency when discharging, default 90%. Lower efficiency at higher charging speeds, model
        as piecewise linear
    :param time_step_s: length of each time step (default: 1 hour)
    :return:
    """

    assert (len(p_max) == number_of_trucks)
    assert (len(soc_max) == number_of_trucks)

    # currently, piecewise linear only supported if both eta in and out are split the same way
    assert (len(eta_in) == len(eta_out))
    pl_max = len(eta_in)

    ev_model = pyo.ConcreteModel()

    # INDEX SETS
    # number of time steps k
    ev_model.k_max = pyo.Param(initialize=k_max)
    ev_model.k = pyo.RangeSet(0, ev_model.k_max)

    # number of buses or trucks t
    ev_model.t_max = pyo.Param(initialize=number_of_trucks)
    ev_model.t = pyo.RangeSet(0, ev_model.t_max - 1)

    # number of piecewise linear elements for charging
    ev_model.pl_max = pyo.Param(initialize=pl_max)
    ev_model.pl = pyo.RangeSet(0, ev_model.pl_max - 1)

    # TRUCK CHARGING MODEL
    # variables (power in/ out, state of charge)
    ev_model.P_in = pyo.Var(ev_model.k, ev_model.t, domain=pyo.NonNegativeReals)
    ev_model.P_in_pl = pyo.Var(ev_model.k, ev_model.t, ev_model.pl, domain=pyo.NonNegativeReals)
    ev_model.eta_in = pyo.Param(ev_model.pl, initialize=eta_in)

    if V2G:
        ev_model.P_out = pyo.Var(ev_model.k, ev_model.t, domain=pyo.NonNegativeReals)
        ev_model.P_out_pl = pyo.Var(ev_model.k, ev_model.t, ev_model.pl, domain=pyo.NonNegativeReals)
        ev_model.eta_out = pyo.Param(ev_model.pl, initialize=eta_out)

    ev_model.SoC = pyo.Var(ev_model.k, ev_model.t, domain=pyo.NonNegativeReals)
    ev_model.SoC_slack = pyo.Var(ev_model.t, domain=pyo.NonNegativeReals)

    # bounds
    for _, index in enumerate(ev_model.P_in_index):
        ev_model.P_in[index].bounds = (0, p_max[index[1]])
        ev_model.SoC[index].bounds = (0, soc_max[index[1]])
        if V2G:
            ev_model.P_out[index].bounds = (0, p_max[index[1]])

    for _, index in enumerate(ev_model.P_in_pl_index):
        ev_model.P_in_pl[index].bounds = (0, p_max[index[1]] / ev_model.pl_max)
        if V2G:
            ev_model.P_out_pl[index].bounds = (0, p_max[index[1]] / ev_model.pl_max)

    # fix SoC before arrival and from departure
    def soc_slack_rule(model, i, t):
        return model.SoC[i, t] + model.SoC_slack[t] >= departure_soc[t]

    ev_model.soc_slack_constraint = pyo.ConstraintList()

    for t in ev_model.t:
        arrival_after = arrival_time[t] - t_0
        arrival_index = np.floor(arrival_after.seconds / time_step_s)
        departure_after = departure_time[t] - t_0
        departure_index = np.floor(departure_after.seconds / time_step_s)

        for k in ev_model.k:
            # if before arrival
            if k < arrival_index:
                # set SoC at arrival and prevent charging
                ev_model.SoC[k, t].fix(arrival_soc[t])
                ev_model.P_in[k, t].fix(0)
                if V2G:
                    ev_model.P_out[k, t].fix(0)
            if k >= departure_index:
                # prevent charging after departure
                ev_model.P_in[k, t].fix(0)
                if V2G:
                    ev_model.P_out[k, t].fix(0)
            if k == departure_index:
                ev_model.soc_slack_constraint.add(ev_model.SoC[k, t] + ev_model.SoC_slack[t] >= departure_soc[t])

    # truck SoC evolution
    def soc_evolution(model, i, t):
        if i < model.k_max:
            if V2G:
                return model.SoC[i + 1, t] == model.SoC[i, t] \
                    + sum(ev_model.eta_in[p] * ev_model.P_in_pl[i, t, p] for p in ev_model.pl) \
                    - sum((1 / ev_model.eta_out[p]) * ev_model.P_out_pl[i, t, p] for p in ev_model.pl)
            else:
                return model.SoC[i + 1, t] == model.SoC[i, t] \
                    + sum(ev_model.eta_in[p] * ev_model.P_in_pl[i, t, p] for p in ev_model.pl)
        else:
            return pyo.Constraint.Skip

    ev_model.SoC_evo = pyo.Constraint(ev_model.k, ev_model.t, rule=soc_evolution)

    # POWER BALANCE LOCATION
    ev_model.P_total = pyo.Var(ev_model.k)
    ev_model.residual_load = pyo.Param(ev_model.k, initialize=residual_load)

    def truck_power_in_total_rule(model, i, t):
        # Sum up the piecewise linear elements charging the truck
        return model.P_in[i, t] == sum(model.P_in_pl[i, t, :])

    def truck_power_out_total_rule(model, i, t):
        # Sum up the piecewise linear elements discharging the truck
        return model.P_out[i, t] == sum(model.P_out_pl[i, t, :])

    def power_balance_rule(model, i):
        # Sum up the total power consumption at the location
        if V2G:
            return model.P_total[i] == sum(model.P_in[i, :]) - sum(model.P_out[i, :]) + model.residual_load[i]
        else:
            return model.P_total[i] == sum(model.P_in[i, :]) + model.residual_load[i]

    def total_power_limit_rule(model, i):
        # Limit the total power consumption
        return model.P_total[i] <= p_total_max

    ev_model.truck_power_in_constraint = pyo.Constraint(ev_model.k, ev_model.t, rule=truck_power_in_total_rule)
    if V2G:
        ev_model.truck_power_out_constraint = pyo.Constraint(ev_model.k, ev_model.t, rule=truck_power_out_total_rule)
    ev_model.power_balance_constraint = pyo.Constraint(ev_model.k, rule=power_balance_rule)
    ev_model.power_limit_constraint = pyo.Constraint(ev_model.k, rule=total_power_limit_rule)

    # OBJECTIVE
    # cost parameters
    ev_model.c = pyo.Param(ev_model.k, initialize=price)

    # cost function
    def cost_function(model):
        return 0.001*pyo.summation(model.c, model.P_total) + 1000000*pyo.summation(model.SoC_slack)

    ev_model.revenue = pyo.Objective(rule=cost_function, sense=pyo.minimize)
    if LOCAL:
        ev_optimiser = pyo.SolverFactory('glpk', executable=PATH_TO_GLPK)
    else:
        ev_optimiser = pyo.SolverFactory('glpk')

    ev_solution = ev_optimiser.solve(ev_model)

    return ev_model, ev_optimiser, ev_solution


def dumb_dispatch_model_EV_fleet(price, k_max, t_0,
                                 arrival_time, departure_time, arrival_soc, departure_soc,
                                 number_of_trucks, p_max, soc_max,
                                 p_total_max, residual_load,
                                 eta_in=np.arange(0.9, 0.88, -(0.9-0.88)/3),
                                 time_step_s=3600):
    """

    :param price: list of power prices
    :param k_max: number of time steps considered
    :param t_0: time stamp of first time step (e.g., UTC, CET)
    :param arrival_time: list of EV arrival times
    :param departure_time: list of EV expected departure times
    :param arrival_soc: list of EV arrival State of Charge
    :param departure_soc: list of EV required State of Charge at departure
    :param number_of_trucks: number of EVs
    :param p_max: list of maximum charging/ discharging power of each EV
    :param soc_max: list of battery capacity of each EV
    :param p_total_max: total maximum available power at this location
    :param eta_in: efficiency when charging, default 90%. Lower efficiency at higher charging speeds, model as
        piecewise linear
    :param eta_out: efficiency when discharging, default 90%. Lower efficiency at higher charging speeds, model
        as piecewise linear
    :param time_step_s: length of each time step (default: 1 hour)
    :return:
    """

    assert (len(p_max) == number_of_trucks)
    assert (len(soc_max) == number_of_trucks)

    pl_max = len(eta_in)

    ev_model = pyo.ConcreteModel()

    # INDEX SETS
    # number of time steps k
    ev_model.k_max = pyo.Param(initialize=k_max)
    ev_model.k = pyo.RangeSet(0, ev_model.k_max)

    # number of buses or trucks t
    ev_model.t_max = pyo.Param(initialize=number_of_trucks)
    ev_model.t = pyo.RangeSet(0, ev_model.t_max-1)

    # number of piecewise linear elements for charging
    ev_model.pl_max = pyo.Param(initialize=pl_max)  # HARD CODED
    ev_model.pl = pyo.RangeSet(0, ev_model.pl_max-1)

    # TRUCK CHARGING MODEL
    # variables (power in/ out, state of charge)
    ev_model.P_in = pyo.Var(ev_model.k, ev_model.t, domain=pyo.NonNegativeReals)
    ev_model.P_in_pl = pyo.Var(ev_model.k, ev_model.t, ev_model.pl, domain=pyo.NonNegativeReals)
    ev_model.eta_in = pyo.Param(ev_model.pl, initialize=eta_in)

    ev_model.SoC = pyo.Var(ev_model.k, ev_model.t, domain=pyo.NonNegativeReals)
    ev_model.SoC_slack = pyo.Var(ev_model.t, domain=pyo.NonNegativeReals)

    # bounds
    for _, index in enumerate(ev_model.P_in_index):
        ev_model.P_in[index].bounds = (0, p_max[index[1]])
        ev_model.SoC[index].bounds = (0, soc_max[index[1]])

    for _, index in enumerate(ev_model.P_in_pl_index):
        ev_model.P_in_pl[index].bounds = (0, p_max[index[1]]/ev_model.pl_max)

    # fix SoC before arrival and from departure
    def soc_slack_rule(model, i, t):
        return model.SoC[i, t] + model.SoC_slack[t] >= departure_soc[t]

    ev_model.soc_slack_constraint = pyo.ConstraintList()

    for t in ev_model.t:
        arrival_after = arrival_time[t] - t_0
        arrival_index = np.floor(arrival_after.seconds / time_step_s)
        departure_after = departure_time[t] - t_0
        departure_index = np.floor(departure_after.seconds / time_step_s)

        for k in ev_model.k:
            # if before arrival
            if k < arrival_index:
                # set SoC at arrival and prevent charging
                ev_model.SoC[k, t].fix(arrival_soc[t])
                ev_model.P_in[k, t].fix(0)
            if k >= departure_index:
                # prevent charging after departure
                ev_model.P_in[k, t].fix(0)
            if k == departure_index:
                ev_model.soc_slack_constraint.add(ev_model.SoC[k, t] + ev_model.SoC_slack[t] >= departure_soc[t])

    # truck SoC evolution
    def soc_evolution(model, i, t):
        if i < model.k_max:
            return model.SoC[i+1, t] == model.SoC[i, t] \
                + sum(ev_model.eta_in[p] * ev_model.P_in_pl[i, t, p] for p in ev_model.pl)
        else:
            return pyo.Constraint.Skip

    ev_model.SoC_evo = pyo.Constraint(ev_model.k, ev_model.t, rule=soc_evolution)

    # POWER BALANCE LOCATION
    ev_model.P_total = pyo.Var(ev_model.k)
    ev_model.residual_load = pyo.Param(ev_model.k, initialize=residual_load)

    def truck_power_in_total_rule(model, i, t):
        # Sum up the piecewise linear elements charging the truck
        return model.P_in[i, t] == sum(model.P_in_pl[i, t, :])

    def power_balance_rule(model, i):
        # Sum up the total power consumption at the location
        return model.P_total[i] == sum(model.P_in[i, :]) + model.residual_load[i]

    def total_power_limit_rule(model, i):
        # Limit the total power consumption
        return model.P_total[i] <= p_total_max

    ev_model.truck_power_in_constraint = pyo.Constraint(ev_model.k, ev_model.t, rule=truck_power_in_total_rule)
    ev_model.power_balance_constraint = pyo.Constraint(ev_model.k, rule=power_balance_rule)
    ev_model.power_limit_constraint = pyo.Constraint(ev_model.k, rule=total_power_limit_rule)

    # OBJECTIVE
    # cost parameters
    ev_model.c = pyo.Param(ev_model.k, initialize=price)

    # cost function
    def cost_function(model):
        return pyo.summation(model.SoC) - pyo.summation(model.P_total) - 1000000 * pyo.summation(model.SoC_slack)

    ev_model.cost = pyo.Objective(rule=cost_function, sense=pyo.maximize)

    # SOLVE
    # select and run solver
    if LOCAL:
        ev_optimiser = pyo.SolverFactory('glpk', executable=PATH_TO_GLPK)
    else:
        ev_optimiser = pyo.SolverFactory('glpk')
    ev_solution = ev_optimiser.solve(ev_model)

    return ev_model, ev_optimiser, ev_solution


def input_data_csv_to_json(filename='./mockdata/100_cars.csv', outfile='./mockdata/100cars.json'):
    df = pd.read_csv(filename, index_col=0)
    df.to_json(outfile)
    return
