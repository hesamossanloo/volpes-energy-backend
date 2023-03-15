import json
import os

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from flask import Flask
from flask import request
from google.cloud import secretmanager
from markupsafe import escape
from ortools.linear_solver import pywraplp

# PATH_TO_GLPK = 'C:\\Users\\tbors\\anaconda3\\Library\\bin\\glpsol.exe'

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Welcome to Volpes Energy. The API is working. Yay!"


@app.route("/marketdata")
def get_source():
    market_data_type = request.args.get("type", "not_provided")
    data = "the provided query string, type={}, is not supported".format(escape(market_data_type))

    secrets = secretmanager.SecretManagerServiceClient()
    SHOWING_WE_KNOW_HOW_TO_USE_SM = secrets.access_secret_version(
        request={"name": "projects/115358684500/secrets/SOMETHING_IMPORTANT/versions/1"}).payload.data.decode("utf-8")

    print(SHOWING_WE_KNOW_HOW_TO_USE_SM)
    f = open("mockdata/day-ahead.json", "rb")
    json_object = json.load(f)
    f.close()

    if not market_data_type:
        data = "the type query parameter is not provided."
    elif market_data_type == "day-ahead":
        data = json.dumps(json_object)
    return data


@app.route("/ev_scheduler")
def ev_optimal_dispatch():
    # market_data_type = request.args.get("type", "not_provided")
    # data = "the provided query string, type={}, is not supported".format(escape(market_data_type))

    df_trucks = pd.read_csv('./mockdata/truck_data-5_trucks.csv', index_col=0)
    number_of_trucks = df_trucks.shape[1]
    p_max = df_trucks.loc['P_max', :].values.astype(float)
    soc_max = df_trucks.loc['SoC_max', :].values.astype(float)
    arrival_time = pd.to_datetime(df_trucks.loc['Arrival_time', :])
    departure_time = pd.to_datetime(df_trucks.loc['Departure_time', :])
    arrival_soc = df_trucks.loc['Arrival_SoC', :].values.astype(float)
    departure_soc = df_trucks.loc['Departure_SoC', :].values.astype(float)

    eta_in = [0.9, 0.89, 0.88, 0.87]
    eta_out = [0.9, 0.89, 0.88, 0.87]
    time_step_s = 3600

    price = [
        475.74, 468.79, 491.36, 520.19, 542.35, 571.04, 541.34, 493.3,
        473.79, 396.22, 347.76, 304.12, 296.83, 279.18, 276.44, 268.4,
        273.55, 306.91, 356.83, 484.21, 555.18, 537.25, 503.05, 490.51,
        468.78]  # Oslo 2022-12-12 12:00 to 2022-12-13 12:00

    residual_load = [
        219.85, 183.85, 172.75, 190.575, 200.875, 204.05, 168.15, 116.825, 95.6, 83.8, 76.075, 71.175, 61.175, 52.525, 48.375, 49.775, 59.5, 66.8, 80.55, 115.3, 188.475, 229.125, 232.425, 239.2,
        219.85]  # G0 winter 12:00-12:00

    k_max = 24

    p_total_max = 450

    t_0 = pd.to_datetime('2022-12-12 12:00')

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

    ev_model = pyo.ConcreteModel()

    # INDEX SETS
    # number of time steps k
    ev_model.k_max = pyo.Param(initialize=k_max)
    ev_model.k = pyo.RangeSet(0, ev_model.k_max)

    # number of buses or trucks t
    ev_model.t_max = pyo.Param(initialize=number_of_trucks)
    ev_model.t = pyo.RangeSet(0, ev_model.t_max - 1)

    # number of piecewise linear elements for charging
    ev_model.pl_max = pyo.Param(initialize=4)  # HARD CODED
    ev_model.pl = pyo.RangeSet(0, ev_model.pl_max - 1)

    # TRUCK CHARGING MODEL
    # variables (power in/ out, state of charge)
    ev_model.P_in = pyo.Var(ev_model.k, ev_model.t, domain=pyo.NonNegativeReals)
    ev_model.P_in_pl = pyo.Var(ev_model.k, ev_model.t, ev_model.pl, domain=pyo.NonNegativeReals)
    ev_model.eta_in = pyo.Param(ev_model.pl, initialize=eta_in)

    ev_model.P_out = pyo.Var(ev_model.k, ev_model.t, domain=pyo.NonNegativeReals)
    ev_model.P_out_pl = pyo.Var(ev_model.k, ev_model.t, ev_model.pl, domain=pyo.NonNegativeReals)
    ev_model.eta_out = pyo.Param(ev_model.pl, initialize=eta_out)

    ev_model.SoC = pyo.Var(ev_model.k, ev_model.t, domain=pyo.NonNegativeReals)

    # bounds
    for _, index in enumerate(ev_model.P_in_index):
        ev_model.P_in[index].bounds = (0, p_max[index[1]])
        ev_model.P_out[index].bounds = (0, p_max[index[1]])
        ev_model.SoC[index].bounds = (0, soc_max[index[1]])

    for _, index in enumerate(ev_model.P_in_pl_index):
        ev_model.P_in_pl[index].bounds = (0, p_max[index[1]] / 4)  # HARD CODED
        ev_model.P_out_pl[index].bounds = (0, p_max[index[1]] / 4)  # HARD CODED

    # fix SoC before arrival and from departure
    for t in ev_model.t:
        arrival_after = arrival_time[t] - t_0
        arrival_index = np.floor(arrival_after.seconds / time_step_s)
        departure_after = departure_time[t] - t_0
        departure_index = np.floor(departure_after.seconds / time_step_s)

        for k in ev_model.k:
            # if before arrival
            if k < arrival_index:
                ev_model.SoC[k, t].fix(arrival_soc[t])
            if k >= departure_index:
                ev_model.SoC[k, t].fix(departure_soc[t])

    # ignore last charging time step (set charging power to zero)
    for t in ev_model.t:
        ev_model.P_in[k_max, t].fix(0)
        ev_model.P_out[k_max, t].fix(0)

    # truck SoC evolution
    def soc_evolution(model, i, t):
        if i < model.k_max:
            return model.SoC[i + 1, t] == model.SoC[i, t] \
                + sum(ev_model.eta_in[p] * ev_model.P_in_pl[i, t, p] for p in ev_model.pl) \
                - sum((1 / ev_model.eta_out[p]) * ev_model.P_out_pl[i, t, p] for p in ev_model.pl)
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
        return model.P_total[i] == sum(model.P_in[i, :]) - sum(model.P_out[i, :]) + model.residual_load[i]

    def total_power_limit_rule(model, i):
        # Limit the total power consumption
        return model.P_total[i] <= p_total_max

    ev_model.truck_power_in_constraint = pyo.Constraint(ev_model.k, ev_model.t, rule=truck_power_in_total_rule)
    ev_model.truck_power_out_constraint = pyo.Constraint(ev_model.k, ev_model.t, rule=truck_power_out_total_rule)
    ev_model.power_balance_constraint = pyo.Constraint(ev_model.k, rule=power_balance_rule)
    ev_model.power_limit_constraint = pyo.Constraint(ev_model.k, rule=total_power_limit_rule)

    # OBJECTIVE
    # cost parameters
    ev_model.c = pyo.Param(ev_model.k, initialize=price)

    # cost function
    def cost_function(model):
        return pyo.summation(model.c, model.P_total)

    ev_model.revenue = pyo.Objective(rule=cost_function, sense=pyo.minimize)

    # pretty print
    ev_optimiser = pyo.SolverFactory('glpk')
    # ev_optimiser = pyo.SolverFactory('glpk', executable=PATH_TO_GLPK)
    # ev_optimiser = pyo.SolverFactory('ipopt')

    ev_solution = ev_optimiser.solve(ev_model)

    # COLLECT OUTPUT
    # initiate data frames
    df_ev_dispatch = pd.DataFrame()

    # collect result data
    df_ev_dispatch['SoC'] = pd.DataFrame.from_dict(ev_model.SoC.extract_values(), orient='index',
                                                   columns=[str(ev_model.SoC)])
    df_ev_dispatch['P_in'] = pd.DataFrame.from_dict(ev_model.P_in.extract_values(), orient='index')
    df_ev_dispatch['P_out'] = pd.DataFrame.from_dict(ev_model.P_out.extract_values(), orient='index')

    # convert tuple index to multi index
    df_ev_dispatch.index = pd.MultiIndex.from_tuples(df_ev_dispatch.index)

    # unstack
    df_ev_dispatch = df_ev_dispatch.unstack(level=-1)

    # add residual load
    df_ev_dispatch['residual_load'] = pd.DataFrame.from_dict(ev_model.residual_load.extract_values(), orient='index')

    # write output
    # df_ev_dispatch[['P_in', 'residual_load']].to_csv('./out/test_P_in.csv')

    # plot...
    df = df_ev_dispatch['P_in']
    df = df.abs()

    return df.to_json()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
