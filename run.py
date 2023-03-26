import json
import os
import requests

import pandas as pd
import datetime as dt
from flask import Flask
from flask import request
from flask_cors import CORS
from google.cloud import secretmanager
from markupsafe import escape

""" UTIILITY FUNCTIONS """
import volpes_ev_utilities as vu
app = Flask(__name__)
CORS(app)

""" APP ROUTES """
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


@app.route("/ev_scheduler", methods=['GET', 'POST'])
def ev_dispatcher():
    # market_data_type = request.args.get("type", "not_provided")
    # data = "the provided query string, type={}, is not supported".format(escape(market_data_type))

    # TODO: Fix price source
    PRICEZONE = 'Oslo'

    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return '', 204, headers

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    # Receive truck data via post
    if request.method == 'POST':
        input_dict = request.json
    else:
        input_dict = json.load('./mockdata/100cars.json')

    # read fleet parameters
    df_EVs = pd.read_json(json.dumps(input_dict.get("data")))
    number_of_trucks = df_EVs.shape[1]
    p_max = df_EVs.loc['P_max', :].values.astype(float)
    soc_max = df_EVs.loc['SoC_max', :].values.astype(float)
    arrival_time = pd.to_datetime(df_EVs.loc['Arrival_time', :])
    departure_time = pd.to_datetime(df_EVs.loc['Departure_time', :])
    arrival_soc = df_EVs.loc['Arrival_SoC', :].values.astype(float)
    departure_soc = df_EVs.loc['Departure_SoC', :].values.astype(float)

    # set or read location parameters
    k_max = 24
    p_total_max = 650
    start_time = '2022-12-12 06:00'
    time_step_hours = 1

    if 'k_max' in input_dict:
        k_max = input_dict.get('k_max')
    if 'P_max_total' in input_dict:
        p_total_max = input_dict.get('P_max_total')
    if 'start_datetime' in input_dict:
        start_time = input_dict.get('start_datetime')
    if 'time_delta' in input_dict:
        time_delta = input_dict.get('time_delta')

    start_time = pd.to_datetime(start_time)
    end_time = start_time + dt.timedelta(hours=time_step_hours * k_max)
    time_index = pd.date_range(
        start=start_time,
        freq='{}S'.format(round(time_step_hours * 3600)),
        periods=k_max + 1
    )

    # read power prices
    # TODO: REMOVE LOCAL/ CLOUD FLAG BEFORE DEMO
    if not vu.LOCAL:
        secrets = secretmanager.SecretManagerServiceClient()
        url = secrets.access_secret_version(
            request={"name": "projects/115358684500/secrets/VOLPES_MARKET_DATA_URL/versions/1"}).payload.data.decode(
            "utf-8")
    else:
        url = 'https://volpes-energy-backend-fiiwhtua3a-ew.a.run.app/marketdata?type=day-ahead'

    params = dict(type='day-ahead')
    resp = requests.get(url=url, params=params)

    price_df = pd.DataFrame(resp.json())
    price_df.index = pd.date_range(start='2022/12/12 00:00', freq='h', periods=48, tz='CET')
    prices = price_df.loc[start_time:end_time, PRICEZONE].astype('float').values
    prices = prices[0: k_max + 1]

    # TODO: READ FROM DATA SOURCE/ BUILD ML PREDICTION [NOT FOR HACKATHON]
    residual_load = [
        219.85, 183.85, 172.75, 190.575, 200.875, 204.05, 168.15, 116.825, 95.6, 83.8, 76.075, 71.175, 61.175, 52.525, 48.375, 49.775, 59.5, 66.8, 80.55, 115.3, 188.475, 229.125, 232.425, 239.2,
        219.85]  # G0 winter 12:00-12:00
    residual_load = residual_load[0:k_max+1]

    # RUN MODELS
    ev_model, ev_opt, ev_solution = vu.ev_optimal_dispatch(
        price=prices,
        k_max=k_max, t_0=start_time,
        arrival_time=arrival_time, departure_time=departure_time, arrival_soc=arrival_soc, departure_soc=departure_soc,
        number_of_trucks=number_of_trucks, p_max=p_max, soc_max=soc_max, p_total_max=p_total_max,
        residual_load=residual_load)

    ev_dumb_model, ev_dumb_opt, ev_dumb_solution = vu.dumb_dispatch_model_EV_fleet(
        price=prices,
        k_max=k_max, t_0=start_time,
        arrival_time=arrival_time, departure_time=departure_time, arrival_soc=arrival_soc, departure_soc=departure_soc,
        number_of_trucks=number_of_trucks, p_max=p_max, soc_max=soc_max, p_total_max=p_total_max,
        residual_load=residual_load)

    # COLLECT OUTPUT
    # initiate data frames
    df_ev_dispatch = pd.DataFrame()
    # collect result data
    df_ev_dispatch['SoC'] = pd.DataFrame.from_dict(ev_model.SoC.extract_values(), orient='index',
                                                   columns=[str(ev_model.SoC)])
    df_ev_dispatch['P_in'] = pd.DataFrame.from_dict(ev_model.P_in.extract_values(), orient='index').abs()
    # convert tuple index to multi index
    df_ev_dispatch.index = pd.MultiIndex.from_tuples(df_ev_dispatch.index)
    # unstack
    df_ev_dispatch = df_ev_dispatch.unstack(level=-1)
    df_ev_dispatch['P_in'].columns = df_EVs.columns
    # rename columns
    df_ev_dispatch.columns = df_ev_dispatch.columns.set_levels(df_EVs.columns.values, level=1)
    # add residual load
    df_ev_dispatch['residual_load'] = pd.DataFrame.from_dict(ev_model.residual_load.extract_values(), orient='index')

    df_dumb = pd.DataFrame()
    df_dumb['P_in'] = pd.DataFrame.from_dict(ev_dumb_model.P_in.extract_values(), orient='index').abs()
    df_dumb.index = pd.MultiIndex.from_tuples(df_dumb.index)
    df_dumb = df_dumb.unstack(level=-1)

    savings = (1 - sum(df_ev_dispatch[['P_in']].sum(axis=1) * prices)/sum(df_dumb[['P_in']].sum(axis=1) * prices)) * 100

    # convert DateTime index (returned as Epoch) to formatted string
    df_ev_dispatch.index = time_index.strftime('%Y-%m-%d %H:%M')

    df_P_in = df_ev_dispatch['P_in']
    power_dict = {'power': {}}

    for i in df_P_in.index:
        power_dict['power'][i] = df_P_in.loc[i, :].to_list()

    return ({'power': json.loads(json.dumps(power_dict)),  # 'input': json.loads(df_EVs.to_json()),
            'unserved demand': json.loads(pd.DataFrame.from_dict(ev_model.SoC_slack.extract_values(), orient='index').to_json()),
            'savings': '{:2.1f}%'.format(savings),
            'secret url': url}, 200, headers)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
