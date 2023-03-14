import json
import os

from flask import Flask
from flask import request
from markupsafe import escape
from google.cloud import secretmanager

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Welcome to Volpes Energy. The API is working. Yay!"


@app.route("/marketdata")
def get_source():
    market_data_type = request.args.get("type", "not_provided")
    data = "the provided query string, type={}, is not supported".format(escape(market_data_type))

    secrets = secretmanager.SecretManagerServiceClient()
    SHOWING_WE_KNOW_HOW_TO_USE_SM = secrets.get_secret(request={"name": "projects/115358684500/secrets/SOMETHING_IMPORTANT/versions/1"})

    f = open("mockdata/day-ahead.json", "rb")
    json_object = json.load(f)
    f.close()

    if not market_data_type:
        data = "the type query parameter is not provided. Hint: {}".format((SHOWING_WE_KNOW_HOW_TO_USE_SM.response.payload.data.decode("UTF-8")))
    elif market_data_type == "day-ahead":
        data = json.dumps(json_object)
    return data


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
