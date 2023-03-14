import json
import os

from flask import Flask
from flask import request
from markupsafe import escape

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Welcome to Volpes Energy. The API is working. Yay!'


@app.route('/marketdata')
def get_source():
    market_data_type = request.args.get('type', 'not_provided')
    data = 'the provided query string, type={}, is not supported'.format(escape(market_data_type))

    f = open("mockdata/day-ahead.json", "rb")
    json_object = json.load(f)
    f.close()

    if not market_data_type:
        data = 'the type query parameter is not provided'
    elif market_data_type == 'day-ahead':
        data = json.dumps(json_object)
    return data


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
