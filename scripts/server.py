#!/usr/bin/env python3

import os
from flask import Flask, jsonify, request

import json
from main import forecast


HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}


def flask_app():
    app = Flask(__name__)

    @app.route('/', methods=['GET'])
    def server_is_up():
        # print("success")
        return 'server is up - nice job! \n \n'

    @app.route('/forecast_price', methods=['POST'])
    def start():
        to_predict = request.json

        print(to_predict)
        pred = forecast(to_predict)
        return jsonify({"forecast price": pred.tolist()})
    return app

if __name__ == '__main__':
    app = flask_app()
    app.run(debug=True, host='0.0.0.0')
