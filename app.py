from flask import Flask, request, render_template, jsonify
import json
import numpy as np
import pandas as pd
import mlflow

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

mlflow.set_tracking_uri("http://host.docker.internal:5000")

try:
    with open('current_model', 'r') as f:
        run_id = f.readline().rstrip()
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
except FileNotFoundError:
    f = open('current_model', 'w')
    f.close()
    run_id = ''
    model = None
except Exception:
    run_id = ''
    model = None

# model features
features = set(["hotel", "meal", "market_segment", "distribution_channel",
                "reserved_room_type", "deposit_type", "customer_type",
                "year", "month", "day", "lead_time", "arrival_date_week_number",
                "arrival_date_day_of_month", "stays_in_weekend_nights",
                "stays_in_week_nights", "adults", "children", "babies",
                "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
                "agent", "company", "adr", "required_car_parking_spaces", "total_of_special_requests"])


def set_model(id):
    '''
        Sets a new model(run id) for the api to use if that run id is a valid one

        Parameters:
            id (str): run id to be set
    '''
    global run_id, model

    try:
        run_id = id
        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    except:
        run_id = ''
        model = None


def model_check():
    '''
        Checks if run id of model to be used has changed on the file that tracks it.
        If so updates the run id and loaded model here on the api
    '''
    global run_id

    # checking if target model changed
    with open('current_model', 'r') as f:
        currently_set_run_id = f.readline().rstrip()

    # reloading the model if target model has changed
    if run_id != currently_set_run_id:
        set_model(currently_set_run_id)


@app.route('/', methods=['GET', 'POST'])
def predict():
    global run_id, model

    model_check()

    if request.method == 'POST':
        # From command line/post request json
        if request.data:
            if model is None:
                return jsonify({"error": "No model set. Please run 'python set_model.py [run_id]' to set a model."})

            print("Using model with run ID: ", run_id, flush=True)

            try: 
                data = request.get_json()
                if type(data) is str:
                    data = json.loads(data)
            except Exception as e:
                return jsonify({"error": "Could not json decode the input."})

            if type(data) is not list:
                data = [data]

            features_dict = {}
            for elem in data:
                input_features = set(elem.keys())
                if features != input_features:
                    return jsonify({"error": "Feature names mismatch"})
            
                for key, value in elem.items():
                    if key in features_dict:
                        features_dict[key].append(value)
                    else:
                        features_dict[key] = [value]

            df = pd.DataFrame(features_dict)

            pred = model.predict(df)

            return jsonify({"predictions": pred.tolist()})

        # From frontend form
        elif request.form:
            try:
                input_str = request.form.get('input-form').strip()
            except Exception:
                return jsonify({"error": "Something is wrong with the input data. Try setting data to json type in the header. If using curl add: -H \"Content-Type: application/json\""})

            if model is None:
                return render_template('index.html', prediction='', error="No model set. Please run 'python set_model.py [run_id]' to set a model.")

            print("Using model with run ID: ", run_id, flush=True)

            if request.form.get('input') == '':
                return render_template('index.html', prediction='', error="No input data provided")

            try: 
                data = json.loads(input_str)
            except Exception as e:
                return render_template('index.html', prediction='', error="Could not json decode the input.")

            if type(data) is not list:
                data = [data]

            features_dict = {}
            for elem in data:
                input_features = set(elem.keys())
                if features != input_features:
                    return render_template('index.html', prediction='', error="Feature names mismatch")
            
                for key, value in elem.items():
                    if key in features_dict:
                        features_dict[key].append(value)
                    else:
                        features_dict[key] = [value]

            df = pd.DataFrame(features_dict)

            pred = model.predict(df)
            return render_template('index.html', prediction=pred, error='')

        return jsonify({"error": "No input data provided"})

    return render_template('index.html', prediction='', error='')


if __name__ == '__main__':
    app.run(debug=True)