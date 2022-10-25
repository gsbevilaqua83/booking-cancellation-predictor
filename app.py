from flask import Flask, request, render_template, jsonify
import json
import numpy as np
import pandas as pd
import mlflow

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

mlflow.set_tracking_uri("http://host.docker.internal:5000")

model_name = "booking_cancellation-cat_boost_classifier"

# checking and creating model in mlflow
mlclient = mlflow.MlflowClient()
if model_name not in set([model.name for model in mlclient.list_registered_models()]):
    try: # needed to deal with gunicorn workers concurrency
        mlclient.create_registered_model(model_name)
    except Exception:
        pass

model = None

# model features
features = set(["hotel", "meal", "market_segment", "distribution_channel",
                "reserved_room_type", "deposit_type", "customer_type",
                "year", "month", "day", "lead_time", "arrival_date_week_number",
                "arrival_date_day_of_month", "stays_in_weekend_nights",
                "stays_in_week_nights", "adults", "children", "babies",
                "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
                "agent", "company", "adr", "required_car_parking_spaces", "total_of_special_requests"])


def model_check():
    '''
        Finds the latest model in Production stage registered in MLFlow.
        If no model is in Production it sets model to None
    '''

    global model
    available_models = mlclient.search_registered_models()
    has_model_in_production = False
    for _model in available_models:
        if _model.name != model_name:
            continue

        for version in _model.latest_versions:
            if version.current_stage == 'Production':
                has_model_in_production = True
                break
        if has_model_in_production:
            break
    
    if has_model_in_production:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/Production"
        )
    else:
        model = None


@app.route('/', methods=['GET', 'POST'])
def predict():
    global model

    model_check()

    if request.method == 'POST':
        # From command line/post request json
        if request.data:
            if model is None:
                return jsonify({"error": "No registered model in production yet. If already trained a model, please register it using MLFlow UI and transition it to Production."})

            try: 
                data = request.get_json()
                if type(data) is str:
                    data = json.loads(data)
            except Exception as e:
                return jsonify({"error": "Could not json decode the input."})

            if type(data) is not list:
                data = [data]

            if data == []:
                return jsonify({"error": "Cannot predict on empty list."})

            features_dict = {}
            for elem in data:
                if type(elem) is not dict:
                    return jsonify({"error": f"Cannot predict on list of {type(elem)}."})

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
                return render_template('index.html', prediction='', error="No registered model in production yet. If already trained a model, please register it using MLFlow UI and transition it to Production.")

            if request.form.get('input') == '':
                return render_template('index.html', prediction='', error="No input data provided")

            try: 
                data = json.loads(input_str)
            except Exception as e:
                return render_template('index.html', prediction='', error="Could not json decode the input.")

            if type(data) is not list:
                data = [data]

            if data == []:
                return render_template('index.html', prediction='', error="Cannot predict on empty list.")

            features_dict = {}
            for elem in data:
                if type(elem) is not dict:
                    return render_template('index.html', prediction='', error=f"Cannot predict on list of {type(elem)}.")

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